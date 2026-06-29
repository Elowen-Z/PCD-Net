# v5/train_v6.py
"""
V6 训练主程序 — 配合 NoiseAwareDenoiserV6 + DenoiserLossV6

相对 train_v5 的关键差异:
  - 模型: V6 (UNet base_ch=32, 无 mask 乘法, 无 NLL, n_refine=3)
  - 损失: V6 (纯 MSE + freq + grad + detect BCE + VQ)
  - 默认 lr=1e-4, batch=16, epochs=30 (从头训)
  - 默认 save_dir = v5/checkpoints_v6_seed{S}

用法:
    python -m v5.train_v6 --seed 0
    python -m v5.train_v6 --seed 0 --epochs 30 --batch_size 16
    python -m v5.train_v6 --seed 0 --resume_from v5/checkpoints_v6_seed0/best_model_v6.pth
"""

from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from v3.dataset_v3   import STEADDatasetV3
from v5.dataset_mining import MiningDatasetV6
from v5.model_v6     import NoiseAwareDenoiserV6
from v5.loss_v6      import DenoiserLossV6


# ============================================================
#  默认配置
# ============================================================
DEFAULT_CONFIG = {
    "event_h5":   "D:/X/p_wave/data/chunk2.hdf5",
    "event_csv":  "D:/X/p_wave/data/chunk2.csv",
    "noise_h5":   "D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv":  "D:/X/p_wave/data/chunk1.csv",
    "raw_h5":     None,
    "raw_csv":    None,

    # 模型
    "z_dim":          128,
    "signal_len":     6000,
    "cond_len":       400,
    "num_prototypes": 16,
    "num_heads":      4,
    "n_refine":       3,
    "base_ch":        32,
    "vq_temperature": 0.3,

    # 训练
    "epochs":         30,
    "batch_size":     16,
    "lr":             1e-4,
    "weight_decay":   1e-4,
    "warmup_epochs":  2,
    "num_workers":    2,
    "val_frac":       0.1,
    "grad_clip":      1.0,
    "use_amp":        True,
    "viz_per_epoch":  4,

    # 损失权重
    "alpha_mse":          1.0,
    "alpha_freq":         0.20,
    "alpha_grad":         0.20,
    "alpha_detect":       1.0,
    "alpha_vq_commit":    0.25,
    "alpha_vq_diversity": 0.30,
    "valid_weight":       3.0,
    "bg_weight":          0.3,

    # 数据增强
    "snr_range":    (0.1, 20.0),
    "clean_prob":   0.10,
    "part_b_ratio": 0.0,   # V6 不使用自监督 part B（无监督训练目标 = 自身，会污染 MSE）
}


# ============================================================
#  工具
# ============================================================
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(csv_path, val_frac, seed, prefix):
    df       = pd.read_csv(csv_path, low_memory=False)
    val_df   = df.sample(frac=val_frac, random_state=seed)
    train_df = df.drop(val_df.index)
    Path("v5").mkdir(exist_ok=True)
    train_path = f"v5/{prefix}_train.csv"
    val_path   = f"v5/{prefix}_val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    print(f"[Split] Train={len(train_df)}, Val={len(val_df)}")
    return train_path, val_path


def compute_snr_batch(clean, residual, valid_mask):
    mask = valid_mask.unsqueeze(1)
    n = mask.sum(dim=[1, 2]) * clean.shape[1] + 1e-10
    sig = (clean    ** 2 * mask).sum(dim=[1, 2]) / n
    res = (residual ** 2 * mask).sum(dim=[1, 2]) / n + 1e-10
    return torch.clamp(10.0 * torch.log10(sig / res), -50, 50)


def cc_batch(pred, target):
    """逐样本 Pearson CC (在 Z 通道, 全长 6000), 返回 [B]"""
    p = pred[:, 2, :]    # [B, T]
    t = target[:, 2, :]
    p = p - p.mean(dim=-1, keepdim=True)
    t = t - t.mean(dim=-1, keepdim=True)
    num = (p * t).sum(dim=-1)
    den = torch.sqrt((p ** 2).sum(-1) * (t ** 2).sum(-1) + 1e-10)
    return num / den


def proto_stats(probs):
    with torch.no_grad():
        avg = probs.mean(0)
        ent = -(avg * (avg + 1e-8).log()).sum().item()
        active_k = int((avg > 1e-3).sum().item())
        max_p = avg.max().item()
    return {"proto_entropy": ent, "proto_active_k": active_k, "proto_max_p": max_p}


def _grads_finite(model):
    for p in model.parameters():
        if p.grad is None: continue
        if not torch.isfinite(p.grad).all(): return False
    return True


def _save_denoise_plot(x_in, x_out, y_target, valid_mask, save_path: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C, T = x_in.shape
    fig, axes = plt.subplots(C, 1, figsize=(12, 2.2 * C), sharex=True)
    if C == 1: axes = [axes]
    ch_names = ["E", "N", "Z"][:C]
    for c in range(C):
        ax = axes[c]
        ax.plot(x_in[c],     color="0.55", lw=0.6, label="Noisy")
        ax.plot(y_target[c], color="tab:green", lw=0.8, label="Target", alpha=0.85)
        ax.plot(x_out[c],    color="tab:red",  lw=0.8, label="Denoised", alpha=0.85)
        sig = np.where(valid_mask > 0.5)[0]
        if sig.size:
            ax.axvspan(sig.min(), sig.max(), color="yellow", alpha=0.10)
        ax.set_ylabel(ch_names[c]); ax.grid(alpha=0.3)
        if c == 0: ax.legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("Sample")
    fig.suptitle(save_path.stem, fontsize=10); fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight"); plt.close(fig)


# ============================================================
#  训练 / 验证
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device,
                    epoch, grad_clip, scaler=None, use_amp=False,
                    max_loss_for_backward=None):
    model.train()
    tot_loss, tot_n, skipped = 0.0, 0, 0
    detail_sum, proto_sum = {}, {}
    amp_dtype = torch.bfloat16

    for bi, batch in enumerate(loader):
        x          = batch["x"].to(device, non_blocking=True)
        y_clean    = batch["y_clean"].to(device, non_blocking=True)
        z_cond     = batch["z_cond"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        has_target = batch["has_target"].to(device, non_blocking=True)

        if not all(torch.isfinite(t).all() for t in [x, y_clean, z_cond]):
            skipped += 1; continue

        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                clean, quality, z_noise, aux = model(x, z_cond)
        except Exception as e:
            skipped += 1
            if skipped <= 3: print(f"  [warn] fwd fail batch {bi}: {e}")
            continue

        if not torch.isfinite(clean).all():
            skipped += 1; continue

        try:
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                loss, detail = criterion(
                    pred=clean, target=y_clean,
                    valid_mask=valid_mask, has_target=has_target,
                    det_mask=aux.get("det_mask"),
                    vq_commit=aux["vq_commit"],
                    vq_diversity=aux["vq_diversity"],
                )
        except Exception as e:
            skipped += 1
            if skipped <= 3: print(f"  [warn] loss fail batch {bi}: {e}")
            continue

        if not torch.isfinite(loss) or not loss.requires_grad:
            optimizer.zero_grad(set_to_none=True); skipped += 1; continue

        if max_loss_for_backward is not None and float(loss.detach()) > max_loss_for_backward:
            optimizer.zero_grad(set_to_none=True); skipped += 1
            if skipped <= 3:
                print(f"  [warn] skip high loss batch {bi}: loss={float(loss.detach()):.3f}")
            continue

        if use_amp and scaler is not None:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            if not _grads_finite(model):
                optimizer.zero_grad(set_to_none=True); scaler.update()
                skipped += 1; continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if not _grads_finite(model):
                optimizer.zero_grad(set_to_none=True); skipped += 1; continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = x.size(0)
        tot_loss += loss.item() * bs
        tot_n    += bs
        for k, v in detail.items():
            if np.isfinite(v):
                detail_sum[k] = detail_sum.get(k, 0.0) + v * bs

        ps = proto_stats(aux["prototype_probs"])
        for k, v in ps.items():
            proto_sum[k] = proto_sum.get(k, 0.0) + v * bs

        if (bi + 1) % 100 == 0:
            print(f"  Ep {epoch} [{bi+1}/{len(loader)}] "
                  f"loss={loss.item():.3f} "
                  f"mse={detail.get('mse', 0):.4f} "
                  f"freq={detail.get('freq', 0):.4f} "
                  f"det={detail.get('detect_bce', 0):.3f} "
                  f"K={ps['proto_active_k']} (skip={skipped})")

    if tot_n == 0:
        return float("nan"), {}, {}
    return (tot_loss / tot_n,
            {k: v / tot_n for k, v in detail_sum.items()},
            {k: v / tot_n for k, v in proto_sum.items()})


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device,
                       viz_dir: Optional[Path] = None, viz_n: int = 4):
    model.eval()
    tot_loss, tot_n = 0.0, 0
    detail_sum, proto_sum = {}, {}
    snr_in_list, snr_out_list, cc_list = [], [], []
    det_tp = det_tn = det_fp = det_fn = 0
    viz_saved = 0

    for batch in loader:
        x          = batch["x"].to(device)
        y_clean    = batch["y_clean"].to(device)
        z_cond     = batch["z_cond"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        has_target = batch["has_target"].to(device)

        if not all(torch.isfinite(t).all() for t in [x, y_clean, z_cond]):
            continue
        try:
            clean, quality, z_noise, aux = model(x, z_cond)
        except Exception:
            continue
        if not torch.isfinite(clean).all(): continue
        try:
            loss, detail = criterion(
                pred=clean, target=y_clean,
                valid_mask=valid_mask, has_target=has_target,
                det_mask=aux.get("det_mask"),
                vq_commit=aux["vq_commit"],
                vq_diversity=aux["vq_diversity"],
            )
        except Exception:
            continue
        if not torch.isfinite(loss): continue

        bs = x.size(0)
        tot_loss += loss.item() * bs; tot_n += bs
        for k, v in detail.items():
            if np.isfinite(v):
                detail_sum[k] = detail_sum.get(k, 0.0) + v * bs

        sup = has_target.bool()
        if sup.any():
            n_in  = x[sup]     - y_clean[sup]
            n_out = clean[sup] - y_clean[sup]
            try:
                snr_in  = compute_snr_batch(y_clean[sup], n_in,  valid_mask[sup])
                snr_out = compute_snr_batch(y_clean[sup], n_out, valid_mask[sup])
                snr_in_list.append(snr_in.cpu().numpy())
                snr_out_list.append(snr_out.cpu().numpy())
                cc = cc_batch(clean[sup], y_clean[sup])
                cc_list.append(cc.cpu().numpy())
            except Exception:
                pass

        ps = proto_stats(aux["prototype_probs"])
        for k, v in ps.items():
            proto_sum[k] = proto_sum.get(k, 0.0) + v * bs

        det_m = aux.get("det_mask")
        if det_m is not None and sup.any():
            pred_t  = (det_m[sup].squeeze(1) > 0.5)
            truth_t = (valid_mask[sup] > 0.5)
            det_tp += ( pred_t &  truth_t).sum().item()
            det_tn += (~pred_t & ~truth_t).sum().item()
            det_fp += ( pred_t & ~truth_t).sum().item()
            det_fn += (~pred_t &  truth_t).sum().item()

        if viz_dir is not None:
            viz_dir.mkdir(parents=True, exist_ok=True)
            if viz_saved < viz_n and sup.any():
                idx_sup = torch.nonzero(sup, as_tuple=False).squeeze(-1)
                for j in idx_sup.tolist():
                    if viz_saved >= viz_n: break
                    try:
                        _save_denoise_plot(
                            x[j].cpu().numpy(),
                            clean[j].cpu().numpy(),
                            y_clean[j].cpu().numpy(),
                            valid_mask[j].cpu().numpy(),
                            viz_dir / f"sample_{viz_saved:02d}.png",
                        )
                    except Exception as e:
                        print(f"  [viz] skip ({e})")
                    viz_saved += 1

    if tot_n == 0:
        return float("nan"), float("nan"), float("nan"), {}, {}
    avg     = tot_loss / tot_n
    detail  = {k: v / tot_n for k, v in detail_sum.items()}
    proto   = {k: v / tot_n for k, v in proto_sum.items()}

    if det_tp + det_tn + det_fp + det_fn > 0:
        tot_d   = det_tp + det_tn + det_fp + det_fn + 1e-10
        det_acc  = (det_tp + det_tn) / tot_d
        det_fpr  = det_fp / (det_fp + det_tn + 1e-10)
        det_prec = det_tp / (det_tp + det_fp + 1e-10)
        det_rec  = det_tp / (det_tp + det_fn + 1e-10)
        det_f1   = 2 * det_prec * det_rec / (det_prec + det_rec + 1e-10)
    else:
        det_acc = det_fpr = det_prec = det_rec = det_f1 = 0.0
    detail.update({"det_acc": det_acc, "det_fpr": det_fpr,
                   "det_prec": det_prec, "det_rec": det_rec, "det_f1": det_f1})

    snr_in  = np.concatenate(snr_in_list)  if snr_in_list  else np.array([])
    snr_out = np.concatenate(snr_out_list) if snr_out_list else np.array([])
    cc_arr  = np.concatenate(cc_list)      if cc_list      else np.array([])
    valid   = (np.isfinite(snr_in) & np.isfinite(snr_out)
               & (np.abs(snr_in) < 50) & (np.abs(snr_out) < 50))
    snr_gain = float((snr_out[valid] - snr_in[valid]).mean()) if valid.any() else float("nan")
    cc_mean  = float(np.nanmean(cc_arr)) if cc_arr.size else float("nan")

    print(f"  Val Loss={avg:.4f}  ΔSNR={snr_gain:+.2f}dB  "
          f"CC={cc_mean:.4f}  K={proto.get('proto_active_k', 0):.1f}")
    print(f"  [Detect] Acc={det_acc:.4f}  Prec={det_prec:.4f}  "
          f"Rec={det_rec:.4f}  F1={det_f1:.4f}")
    return avg, snr_gain, cc_mean, detail, proto


# ============================================================
#  主入口
# ============================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_prototypes", type=int, default=None)
    parser.add_argument("--n_refine", type=int, default=None)
    parser.add_argument("--base_ch", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    # 新增：数据路径参数
    parser.add_argument("--event_h5", type=str, default=None)
    parser.add_argument("--event_csv", type=str, default=None)
    parser.add_argument("--noise_h5", type=str, default=None)
    parser.add_argument("--noise_csv", type=str, default=None)
    parser.add_argument("--raw_h5", type=str, default=None)
    parser.add_argument("--raw_csv", type=str, default=None)
    # 数据集类型: stead = STEAD/chunk2/non_natural (固定长度) ; mining = LN_mining (矿震, 变长+多采样率)
    parser.add_argument("--dataset_type", type=str, default="stead",
                        choices=["stead", "mining"])
    args = parser.parse_args()


    cfg = dict(DEFAULT_CONFIG)
    cfg["seed"] = args.seed
    if args.epochs:         cfg["epochs"]         = args.epochs
    if args.batch_size:     cfg["batch_size"]     = args.batch_size
    if args.lr:             cfg["lr"]             = args.lr
    if args.num_prototypes: cfg["num_prototypes"] = args.num_prototypes
    if args.n_refine is not None: cfg["n_refine"] = args.n_refine
    if args.base_ch is not None:     cfg["base_ch"]     = args.base_ch
    if args.num_workers is not None: cfg["num_workers"] = args.num_workers
    cfg["save_dir"] = args.save_dir or f"v5/checkpoints_v6_seed{args.seed}"
    # 新增：用命令行参数覆盖数据路径
    if args.event_h5:   cfg["event_h5"]   = args.event_h5
    if args.event_csv:  cfg["event_csv"]  = args.event_csv
    if args.noise_h5:   cfg["noise_h5"]   = args.noise_h5
    if args.noise_csv:  cfg["noise_csv"]  = args.noise_csv
    if args.raw_h5:     cfg["raw_h5"]     = args.raw_h5
    if args.raw_csv:    cfg["raw_csv"]    = args.raw_csv

    set_seed(cfg["seed"])
    Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg["save_dir"]) / "config.json", "w") as f:
        json.dump({k: str(v) for k, v in cfg.items()}, f, indent=2)

    train_csv, val_csv = split_dataset(
        cfg["event_csv"], cfg["val_frac"], cfg["seed"],
        prefix=f"v6_seed{cfg['seed']}",
    )

    DS_CLS = MiningDatasetV6 if args.dataset_type == "mining" else STEADDatasetV3
    print(f"[INFO] dataset_type = {args.dataset_type}  -> {DS_CLS.__name__}")

    def make_ds(csv_path, aug):
        return DS_CLS(
            event_h5_path  = cfg["event_h5"],
            event_csv_path = csv_path,
            noise_h5_path  = cfg["noise_h5"],
            noise_csv_path = cfg["noise_csv"],
            raw_h5_path    = cfg["raw_h5"],
            raw_csv_path   = cfg["raw_csv"],
            signal_len     = cfg["signal_len"],
            cond_len       = cfg["cond_len"],
            snr_range      = cfg["snr_range"],
            clean_prob     = cfg["clean_prob"] if aug else 0.0,
            part_b_ratio   = cfg["part_b_ratio"] if aug else 0.0,
            seed           = cfg["seed"],
        )

    train_ds = make_ds(train_csv, aug=True)
    val_ds   = make_ds(val_csv,   aug=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg["num_workers"] > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg["num_workers"] > 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[INFO] Device: {device}  Seed: {cfg['seed']}  AMP={cfg['use_amp']}")

    model = NoiseAwareDenoiserV6(
        in_ch=3,
        z_dim=cfg["z_dim"],
        cond_len=cfg["cond_len"],
        num_prototypes=cfg["num_prototypes"],
        num_heads=cfg["num_heads"],
        n_refine=cfg["n_refine"],
        base_ch=cfg["base_ch"],
        vq_temperature=cfg["vq_temperature"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] V6 params: {n_params/1e6:.2f}M | "
          f"base_ch={cfg['base_ch']} | n_refine={cfg['n_refine']}")

    ckpt_path = Path(cfg["save_dir"]) / "best_model_v6.pth"
    if args.resume_from and Path(args.resume_from).exists():
        try:
            sd = torch.load(args.resume_from, map_location=device)
            sd = sd["model_state_dict"] if isinstance(sd, dict) and "model_state_dict" in sd else sd
            model.load_state_dict(sd, strict=False)
            print(f"[INFO] resumed (non-strict) from {args.resume_from}")
        except Exception as e:
            print(f"[WARN] resume_from failed ({e})")
    elif ckpt_path.exists():
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"[INFO] resumed from {ckpt_path}")
        except Exception as e:
            print(f"[INFO] resume skipped ({e})")

    criterion = DenoiserLossV6(
        alpha_mse          = cfg["alpha_mse"],
        alpha_freq         = cfg["alpha_freq"],
        alpha_grad         = cfg["alpha_grad"],
        alpha_detect       = cfg["alpha_detect"],
        alpha_vq_commit    = cfg["alpha_vq_commit"],
        alpha_vq_diversity = cfg["alpha_vq_diversity"],
        valid_weight       = cfg["valid_weight"],
        bg_weight          = cfg["bg_weight"],
    )

    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    _use_amp = cfg["use_amp"] and device.type == "cuda"
    _need_scaler = _use_amp and not torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=_need_scaler)

    def lr_lambda(epoch):
        if epoch < cfg["warmup_epochs"]:
            return (epoch + 1) / cfg["warmup_epochs"]
        progress = (epoch - cfg["warmup_epochs"]) \
                   / max(1, cfg["epochs"] - cfg["warmup_epochs"])
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float("inf")
    best_cc  = -1.0
    history  = []

    print("\n" + "=" * 60)
    print(f"  V6 training | seed={cfg['seed']} | base_ch={cfg['base_ch']} | "
          f"n_refine={cfg['n_refine']}")
    print("=" * 60)

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\n[Epoch {epoch}/{cfg['epochs']}]  "
              f"LR={optimizer.param_groups[0]['lr']:.2e}")

        train_loss, t_detail, t_proto = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, cfg["grad_clip"],
            scaler=scaler, use_amp=_use_amp,
        )
        val_loss, val_gain, val_cc, v_detail, v_proto = validate_one_epoch(
            model, val_loader, criterion, device,
            viz_dir=Path(cfg["save_dir"]) / f"viz_epoch{epoch:02d}",
            viz_n=cfg["viz_per_epoch"],
        )
        scheduler.step()

        ema_usage = model.vq.ema_usage.detach().cpu().numpy().tolist()
        history.append({
            "epoch":         epoch,
            "train_loss":    train_loss if np.isfinite(train_loss) else None,
            "val_loss":      val_loss   if np.isfinite(val_loss)   else None,
            "val_snr_gain":  val_gain   if np.isfinite(val_gain)   else None,
            "val_cc":        val_cc     if np.isfinite(val_cc)     else None,
            **{f"train_{k}": v for k, v in t_detail.items()},
            **{f"val_{k}":   v for k, v in v_detail.items()},
            **{f"train_{k}": v for k, v in t_proto.items()},
            **{f"val_{k}":   v for k, v in v_proto.items()},
            "ema_usage": ema_usage,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # 以 val_cc 为主指标保存 best（更直接反映波形相关性）
        if np.isfinite(val_cc) and val_cc > best_cc:
            best_cc = val_cc
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [best] saved val_cc={val_cc:.4f} val_loss={val_loss:.4f}")

        # 每个 epoch 都存 last.pth 以便从中断处恢复
        torch.save(model.state_dict(),
                   Path(cfg["save_dir"]) / "last.pth")

        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                       Path(cfg["save_dir"]) / f"ckpt_epoch{epoch}.pth")

        with open(Path(cfg["save_dir"]) / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        tl = f"{train_loss:.4f}" if np.isfinite(train_loss) else "nan"
        vl = f"{val_loss:.4f}"   if np.isfinite(val_loss)   else "nan"
        print(f"  Train={tl} | Val={vl} | BestCC={best_cc:.4f}")

    print(f"\n[done] best val_cc = {best_cc:.4f}  best val_loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
