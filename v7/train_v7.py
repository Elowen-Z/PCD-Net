"""Train the paper-aligned PCD-Net V7."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

# This must be set before importing torch to avoid unstable cuDNN v8 plans on
# affected Windows/PyTorch/CUDA combinations.
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from v3.dataset_v3 import STEADDatasetV3
from v5.dataset_mining import MiningDatasetV6
from v5.train_v6 import cc_batch, compute_snr_batch, set_seed, split_dataset
from v7.loss_v7 import DenoiserLossV7
from v7.model_v7 import NoiseAwareDenoiserV7


DEFAULT_CONFIG = {
    "event_h5": "D:/X/p_wave/data/chunk2.hdf5",
    "event_csv": "D:/X/p_wave/data/chunk2.csv",
    "noise_h5": "D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv": "D:/X/p_wave/data/chunk1.csv",
    "raw_h5": None,
    "raw_csv": None,
    "signal_len": 6000,
    "cond_len": 400,
    "z_dim": 128,
    "num_prototypes": 16,
    "top_m": 4,
    "num_heads": 4,
    "n_refine": 3,
    "base_ch": 32,
    "vq_temperature": 0.3,
    "epochs": 30,
    "batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "val_frac": 0.1,
    "grad_clip": 1.0,
    "use_amp": True,
    "amp_dtype": "float16",
    "snr_range": (0.1, 20.0),
    "clean_prob": 0.10,
    "part_b_ratio": 0.0,
    "alpha_mse": 1.0,
    "alpha_freq": 0.20,
    "alpha_grad": 0.20,
    "alpha_detect": 1.0,
    "alpha_vq_commit": 0.25,
    "alpha_vq_diversity": 0.30,
    "alpha_sparse": 0.05,
    "alpha_balance": 0.02,
    "alpha_quality": 0.20,
    "alpha_intermediate": 0.10,
    "valid_weight": 3.0,
    "bg_weight": 0.3,
    "stop_threshold": 0.95,
    "min_refine_steps": 1,
    "print_every": 0,
    "checkpoint_every": 0,
    "val_batch_size": None,
    "val_num_workers": 0,
    "pin_memory": True,
    "st_window": 100,
    "train_samples": None,
    "val_samples": None,
}


def configure_cuda_backend(cuda_safe: bool) -> None:
    if not torch.cuda.is_available():
        return
    if cuda_safe:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        print(
            "[CUDA safe] cuDNN, Flash SDP, memory-efficient SDP and TF32 "
            "are disabled."
        )


def _move_batch(batch, device):
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("x", "y_clean", "z_cond", "valid_mask", "has_target")
    }


def _loss_call(criterion, output, batch):
    clean, quality, _, aux = output
    return criterion(
        pred=clean,
        target=batch["y_clean"],
        valid_mask=batch["valid_mask"],
        has_target=batch["has_target"],
        det_mask=aux.get("det_mask"),
        quality=quality,
        quality_logits=aux.get("quality_logits"),
        prototype_probs=aux.get("prototype_probs"),
        sparse_probs=aux.get("sparse_probs"),
        vq_commit=aux.get("vq_commit"),
        vq_diversity=aux.get("vq_diversity"),
        refine_history=aux.get("refine_history"),
        quality_logits_history=aux.get("quality_logits_history"),
    )


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    cfg,
    scaler,
    epoch,
    recovery_path,
):
    model.train()
    total, count, skipped = 0.0, 0, 0
    details = {}
    amp = cfg["use_amp"] and device.type == "cuda"
    amp_dtype = (
        torch.float16
        if cfg["amp_dtype"] == "float16"
        else torch.bfloat16
    )

    for batch_index, batch_raw in enumerate(loader, start=1):
        batch = _move_batch(batch_raw, device)
        if not all(
            torch.isfinite(batch[k]).all()
            for k in ("x", "y_clean", "z_cond")
        ):
            skipped += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp, dtype=amp_dtype):
            output = model(batch["x"], batch["z_cond"])
            loss, detail = _loss_call(criterion, output, batch)

        if not torch.isfinite(loss):
            skipped += 1
            continue

        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

        bs = batch["x"].size(0)
        total += float(loss.detach()) * bs
        count += bs
        for key, value in detail.items():
            if np.isfinite(value):
                details[key] = details.get(key, 0.0) + value * bs
        if cfg["print_every"] > 0 and batch_index % cfg["print_every"] == 0:
            print(
                f"  [epoch {epoch:02d} batch {batch_index:05d}/"
                f"{len(loader):05d}] loss={total / max(count, 1):.4f} "
                f"skipped={skipped}",
                flush=True,
            )
        if (
            cfg["checkpoint_every"] > 0
            and batch_index % cfg["checkpoint_every"] == 0
        ):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": epoch,
                    "batch": batch_index,
                    "epoch_complete": False,
                    "config": cfg,
                },
                recovery_path,
            )
            print(
                f"  [recovery] saved epoch={epoch} batch={batch_index}",
                flush=True,
            )

    mean_detail = {k: v / max(count, 1) for k, v in details.items()}
    return total / max(count, 1), mean_detail, skipped


@torch.no_grad()
def validate(model, loader, criterion, device, st_window=100):
    model.eval()
    total, count = 0.0, 0
    gains, correlations, quality_values, fidelity_values = [], [], [], []
    adaptive_gains, adaptive_correlations = [], []
    rmses, prds, st_maes, event_st_maes = [], [], [], []
    adaptive_rmses, adaptive_prds = [], []
    adaptive_st_maes, adaptive_event_st_maes = [], []
    selected_mass, selected_entropy = [], []
    effective_steps, stopped_early = [], []

    for batch_raw in loader:
        batch = _move_batch(batch_raw, device)
        # Always unfold every pass for stable checkpoint selection. Adaptive
        # deployment metrics are reconstructed from the per-pass histories.
        output = model(batch["x"], batch["z_cond"], adaptive_stop=False)
        clean, quality, _, aux = output
        loss, _ = _loss_call(criterion, output, batch)
        if not torch.isfinite(loss):
            continue

        supervised = batch["has_target"].bool()
        if not supervised.any():
            continue
        pred = clean[supervised]
        target = batch["y_clean"][supervised]
        noisy = batch["x"][supervised]
        mask = batch["valid_mask"][supervised]

        snr_in = compute_snr_batch(target, noisy - target, mask)
        snr_out = compute_snr_batch(target, pred - target, mask)
        gains.extend((snr_out - snr_in).detach().cpu().tolist())
        correlations.extend(cc_batch(pred, target).detach().cpu().tolist())
        error = pred - target
        rmse = torch.sqrt(torch.mean(error ** 2, dim=(1, 2)))
        reference_rms = torch.sqrt(
            torch.mean(target ** 2, dim=(1, 2))
        ).clamp_min(1e-8)
        rmses.extend(rmse.detach().cpu().tolist())
        prds.extend((100.0 * rmse / reference_rms).detach().cpu().tolist())

        def collect_st_mae(prediction, output_all, output_event):
            absolute_error = (prediction - target).abs().mean(dim=1)
            window = st_window
            length = absolute_error.size(-1)
            padded_length = ((length + window - 1) // window) * window
            if padded_length != length:
                absolute_error = torch.nn.functional.pad(
                    absolute_error, (0, padded_length - length)
                )
                mask_windows = torch.nn.functional.pad(
                    mask, (0, padded_length - length)
                )
            else:
                mask_windows = mask
            window_error = absolute_error.unfold(
                -1, window, window
            ).mean(-1)
            window_mask = mask_windows.unfold(
                -1, window, window
            ).mean(-1)
            output_all.extend(window_error.mean(-1).detach().cpu().tolist())
            event_weight = (window_mask > 0).float()
            event_value = (
                (window_error * event_weight).sum(-1)
                / event_weight.sum(-1).clamp_min(1.0)
            )
            output_event.extend(event_value.detach().cpu().tolist())

        collect_st_mae(pred, st_maes, event_st_maes)

        adaptive_pred = pred.clone()
        sample_steps = torch.full(
            (pred.size(0),),
            len(aux["refine_history"]),
            device=device,
            dtype=torch.long,
        )
        undecided = torch.ones(pred.size(0), device=device, dtype=torch.bool)
        for step, (step_pred, step_quality) in enumerate(
            zip(aux["refine_history"], aux["quality_history"]), start=1
        ):
            if step < model.min_refine_steps:
                continue
            q_step = step_quality[supervised].squeeze(-1)
            stop_now = undecided & (q_step >= model.stop_threshold)
            if stop_now.any():
                adaptive_pred[stop_now] = step_pred[supervised][stop_now]
                sample_steps[stop_now] = step
                undecided[stop_now] = False

        adaptive_snr_out = compute_snr_batch(
            target, adaptive_pred - target, mask
        )
        adaptive_gains.extend(
            (adaptive_snr_out - snr_in).detach().cpu().tolist()
        )
        adaptive_correlations.extend(
            cc_batch(adaptive_pred, target).detach().cpu().tolist()
        )
        adaptive_error = adaptive_pred - target
        adaptive_rmse = torch.sqrt(
            torch.mean(adaptive_error ** 2, dim=(1, 2))
        )
        adaptive_rmses.extend(adaptive_rmse.detach().cpu().tolist())
        adaptive_prds.extend(
            (100.0 * adaptive_rmse / reference_rms).detach().cpu().tolist()
        )
        collect_st_mae(
            adaptive_pred, adaptive_st_maes, adaptive_event_st_maes
        )

        q = quality[supervised].squeeze(-1)
        err = ((pred - target) ** 2).mean(dim=(1, 2))
        power = (target ** 2).mean(dim=(1, 2)).clamp_min(1e-8)
        fidelity = torch.exp(-(err / power))
        quality_values.extend(q.detach().cpu().tolist())
        fidelity_values.extend(fidelity.detach().cpu().tolist())

        selected_mass.extend(
            aux["selected_mass"][supervised].detach().cpu().tolist()
        )
        sparse = aux["sparse_probs"][supervised].clamp_min(1e-8)
        entropy = -(sparse * sparse.log()).sum(-1)
        selected_entropy.extend(entropy.detach().cpu().tolist())
        effective_steps.extend(sample_steps.detach().cpu().tolist())
        stopped_early.extend(
            (sample_steps < len(aux["refine_history"]))
            .float()
            .detach()
            .cpu()
            .tolist()
        )

        bs = int(supervised.sum())
        total += float(loss) * bs
        count += bs

    q_corr = float("nan")
    if len(quality_values) > 1 and np.std(quality_values) > 0:
        q_corr = float(np.corrcoef(quality_values, fidelity_values)[0, 1])
    return {
        "val_loss": total / max(count, 1),
        "val_gain": float(np.mean(gains)) if gains else float("nan"),
        "val_cc": float(np.mean(correlations)) if correlations else float("nan"),
        "val_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "val_prd": float(np.mean(prds)) if prds else float("nan"),
        "val_st_mae": float(np.mean(st_maes)) if st_maes else float("nan"),
        "val_event_st_mae": (
            float(np.mean(event_st_maes))
            if event_st_maes else float("nan")
        ),
        "adaptive_gain": (
            float(np.mean(adaptive_gains))
            if adaptive_gains else float("nan")
        ),
        "adaptive_cc": (
            float(np.mean(adaptive_correlations))
            if adaptive_correlations else float("nan")
        ),
        "adaptive_rmse": (
            float(np.mean(adaptive_rmses))
            if adaptive_rmses else float("nan")
        ),
        "adaptive_prd": (
            float(np.mean(adaptive_prds))
            if adaptive_prds else float("nan")
        ),
        "adaptive_st_mae": (
            float(np.mean(adaptive_st_maes))
            if adaptive_st_maes else float("nan")
        ),
        "adaptive_event_st_mae": (
            float(np.mean(adaptive_event_st_maes))
            if adaptive_event_st_maes else float("nan")
        ),
        "quality_corr": q_corr,
        "selected_mass": float(np.mean(selected_mass)) if selected_mass else 0.0,
        "selected_entropy": (
            float(np.mean(selected_entropy)) if selected_entropy else 0.0
        ),
        "effective_steps": (
            float(np.mean(effective_steps)) if effective_steps else 0.0
        ),
        "early_stop_rate": (
            float(np.mean(stopped_early)) if stopped_early else 0.0
        ),
    }


def make_loaders(cfg, dataset_type):
    train_csv, val_csv = split_dataset(
        cfg["event_csv"],
        cfg["val_frac"],
        cfg["seed"],
        prefix=f"v7_seed{cfg['seed']}",
    )
    dataset_cls = MiningDatasetV6 if dataset_type == "mining" else STEADDatasetV3

    def make_dataset(csv_path, training):
        return dataset_cls(
            event_h5_path=cfg["event_h5"],
            event_csv_path=csv_path,
            noise_h5_path=cfg["noise_h5"],
            noise_csv_path=cfg["noise_csv"],
            raw_h5_path=cfg["raw_h5"],
            raw_csv_path=cfg["raw_csv"],
            signal_len=cfg["signal_len"],
            cond_len=cfg["cond_len"],
            snr_range=cfg["snr_range"],
            clean_prob=cfg["clean_prob"] if training else 0.0,
            part_b_ratio=cfg["part_b_ratio"] if training else 0.0,
            seed=cfg["seed"],
        )

    train_ds = make_dataset(train_csv, True)
    val_ds = make_dataset(val_csv, False)
    if cfg["train_samples"] is not None:
        train_count = min(int(cfg["train_samples"]), len(train_ds))
        train_ds = Subset(train_ds, range(train_count))
        print(f"[subset] train_samples={train_count}")
    if cfg["val_samples"] is not None:
        val_count = min(int(cfg["val_samples"]), len(val_ds))
        val_ds = Subset(val_ds, range(val_count))
        print(f"[subset] val_samples={val_count}")
    train_workers = cfg["num_workers"]
    val_workers = cfg["val_num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=train_workers,
        pin_memory=cfg["pin_memory"] and torch.cuda.is_available(),
        persistent_workers=train_workers > 0,
    )
    val_batch_size = cfg["val_batch_size"] or cfg["batch_size"]
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=False,
        persistent_workers=val_workers > 0,
    )
    return train_loader, val_loader


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["stead", "mining"], default="stead")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--num_prototypes", type=int)
    parser.add_argument("--top_m", type=int)
    parser.add_argument("--n_refine", type=int)
    parser.add_argument("--base_ch", type=int)
    parser.add_argument("--alpha_sparse", type=float)
    parser.add_argument("--alpha_balance", type=float)
    parser.add_argument("--alpha_quality", type=float)
    parser.add_argument("--alpha_intermediate", type=float)
    parser.add_argument("--stop_threshold", type=float)
    parser.add_argument("--min_refine_steps", type=int)
    parser.add_argument("--print_every", type=int)
    parser.add_argument("--checkpoint_every", type=int)
    parser.add_argument("--val_batch_size", type=int)
    parser.add_argument("--val_num_workers", type=int)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--st_window", type=int)
    parser.add_argument("--train_samples", type=int)
    parser.add_argument("--val_samples", type=int)
    parser.add_argument("--no_sparse_selection", action="store_true")
    parser.add_argument("--no_prototypes", action="store_true")
    parser.add_argument("--no_cross_attn", action="store_true")
    parser.add_argument("--no_quality_head", action="store_true")
    parser.add_argument("--no_residual_feedback", action="store_true")
    parser.add_argument("--no_adaptive_inference", action="store_true")
    parser.add_argument("--event_h5")
    parser.add_argument("--event_csv")
    parser.add_argument("--noise_h5")
    parser.add_argument("--noise_csv")
    parser.add_argument("--raw_h5")
    parser.add_argument("--raw_csv")
    parser.add_argument("--save_dir")
    parser.add_argument("--init_from")
    parser.add_argument("--resume_from")
    parser.add_argument("--cuda_safe", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--amp_dtype",
        choices=["float16", "bfloat16"],
        default=None,
    )
    return parser


def main():
    args = build_parser().parse_args()
    cfg = dict(DEFAULT_CONFIG)
    cfg["seed"] = args.seed
    for key in (
        "epochs", "batch_size", "lr", "num_workers", "num_prototypes",
        "top_m", "n_refine", "base_ch", "alpha_sparse", "alpha_balance",
        "alpha_quality", "alpha_intermediate", "stop_threshold",
        "min_refine_steps", "print_every", "checkpoint_every", "event_h5",
        "val_batch_size", "val_num_workers", "st_window", "event_csv",
        "train_samples", "val_samples", "noise_h5",
        "noise_csv", "raw_h5", "raw_csv",
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    cfg["use_sparse_selection"] = not args.no_sparse_selection
    cfg["use_prototypes"] = not args.no_prototypes
    cfg["use_cross_attn"] = not args.no_cross_attn
    cfg["use_quality_head"] = not args.no_quality_head
    cfg["use_residual_feedback"] = not args.no_residual_feedback
    cfg["adaptive_inference"] = not args.no_adaptive_inference
    cfg["use_amp"] = cfg["use_amp"] and not args.no_amp
    if args.amp_dtype is not None:
        cfg["amp_dtype"] = args.amp_dtype
    cfg["cuda_safe"] = args.cuda_safe
    cfg["pin_memory"] = not args.no_pin_memory
    cfg["save_dir"] = args.save_dir or f"v7/checkpoints_v7_seed{args.seed}"

    if cfg["top_m"] > cfg["num_prototypes"]:
        raise ValueError("top_m cannot exceed num_prototypes")

    configure_cuda_backend(args.cuda_safe)
    set_seed(cfg["seed"])
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    train_loader, val_loader = make_loaders(cfg, args.dataset_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoiseAwareDenoiserV7(
        in_ch=3,
        z_dim=cfg["z_dim"],
        cond_len=cfg["cond_len"],
        num_prototypes=cfg["num_prototypes"],
        top_m=cfg["top_m"],
        num_heads=cfg["num_heads"],
        n_refine=cfg["n_refine"],
        base_ch=cfg["base_ch"],
        vq_temperature=cfg["vq_temperature"],
        use_prototypes=cfg["use_prototypes"],
        use_sparse_selection=cfg["use_sparse_selection"],
        use_cross_attn=cfg["use_cross_attn"],
        use_quality_head=cfg["use_quality_head"],
        use_residual_feedback=cfg["use_residual_feedback"],
        adaptive_inference=cfg["adaptive_inference"],
        stop_threshold=cfg["stop_threshold"],
        min_refine_steps=cfg["min_refine_steps"],
    ).to(device)

    if args.init_from and args.resume_from:
        raise ValueError("use only one of --init_from and --resume_from")
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = model.load_v6_state_dict(state)
        print(f"[init] missing={len(missing)} unexpected={len(unexpected)}")

    criterion = DenoiserLossV7(
        **{
            key: cfg[key]
            for key in (
                "alpha_mse", "alpha_freq", "alpha_grad", "alpha_detect",
                "alpha_vq_commit", "alpha_vq_diversity", "alpha_sparse",
                "alpha_balance", "alpha_quality", "valid_weight", "bg_weight",
                "alpha_intermediate",
            )
        }
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg["epochs"], 1)
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg["use_amp"] and device.type == "cuda"
    )

    history_path = save_dir / "history.json"
    history, best_cc, start_epoch = [], -math.inf, 1
    resume_batch = None
    resume_epoch_complete = True
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=True)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        completed = bool(checkpoint.get("epoch_complete", True))
        resume_epoch_complete = completed
        resume_batch = checkpoint.get("batch")
        start_epoch = int(checkpoint.get("epoch", 0)) + (1 if completed else 0)
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        if history:
            best_cc = max(
                float(row.get("val_cc", -math.inf)) for row in history
            )
        else:
            best_cc = float(
                checkpoint.get("metrics", {}).get("val_cc", -math.inf)
            )
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            # Older V7 checkpoints already contain the correct current LR in
            # optimizer state. Continue the recursive cosine schedule from
            # that position without performing an extra scheduler step.
            scheduler.last_epoch = start_epoch - 1
            scheduler._step_count = start_epoch
            scheduler._last_lr = [
                group["lr"] for group in optimizer.param_groups
            ]
        if "scaler_state_dict" in checkpoint and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(
            f"[resume] checkpoint={args.resume_from} "
            f"checkpoint_epoch={checkpoint.get('epoch', 0)} "
            f"epoch_complete={completed} next_epoch={start_epoch} "
            f"best_cc={best_cc:.4f}"
        )

    if start_epoch > cfg["epochs"]:
        print(
            f"[done] checkpoint already completed epoch {start_epoch - 1}; "
            f"requested epochs={cfg['epochs']}"
        )
        return

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        validation_only_resume = (
            epoch == start_epoch
            and not resume_epoch_complete
            and resume_batch is not None
            and int(resume_batch) >= len(train_loader)
        )
        if validation_only_resume:
            train_loss = float("nan")
            train_detail = {}
            skipped = 0
            print(
                f"[resume] epoch {epoch} training already completed at "
                f"batch {resume_batch}/{len(train_loader)}; running "
                "validation only.",
                flush=True,
            )
        else:
            train_loss, train_detail, skipped = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                device,
                cfg,
                scaler,
                epoch,
                save_dir / "recovery_model_v7.pth",
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print(
            f"[validate] epoch={epoch} batches={len(val_loader)} "
            f"batch_size={val_loader.batch_size}",
            flush=True,
        )
        metrics = validate(
            model, val_loader, criterion, device, cfg["st_window"]
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "skipped": skipped,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_detail.items()},
            **metrics,
        }
        history.append(row)
        history_path.write_text(
            json.dumps(history, indent=2, allow_nan=True), encoding="utf-8"
        )
        print(
            f"[{epoch:02d}/{cfg['epochs']}] loss={train_loss:.4f} "
            f"gain={metrics['val_gain']:+.2f} CC={metrics['val_cc']:.4f} "
            f"RMSE={metrics['val_rmse']:.5f} "
            f"PRD={metrics['val_prd']:.2f}% "
            f"ST={metrics['val_st_mae']:.5f} "
            f"A-CC={metrics['adaptive_cc']:.4f} "
            f"Qcorr={metrics['quality_corr']:.3f} "
            f"mass={metrics['selected_mass']:.3f} "
            f"steps={metrics['effective_steps']:.2f} "
            f"stop={metrics['early_stop_rate']:.1%}"
        )

        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "epoch_complete": True,
            "metrics": metrics,
            "config": cfg,
        }
        torch.save(state, save_dir / "last_model_v7.pth")
        if metrics["val_cc"] > best_cc:
            best_cc = metrics["val_cc"]
            torch.save(state, save_dir / "best_model_v7.pth")


if __name__ == "__main__":
    main()
