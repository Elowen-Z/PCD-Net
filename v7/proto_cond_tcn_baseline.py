"""Prototype-conditioned temporal denoising baseline for STEAD.

This baseline is designed as a mechanism-matched comparator for PCD-Net:
it uses (1) a background-noise condition, (2) learnable prototype memory, and
(3) a temporal convolutional denoiser. It deliberately does not use PCD-Net's
prototype-conditioned cross-attention, sparse multi-token retrieval feedback,
quality head, or adaptive stopping.

In the paper, describe it as a prototype-conditioned TCN baseline rather than
as an exact reproduction of a named external method.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from v3.dataset_v3 import STEADDatasetV3
from v5.train_v6 import cc_batch, compute_snr_batch, set_seed, split_dataset
from v7.gdiffuse_style_baseline import (
    DEFAULT_EVENT_CSV,
    DEFAULT_EVENT_H5,
    DEFAULT_NOISE_CSV,
    DEFAULT_NOISE_H5,
    st_mae,
)


class NoiseEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 9, padding=4),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, dim, 5, stride=2, padding=2),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(condition))


class FiLMTCNBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dilation: int):
        super().__init__()
        padding = dilation * 3
        self.conv1 = nn.Conv1d(channels, channels, 7, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 7, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.film = nn.Linear(cond_dim, channels * 4)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g1, b1, g2, b2 = self.film(cond).chunk(4, dim=-1)
        h = self.norm1(self.conv1(x))
        h = h * (1.0 + g1.unsqueeze(-1)) + b1.unsqueeze(-1)
        h = F.silu(h)
        h = self.norm2(self.conv2(h))
        h = h * (1.0 + g2.unsqueeze(-1)) + b2.unsqueeze(-1)
        h = F.silu(h)
        return x + h


class PrototypeConditioner(nn.Module):
    def __init__(self, dim: int = 128, num_prototypes: int = 16, top_m: int = 4):
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.top_m = int(top_m)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim) * 0.02)
        self.query = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, noise_state: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query = F.normalize(self.query(noise_state), dim=-1)
        codebook = F.normalize(self.prototypes, dim=-1)
        logits = query @ codebook.t()
        weights = F.softmax(logits, dim=-1)
        m = min(max(self.top_m, 1), self.num_prototypes)
        top_values, top_indices = torch.topk(weights, k=m, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        selected = self.prototypes[top_indices]
        proto_context = (selected * top_values.unsqueeze(-1)).sum(dim=1)
        cond = self.fuse(torch.cat([noise_state, proto_context], dim=-1))
        return cond, {
            "prototype_weights": weights,
            "top_indices": top_indices,
            "top_weights": top_values,
        }


class ProtoCondTCNDenoiser(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        hidden: int = 64,
        cond_dim: int = 128,
        num_prototypes: int = 16,
        top_m: int = 4,
        layers: int = 8,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.noise_encoder = NoiseEncoder(in_ch=in_ch, dim=cond_dim)
        self.conditioner = PrototypeConditioner(
            dim=cond_dim,
            num_prototypes=num_prototypes,
            top_m=top_m,
        )
        self.input = nn.Conv1d(in_ch, hidden, 9, padding=4)
        dilations = [2 ** (i % 6) for i in range(layers)]
        self.blocks = nn.ModuleList(
            [FiLMTCNBlock(hidden, cond_dim, dilation=d) for d in dilations]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden, in_ch, 9, padding=4),
        )

    def forward(self, noisy: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        noise_state = self.noise_encoder(condition)
        proto_cond, aux = self.conditioner(noise_state)
        h = self.input(noisy)
        for block in self.blocks:
            h = block(h, proto_cond)
        removed_noise = self.output(h)
        clean = noisy - self.residual_scale * removed_noise
        aux["noise_state"] = noise_state
        return clean, aux


def make_dataset(csv_path: str, args: argparse.Namespace, seed: int):
    """Build the shared V3 dataset for the prototype-conditioned TCN.

    ``--snr_sampling db`` samples the input SNR uniformly in decibels, which is
    the distribution PCD-Net V9 is trained on.  The legacy default keeps the
    historical behaviour: ``--train_snr_is_db`` / ``--eval_snr_is_db`` convert
    the bounds to linear power ratios and the sampler stays uniform in linear
    power.
    """
    if bool(args.eval_mode):
        low = float(args.eval_snr_min)
        high = float(args.eval_snr_max)
        is_db = bool(args.eval_snr_is_db)
    else:
        low = float(args.train_snr_min)
        high = float(args.train_snr_max)
        is_db = bool(args.train_snr_is_db)

    sampling = getattr(args, "snr_sampling", None) or (
        "db" if is_db else "linear"
    )
    if sampling == "linear":
        low, high = 10.0 ** (low / 10.0), 10.0 ** (high / 10.0)
    return STEADDatasetV3(
        event_h5_path=args.event_h5,
        event_csv_path=csv_path,
        noise_h5_path=args.noise_h5,
        noise_csv_path=args.noise_csv,
        signal_len=args.signal_len,
        cond_len=args.cond_len,
        snr_range=(low, high),
        snr_sampling=sampling,
        clean_prob=0.0,
        part_b_ratio=0.0,
        seed=seed,
    )


def subset_dataset(dataset, max_samples: int | None, seed: int):
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    indices = np.random.default_rng(seed).choice(
        len(dataset), size=max_samples, replace=False
    )
    return Subset(dataset, indices.tolist())


def make_loaders(args: argparse.Namespace):
    train_csv, val_csv = split_dataset(
        args.event_csv,
        0.1,
        args.seed,
        prefix=f"proto_cond_tcn_seed{args.seed}",
    )
    args.eval_mode = False
    train_ds = subset_dataset(
        make_dataset(train_csv, args, args.seed),
        args.train_samples,
        args.seed,
    )
    val_ds = subset_dataset(
        make_dataset(val_csv, args, args.seed + 1000),
        args.val_samples,
        args.seed + 1000,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader, val_csv


def reconstruction_loss(pred: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.unsqueeze(1)
    denom = valid.sum().clamp_min(1.0) * pred.size(1)
    l1 = ((pred - clean).abs() * valid).sum() / denom
    mse = (((pred - clean) ** 2) * valid).sum() / denom
    grad_pred = pred[..., 1:] - pred[..., :-1]
    grad_clean = clean[..., 1:] - clean[..., :-1]
    grad_mask = valid[..., 1:]
    grad = ((grad_pred - grad_clean).abs() * grad_mask).sum() / grad_mask.sum().clamp_min(1.0) / pred.size(1)
    return l1 + 0.25 * mse + 0.10 * grad


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    losses, gains, correlations = [], [], []
    for batch in loader:
        noisy = batch["x"].to(device)
        clean = batch["y_clean"].to(device)
        condition = batch["z_cond"].to(device)
        mask = batch["valid_mask"].to(device)
        pred, _ = model(noisy, condition)
        losses.append(float(reconstruction_loss(pred, clean, mask).detach().cpu()))
        snr_in = compute_snr_batch(clean, noisy - clean, mask)
        snr_out = compute_snr_batch(clean, pred - clean, mask)
        gains.extend((snr_out - snr_in).cpu().tolist())
        correlations.extend(cc_batch(pred, clean).cpu().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "gain": float(np.mean(gains)) if gains else float("nan"),
        "cc": float(np.mean(correlations)) if correlations else float("nan"),
    }


@torch.no_grad()
def evaluate(model: nn.Module, dataset, args: argparse.Namespace, device: torch.device):
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    gains, snr_inputs, snr_outputs = [], [], []
    correlations, rmses, prds, st_noisy_values, st_values, event_st_values = [], [], [], [], [], []
    started = time.perf_counter()
    processed = 0
    model.eval()
    for batch_index, batch in enumerate(loader, start=1):
        noisy = batch["x"].to(device)
        clean = batch["y_clean"].to(device)
        condition = batch["z_cond"].to(device)
        mask = batch["valid_mask"].to(device)
        supervised = batch["has_target"].bool().to(device)
        if not supervised.any():
            continue
        pred, _ = model(noisy, condition)
        pred = pred[supervised]
        target = clean[supervised]
        noisy_sup = noisy[supervised]
        mask_sup = mask[supervised]

        snr_in = compute_snr_batch(target, noisy_sup - target, mask_sup)
        snr_out = compute_snr_batch(target, pred - target, mask_sup)
        snr_inputs.extend(snr_in.cpu().tolist())
        snr_outputs.extend(snr_out.cpu().tolist())
        gains.extend((snr_out - snr_in).cpu().tolist())
        correlations.extend(cc_batch(pred, target).cpu().tolist())

        error = pred - target
        rmse = torch.sqrt(torch.mean(error ** 2, dim=(1, 2)))
        reference = torch.sqrt(torch.mean(target ** 2, dim=(1, 2))).clamp_min(1e-8)
        rmses.extend(rmse.cpu().tolist())
        prds.extend((100.0 * rmse / reference).cpu().tolist())
        st_noisy, _ = st_mae(noisy_sup, target, mask_sup, args.st_window)
        st_all, st_event = st_mae(pred, target, mask_sup, args.st_window)
        st_noisy_values.extend(st_noisy.cpu().tolist())
        st_values.extend(st_all.cpu().tolist())
        event_st_values.extend(st_event.cpu().tolist())
        processed += int(supervised.sum())
        if batch_index % 25 == 0:
            print(
                f"[eval {processed}/{len(dataset)}] "
                f"gain={np.mean(gains):+.2f} CC={np.mean(correlations):.4f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "method": "ProtoCondTCN",
        "samples": processed,
        "delta_snr_db": float(np.mean(gains)),
        "snr_in_db": float(np.mean(snr_inputs)),
        "snr_out_db": float(np.mean(snr_outputs)),
        "cc": float(np.mean(correlations)),
        "rmse": float(np.mean(rmses)),
        "prd_percent": float(np.mean(prds)),
        "st_mae_noisy": float(np.mean(st_noisy_values)),
        "st_mae": float(np.mean(st_values)),
        "event_st_mae": float(np.mean(event_st_values)),
        "elapsed_seconds": elapsed,
        "samples_per_second": processed / max(elapsed, 1e-8),
        "num_prototypes": args.num_prototypes,
        "top_m": args.top_m,
    }


def train(args: argparse.Namespace, device: torch.device, save_dir: Path):
    train_loader, val_loader, val_csv = make_loaders(args)
    model = ProtoCondTCNDenoiser(
        hidden=args.hidden,
        cond_dim=args.cond_dim,
        num_prototypes=args.num_prototypes,
        top_m=args.top_m,
        layers=args.layers,
        residual_scale=args.residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch_index, batch in enumerate(train_loader, start=1):
            noisy = batch["x"].to(device)
            clean = batch["y_clean"].to(device)
            condition = batch["z_cond"].to(device)
            mask = batch["valid_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(noisy, condition)
            loss = reconstruction_loss(pred, clean, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.detach().cpu()) * noisy.size(0)
            count += noisy.size(0)
            if args.print_every > 0 and batch_index % args.print_every == 0:
                print(
                    f"[epoch {epoch:02d} batch {batch_index:05d}/"
                    f"{len(train_loader):05d}] loss={total / max(count, 1):.5f}",
                    flush=True,
                )
        train_loss = total / max(count, 1)
        val_metrics = validate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_gain": val_metrics["gain"],
            "val_cc": val_metrics["cc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"[{epoch:02d}/{args.epochs}] loss={train_loss:.5f} "
            f"val_loss={row['val_loss']:.5f} gain={row['val_gain']:+.2f} "
            f"CC={row['val_cc']:.4f}",
            flush=True,
        )
        state = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": row,
            "config": vars(args),
        }
        torch.save(state, save_dir / "last_proto_cond_tcn.pth")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(state, save_dir / "best_proto_cond_tcn.pth")
    return model, val_csv


def load_model(checkpoint_path: Path, args: argparse.Namespace, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = ProtoCondTCNDenoiser(
        hidden=int(config.get("hidden", args.hidden)),
        cond_dim=int(config.get("cond_dim", args.cond_dim)),
        num_prototypes=int(config.get("num_prototypes", args.num_prototypes)),
        top_m=int(config.get("top_m", args.top_m)),
        layers=int(config.get("layers", args.layers)),
        residual_scale=float(config.get("residual_scale", args.residual_scale)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def write_compare_csv(save_dir: Path, metrics: dict):
    path = save_dir / "global_compare.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Model",
                "N",
                "DeltaSNR(dB)",
                "SNR_in(dB)",
                "SNR_out(dB)",
                "CC",
                "RMSE",
                "PRD",
                "ST-MAE(noisy)",
                "ST-MAE(denoised)",
                "time_sec",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Model": "ProtoCondTCN",
                "N": metrics["samples"],
                "DeltaSNR(dB)": metrics["delta_snr_db"],
                "SNR_in(dB)": metrics["snr_in_db"],
                "SNR_out(dB)": metrics["snr_out_db"],
                "CC": metrics["cc"],
                "RMSE": metrics["rmse"],
                "PRD": metrics["prd_percent"],
                "ST-MAE(noisy)": metrics["st_mae_noisy"],
                "ST-MAE(denoised)": metrics["st_mae"],
                "time_sec": metrics["elapsed_seconds"],
            }
        )
    return path


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_h5", default=DEFAULT_EVENT_H5)
    parser.add_argument("--event_csv", default=DEFAULT_EVENT_CSV)
    parser.add_argument("--noise_h5", default=DEFAULT_NOISE_H5)
    parser.add_argument("--noise_csv", default=DEFAULT_NOISE_CSV)
    parser.add_argument("--save_dir", default="v7/proto_cond_tcn_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--signal_len", type=int, default=6000)
    parser.add_argument("--cond_len", type=int, default=400)
    parser.add_argument("--train_snr_min", type=float, default=0.1)
    parser.add_argument("--train_snr_max", type=float, default=20.0)
    parser.add_argument("--eval_snr_min", type=float, default=0.1)
    parser.add_argument("--eval_snr_max", type=float, default=20.0)
    parser.add_argument("--train_snr_is_db", action="store_true")
    parser.add_argument("--eval_snr_is_db", action="store_true")
    parser.add_argument(
        "--snr_sampling",
        choices=["linear", "db"],
        default=None,
        help=(
            "db = 在 dB 上均匀采样输入 SNR (使用 --*_snr_min/max 的 dB 边界, "
            "跳过功率比换算), 与 PCD-Net V9 训练口径一致; "
            "linear/未给出 = 旧口径 (功率比均匀)"
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--val_samples", type=int, default=500)
    parser.add_argument("--eval_samples", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--cond_dim", type=int, default=128)
    parser.add_argument("--num_prototypes", type=int, default=16)
    parser.add_argument("--top_m", type=int, default=4)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--st_window", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    args.eval_mode = False
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    if args.eval_only:
        if not args.checkpoint:
            args.checkpoint = str(save_dir / "best_proto_cond_tcn.pth")
        model = load_model(Path(args.checkpoint), args, device)
        _, val_csv = split_dataset(
            args.event_csv,
            0.1,
            args.seed,
            prefix=f"proto_cond_tcn_seed{args.seed}",
        )
    else:
        model, val_csv = train(args, device, save_dir)
        best_path = save_dir / "best_proto_cond_tcn.pth"
        if best_path.exists():
            model = load_model(best_path, args, device)
    args.eval_mode = True
    eval_ds = subset_dataset(
        make_dataset(val_csv, args, args.seed + 2000),
        args.eval_samples,
        args.seed + 2000,
    )
    metrics = evaluate(model, eval_ds, args, device)
    print(json.dumps(metrics, indent=2), flush=True)
    (save_dir / "evaluation_proto_cond_tcn_2000.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    csv_path = write_compare_csv(save_dir, metrics)
    print(f"[saved] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
