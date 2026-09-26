"""Evaluate V9 with correct, shuffled, random, and absent backgrounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from v5.evaluate_transfer_v5 import EvalDataset, cc_fn, rmse_fn, snr_db, st_mae_mean
from v9.compare_v9 import build_v9_denoiser
from v7.train_v7 import configure_cuda_backend


MODES = (
    ("correct", "Correct background"),
    ("shuffled", "Shuffled background"),
    ("random", "Random background"),
    ("none", "No background"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--event_h5", default="E:/p_wave/data/chunk2.hdf5")
    parser.add_argument("--event_csv", default="v5/v7_seed0_val.csv")
    parser.add_argument("--noise_h5", default="E:/p_wave/data/chunk1.hdf5")
    parser.add_argument("--noise_csv", default="E:/p_wave/data/chunk1.csv")
    parser.add_argument("--snr_min_db", type=float, default=-15.0)
    parser.add_argument("--snr_max_db", type=float, default=10.0)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--signal_len", type=int, default=6000)
    parser.add_argument("--cond_len", type=int, default=400)
    parser.add_argument("--cuda_safe", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output_dir", default="v9/paper_experiments/ablation_v9_seed0/context")
    return parser


def sha1_16(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def random_context(condition: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    mean = condition.mean(dim=-1, keepdim=True)
    std = condition.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return mean + std * torch.randn(
        condition.shape,
        dtype=condition.dtype,
        device=condition.device,
        generator=generator,
    )


def collect_batch(
    prediction: torch.Tensor,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    nominal_snr: torch.Tensor,
) -> list[dict]:
    rows = []
    for index in range(noisy.size(0)):
        pred = prediction[index].detach().float().cpu().numpy().astype(np.float64)
        source = noisy[index].detach().float().cpu().numpy().astype(np.float64)
        target = clean[index].detach().float().cpu().numpy().astype(np.float64)
        snr_out = snr_db(target.reshape(-1), pred.reshape(-1))
        rows.append({
            "DeltaSNR_dB": snr_out - float(nominal_snr[index]),
            "CC": cc_fn(target.reshape(-1), pred.reshape(-1)),
            "RMSE": rmse_fn(target.reshape(-1), pred.reshape(-1)),
            "ST_MAE": st_mae_mean(target, pred, 100.0),
            "SNR_in_dB": float(nominal_snr[index]),
            "ST_MAE_noisy": st_mae_mean(target, source, 100.0),
        })
    return rows


def plot_table(table: pd.DataFrame, output_dir: Path) -> None:
    metrics = (
        ("DeltaSNR_dB", r"$\Delta$SNR (dB)"),
        ("CC", "CC"),
        ("RMSE", "RMSE"),
        ("ST_MAE", "ST-MAE"),
    )
    colors = ("#2563eb", "#d97706", "#7c3aed", "#6b7280")
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8), dpi=240)
    x = np.arange(len(table))
    for axis, (metric, ylabel) in zip(axes, metrics):
        axis.bar(x, table[metric], color=colors, width=0.72)
        axis.set_xticks(x, ("Correct", "Shuffled", "Random", "None"), rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    fig.suptitle("Background-context ablation", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "context_ablation.png", bbox_inches="tight")
    fig.savefig(output_dir / "context_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 2:
        raise ValueError("batch_size must be at least 2 for shuffled context")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    configure_cuda_backend(args.cuda_safe and device.type == "cuda")
    print(f"[device] {device} cuda_safe={args.cuda_safe}", flush=True)
    denoiser = build_v9_denoiser(args.checkpoint, device)
    model = denoiser.model
    dataset = EvalDataset(
        event_h5_path=args.event_h5,
        event_csv_path=args.event_csv,
        noise_h5_path=args.noise_h5,
        noise_csv_path=args.noise_csv,
        signal_len=args.signal_len,
        cond_len=args.cond_len,
        snr_db_range=(args.snr_min_db, args.snr_max_db),
        noise_boost=1.0,
        max_samples=args.max_samples,
        seed=args.seed,
        context_source="pre_event",
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    generator = torch.Generator(device=device.type)
    generator.manual_seed(args.seed + 100003)
    records = {mode: [] for mode, _ in MODES}

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            condition = batch["z_cond"].to(device)
            nominal_snr = batch["snr_db"]
            contexts = {
                "correct": condition,
                "shuffled": torch.roll(condition, shifts=1, dims=0),
                "random": random_context(condition, generator),
                "none": torch.zeros_like(condition),
            }
            for mode, _ in MODES:
                prediction, _, _, _ = model(
                    noisy, contexts[mode], adaptive_stop=False
                )
                records[mode].extend(
                    collect_batch(prediction, noisy, clean, nominal_snr)
                )
            if batch_index % 100 == 0:
                print(f"[context] samples={len(records['correct'])}", flush=True)

    summaries = []
    raw_frames = []
    for mode, label in MODES:
        frame = pd.DataFrame(records[mode])
        frame.insert(0, "Context", mode)
        raw_frames.append(frame)
        summaries.append({
            "Context": label,
            "N": len(frame),
            "DeltaSNR_dB": float(frame["DeltaSNR_dB"].mean()),
            "CC": float(frame["CC"].mean()),
            "RMSE": float(frame["RMSE"].mean()),
            "ST_MAE": float(frame["ST_MAE"].mean()),
        })
    raw = pd.concat(raw_frames, ignore_index=True)
    table = pd.DataFrame(summaries)
    raw.to_csv(output_dir / "context_ablation_per_sample.csv", index=False, encoding="utf-8-sig")
    table.to_csv(output_dir / "context_ablation_table.csv", index=False, encoding="utf-8-sig")
    latex = table.rename(columns={"DeltaSNR_dB": r"$\Delta$SNR (dB)", "ST_MAE": "ST-MAE"})
    (output_dir / "context_ablation_table.tex").write_text(
        latex.to_latex(index=False, escape=False, float_format="%.4f"), encoding="utf-8"
    )
    audit = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha1_16": sha1_16(args.checkpoint),
        "event_csv": args.event_csv,
        "context_source": "pre_event",
        "input_snr_db": [args.snr_min_db, args.snr_max_db],
        "seed": args.seed,
        "samples": len(dataset),
        "definitions": {
            "correct": "the pre-event segment from the same waveform",
            "shuffled": "a one-position cyclic permutation within each batch",
            "random": "Gaussian context matched per sample and channel in mean and standard deviation",
            "none": "an all-zero context",
        },
    }
    (output_dir / "context_ablation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    plot_table(table, output_dir)
    print(table.to_string(index=False), flush=True)
    print(f"[saved] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
