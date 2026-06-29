"""Sensitivity experiments for PCD-Net appendix.

The script evaluates:
1) refinement-pass sensitivity;
2) adaptive stopping-threshold sensitivity;
3) Top-M sparse prototype sensitivity;
4) effective prototype-count sensitivity.

It does not retrain models. Effective prototype count is implemented by
masking the prototype prior logits during inference while keeping the trained
16-entry codebook unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from v7.evaluate_v7 import build_model, evaluate, make_dataset


def configure_cuda_safe() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def parse_numbers(text: str, cast=float):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(experiment: str, setting: str, metrics: dict) -> dict:
    row = {
        "experiment": experiment,
        "setting": setting,
        "samples": metrics.get("samples"),
        "delta_snr_db": metrics.get("delta_snr_db"),
        "cc": metrics.get("cc"),
        "rmse": metrics.get("rmse"),
        "prd_percent": metrics.get("prd_percent"),
        "st_mae": metrics.get("st_mae"),
        "event_st_mae": metrics.get("event_st_mae"),
        "quality_corr": metrics.get("quality_corr"),
        "selected_mass": metrics.get("selected_mass"),
        "effective_steps": metrics.get("effective_steps"),
        "early_stop_rate": metrics.get("early_stop_rate"),
        "samples_per_second": metrics.get("samples_per_second"),
    }
    return row


def set_effective_k(model, effective_k: int | None) -> None:
    with torch.no_grad():
        model.tta_logits.zero_()
        if effective_k is not None:
            k = int(effective_k)
            if k < 1 or k > model.tta_logits.numel():
                raise ValueError(f"effective_k out of range: {effective_k}")
            model.tta_logits[k:] = -40.0


def plot_metric(rows: list[dict], experiment: str, out_path: Path) -> None:
    selected = [row for row in rows if row["experiment"] == experiment]
    if not selected:
        return
    labels = [str(row["setting"]) for row in selected]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), dpi=220)
    specs = [
        ("delta_snr_db", "Delta SNR (dB)", "#D94B5F"),
        ("cc", "CC", "#4C78A8"),
        ("st_mae", "ST-MAE", "#59A14F"),
    ]
    for ax, (key, title, color) in zip(axes, specs):
        values = [float(row[key]) for row in selected]
        ax.plot(x, values, marker="o", linewidth=1.8, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.28, linestyle="--", linewidth=0.5)
        for xi, value in zip(x, values):
            text = f"{value:.2f}" if key == "delta_snr_db" else f"{value:.3f}"
            ax.text(xi, value, text, ha="center", va="bottom", fontsize=7)
    fig.suptitle(experiment.replace("_", " ").title(), fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="v7/checkpoints_feedback_stead_seed0/best_model_v7.pth",
    )
    parser.add_argument("--dataset_type", choices=["stead", "mining"], default="stead")
    parser.add_argument("--event_h5", default="D:/X/p_wave/data/chunk2.hdf5")
    parser.add_argument("--event_csv", default="D:/X/p_wave/data/chunk2.csv")
    parser.add_argument("--noise_h5", default="D:/X/p_wave/data/chunk1.hdf5")
    parser.add_argument("--noise_csv", default="D:/X/p_wave/data/chunk1.csv")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--st_window", type=int, default=100)
    parser.add_argument("--refine_passes", default="1,2,3,4")
    parser.add_argument("--stop_thresholds", default="0.85,0.90,0.95,0.98")
    parser.add_argument("--top_m_values", default="1,2,4,8")
    parser.add_argument("--effective_k_values", default="4,8,12,16")
    parser.add_argument(
        "--experiments",
        default="refine,threshold,topm,keff",
        help="comma-separated subset: refine,threshold,topm,keff",
    )
    parser.add_argument(
        "--output_dir",
        default="v7/paper_experiments/appendix/sensitivity",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fast_cuda", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.fast_cuda:
        configure_cuda_safe()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, config = build_model(checkpoint, device)
    model.eval()

    dataset = make_dataset(args, config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    selected_experiments = set(item.strip() for item in args.experiments.split(","))

    original_n_refine = model.n_refine
    original_threshold = model.stop_threshold
    original_top_m = model.selector.top_m
    original_logits = model.tta_logits.detach().clone()

    rows = []
    try:
        if "refine" in selected_experiments:
            for passes in parse_numbers(args.refine_passes, int):
                model.n_refine = max(0, int(passes) - 1)
                model.stop_threshold = original_threshold
                model.selector.top_m = original_top_m
                set_effective_k(model, None)
                metrics = evaluate(
                    model,
                    loader,
                    device,
                    adaptive_stop=False,
                    st_window=args.st_window,
                )
                row = metric_row("refinement_passes", str(passes), metrics)
                rows.append(row)
                print(
                    f"[refine={passes}] gain={row['delta_snr_db']:+.3f} "
                    f"CC={row['cc']:.4f} ST={row['st_mae']:.5f}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if "threshold" in selected_experiments:
            model.n_refine = original_n_refine
            model.selector.top_m = original_top_m
            set_effective_k(model, None)
            for threshold in parse_numbers(args.stop_thresholds, float):
                model.stop_threshold = float(threshold)
                metrics = evaluate(
                    model,
                    loader,
                    device,
                    adaptive_stop=True,
                    st_window=args.st_window,
                )
                row = metric_row("stop_threshold", f"{threshold:.2f}", metrics)
                rows.append(row)
                print(
                    f"[threshold={threshold:.2f}] gain={row['delta_snr_db']:+.3f} "
                    f"CC={row['cc']:.4f} ST={row['st_mae']:.5f} "
                    f"steps={row['effective_steps']:.2f}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if "topm" in selected_experiments:
            model.n_refine = original_n_refine
            model.stop_threshold = original_threshold
            set_effective_k(model, None)
            for top_m in parse_numbers(args.top_m_values, int):
                model.selector.top_m = int(top_m)
                metrics = evaluate(
                    model,
                    loader,
                    device,
                    adaptive_stop=True,
                    st_window=args.st_window,
                )
                row = metric_row("top_m", str(top_m), metrics)
                rows.append(row)
                print(
                    f"[top_m={top_m}] gain={row['delta_snr_db']:+.3f} "
                    f"CC={row['cc']:.4f} ST={row['st_mae']:.5f}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if "keff" in selected_experiments:
            model.n_refine = original_n_refine
            model.stop_threshold = original_threshold
            for effective_k in parse_numbers(args.effective_k_values, int):
                model.selector.top_m = min(original_top_m, int(effective_k))
                set_effective_k(model, int(effective_k))
                metrics = evaluate(
                    model,
                    loader,
                    device,
                    adaptive_stop=True,
                    st_window=args.st_window,
                )
                row = metric_row("effective_prototypes", str(effective_k), metrics)
                rows.append(row)
                print(
                    f"[K_eff={effective_k}] gain={row['delta_snr_db']:+.3f} "
                    f"CC={row['cc']:.4f} ST={row['st_mae']:.5f}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        model.n_refine = original_n_refine
        model.stop_threshold = original_threshold
        model.selector.top_m = original_top_m
        with torch.no_grad():
            model.tta_logits.copy_(original_logits)

    write_csv(out_dir / "sensitivity_summary.csv", rows)
    (out_dir / "sensitivity_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    for experiment in sorted({row["experiment"] for row in rows}):
        plot_metric(rows, experiment, out_dir / f"{experiment}.png")
        plot_metric(rows, experiment, out_dir / f"{experiment}.svg")
    print(f"\n[DONE] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
