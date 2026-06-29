"""Multi-seed test-subset resampling stability experiment.

This appendix experiment does not retrain models. It uses the saved per-sample
prediction records from the main STEAD model-comparison experiment and
evaluates whether the reported ranking is stable under different random test
subsets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "PCD-Net",
    "DPRNN",
    "Restormer1D",
    "Wavelet",
    "DeepDenoiser",
    "Bandpass",
]

FILE_STEMS = {
    "PCD-Net": "V7",
    "DPRNN": "DPRNN",
    "Restormer1D": "Restormer1D",
    "Wavelet": "Wavelet",
    "DeepDenoiser": "DeepDenoiser",
    "Bandpass": "Bandpass",
}

METRICS = [
    ("delta_snr", "Delta SNR (dB)", True),
    ("cc", "CC", True),
    ("rmse", "RMSE", False),
    ("st_mae_denoised", "ST-MAE", False),
]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_model_records(input_dir: Path, method: str) -> pd.DataFrame:
    stem = FILE_STEMS[method]
    path = input_dir / f"per_model_{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = [metric for metric, _, _ in METRICS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")
    return frame


def evaluate_resampled(
    records_by_method: dict[str, pd.DataFrame],
    seeds: list[int],
    sample_size: int,
) -> tuple[list[dict], list[dict]]:
    seed_rows = []
    summary_rows = []
    for method in METHOD_ORDER:
        frame = records_by_method[method]
        n_available = len(frame)
        n_sample = n_available if sample_size <= 0 else min(sample_size, n_available)
        for seed in seeds:
            sampled = frame.sample(
                n=n_sample,
                replace=False,
                random_state=seed,
            )
            row = {
                "Method": method,
                "Seed": seed,
                "N": n_sample,
            }
            for metric, _, _ in METRICS:
                row[metric] = float(sampled[metric].mean())
            seed_rows.append(row)

        method_seed_rows = [row for row in seed_rows if row["Method"] == method]
        summary = {"Method": method, "Seeds": len(seeds), "N_per_seed": n_sample}
        for metric, _, _ in METRICS:
            values = np.array([row[metric] for row in method_seed_rows], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary[f"{metric}_mean_std"] = (
                f"{summary[f'{metric}_mean']:.4f} ± {summary[f'{metric}_std']:.4f}"
            )
        summary_rows.append(summary)
    return seed_rows, summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict]) -> None:
    lines = [
        "| Method | Delta SNR (dB) | CC | RMSE | ST-MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {Method} | {delta_snr_mean_std} | {cc_mean_std} | "
            "{rmse_mean_std} | {st_mae_denoised_mean_std} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_errorbars(summary_rows: list[dict], out_png: Path, out_svg: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.7), dpi=240)
    x = np.arange(len(summary_rows))
    colors = ["#D94B5F" if row["Method"] == "PCD-Net" else "#6B7280" for row in summary_rows]
    for ax, (metric, title, higher_better) in zip(axes, METRICS):
        means = [row[f"{metric}_mean"] for row in summary_rows]
        stds = [row[f"{metric}_std"] for row in summary_rows]
        ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.9, edgecolor="black", linewidth=0.4)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([row["Method"] for row in summary_rows], rotation=35, ha="right", fontsize=7.5)
        ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.35)
        if not higher_better:
            ax.text(0.02, 0.96, "lower is better", transform=ax.transAxes, va="top", fontsize=7, color="#555555")
    fig.suptitle("Stability under Random Test-Subset Resampling", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="v7/paper_experiments/model_comparison_stead",
    )
    parser.add_argument(
        "--output_dir",
        default="v7/paper_experiments/appendix/seed_stability_resampling",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="samples per seed; use 0 or negative to use all saved samples",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.seeds)
    records_by_method = {
        method: load_model_records(input_dir, method) for method in METHOD_ORDER
    }
    seed_rows, summary_rows = evaluate_resampled(
        records_by_method,
        seeds=seeds,
        sample_size=args.sample_size,
    )
    write_csv(out_dir / "seed_stability_per_seed.csv", seed_rows)
    write_csv(out_dir / "seed_stability_summary.csv", summary_rows)
    write_markdown(out_dir / "seed_stability_summary.md", summary_rows)
    plot_errorbars(
        summary_rows,
        out_dir / "seed_stability_errorbar.png",
        out_dir / "seed_stability_errorbar.svg",
    )
    print(f"[DONE] {out_dir}", flush=True)
    print((out_dir / "seed_stability_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
