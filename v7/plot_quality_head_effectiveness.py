"""Plot quality-head effectiveness for PCD-Net.

The figure shows whether the predicted denoising quality score is correlated
with actual reconstruction fidelity on held-out samples.
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


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return pearson(xr, yr)


def build_bins(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    work = frame.copy()
    work["quality_bin"] = pd.qcut(
        work["model_quality"],
        q=bins,
        labels=False,
        duplicates="drop",
    )
    grouped = (
        work.groupby("quality_bin", as_index=False)
        .agg(
            quality_mean=("model_quality", "mean"),
            quality_std=("model_quality", "std"),
            fidelity_mean=("fidelity_score", "mean"),
            fidelity_std=("fidelity_score", "std"),
            cc_mean=("cc", "mean"),
            cc_std=("cc", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            st_mae_mean=("st_mae_denoised", "mean"),
            st_mae_std=("st_mae_denoised", "std"),
            n=("model_quality", "size"),
        )
    )
    return grouped


def write_correlation_table(frame: pd.DataFrame, out_path: Path) -> None:
    quality = frame["model_quality"].to_numpy(dtype=float)
    rows = []
    for metric, label, expected in [
        ("fidelity_score", "Fidelity score", "positive"),
        ("cc", "CC", "positive"),
        ("rmse", "RMSE", "negative"),
        ("st_mae_denoised", "ST-MAE", "negative"),
    ]:
        values = frame[metric].to_numpy(dtype=float)
        rows.append(
            {
                "Relationship": f"Quality vs {label}",
                "Pearson_r": pearson(quality, values),
                "Spearman_r": spearman(quality, values),
                "Expected": expected,
            }
        )
    with out_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(frame: pd.DataFrame, bins: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    q = frame["model_quality"].to_numpy(dtype=float)
    fidelity = frame["fidelity_score"].to_numpy(dtype=float)
    cc = frame["cc"].to_numpy(dtype=float)
    rmse = frame["rmse"].to_numpy(dtype=float)
    st = frame["st_mae_denoised"].to_numpy(dtype=float)

    r_fidelity = pearson(q, fidelity)
    r_cc = pearson(q, cc)
    r_rmse = pearson(q, rmse)
    r_st = pearson(q, st)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0), dpi=240)
    ax = axes[0, 0]
    rng = np.random.default_rng(0)
    if len(frame) > 1200:
        idx = rng.choice(len(frame), size=1200, replace=False)
    else:
        idx = np.arange(len(frame))
    ax.scatter(
        q[idx],
        fidelity[idx],
        s=10,
        alpha=0.25,
        color="#4C78A8",
        edgecolors="none",
        label="Samples",
    )
    ax.plot(
        bins["quality_mean"],
        bins["fidelity_mean"],
        color="#D94B5F",
        marker="o",
        linewidth=2.4,
        label="Quantile-bin mean",
    )
    ax.set_xlabel("Predicted quality score")
    ax.set_ylabel("Actual fidelity score")
    ax.set_title(f"Quality vs Fidelity (r = {r_fidelity:.3f})", fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(bins["quality_mean"], bins["cc_mean"], marker="o", color="#059669", linewidth=2.2)
    ax.fill_between(
        bins["quality_mean"],
        bins["cc_mean"] - bins["cc_std"].fillna(0),
        bins["cc_mean"] + bins["cc_std"].fillna(0),
        color="#059669",
        alpha=0.12,
        linewidth=0,
    )
    ax.set_xlabel("Predicted quality score")
    ax.set_ylabel("Mean CC")
    ax.set_title(f"Higher quality -> higher CC (r = {r_cc:.3f})", fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    ax = axes[1, 0]
    ax.plot(bins["quality_mean"], bins["rmse_mean"], marker="o", color="#EA580C", linewidth=2.2)
    ax.set_xlabel("Predicted quality score")
    ax.set_ylabel("Mean RMSE")
    ax.set_title(f"Higher quality -> lower RMSE (r = {r_rmse:.3f})", fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    ax = axes[1, 1]
    ax.plot(bins["quality_mean"], bins["st_mae_mean"], marker="o", color="#8E5EA2", linewidth=2.2)
    ax.set_xlabel("Predicted quality score")
    ax.set_ylabel("Mean ST-MAE")
    ax.set_title(f"Higher quality -> lower ST-MAE (r = {r_st:.3f})", fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    fig.suptitle(
        "Effectiveness of the Denoising Quality Head",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"quality_head_effectiveness.{suffix}", bbox_inches="tight")
    plt.close(fig)

    plot_single_relationship(
        q,
        cc,
        bins["quality_mean"].to_numpy(dtype=float),
        bins["cc_mean"].to_numpy(dtype=float),
        f"Quality vs CC (r = {r_cc:.3f})",
        "Predicted quality score",
        "Actual CC",
        out_dir / "quality_vs_cc",
        color="#D94B5F",
    )
    plot_single_relationship(
        q,
        fidelity,
        bins["quality_mean"].to_numpy(dtype=float),
        bins["fidelity_mean"].to_numpy(dtype=float),
        f"Quality vs Fidelity (r = {r_fidelity:.3f})",
        "Predicted quality score",
        "Actual fidelity score",
        out_dir / "quality_vs_fidelity",
        color="#D94B5F",
    )


def plot_single_relationship(
    x: np.ndarray,
    y: np.ndarray,
    bin_x: np.ndarray,
    bin_y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_stem: Path,
    color: str,
) -> None:
    rng = np.random.default_rng(0)
    if x.size > 1200:
        idx = rng.choice(x.size, size=1200, replace=False)
    else:
        idx = np.arange(x.size)

    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=260)
    ax.scatter(
        x[idx],
        y[idx],
        s=11,
        alpha=0.23,
        color="#8AA9CC",
        edgecolors="none",
        label="Samples",
    )
    ax.plot(
        bin_x,
        bin_y,
        color=color,
        marker="o",
        markersize=5.2,
        linewidth=2.4,
        label="Quantile-bin mean",
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8.5, loc="upper left", frameon=True)
    fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="v7/paper_experiments/model_comparison_stead/per_model_V7.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="v7/paper_experiments/appendix/quality_head_effectiveness",
    )
    parser.add_argument("--bins", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    needed = [
        "model_quality",
        "fidelity_score",
        "cc",
        "rmse",
        "st_mae_denoised",
    ]
    frame = frame.dropna(subset=needed).reset_index(drop=True)
    bins = build_bins(frame, args.bins)
    bins.to_csv(out_dir / "quality_head_binned_metrics.csv", index=False, encoding="utf-8-sig")
    write_correlation_table(frame, out_dir / "quality_head_correlation.csv")
    plot(frame, bins, out_dir)
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
