# -*- coding: utf-8 -*-
"""Plot mining-domain PCD-Net transfer fine-tuning ablation results."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "v7" / "transfer_comparisons" / "mining" / "comparison_summary.csv"
OUT_DIR = ROOT / "v7" / "paper_experiments" / "mining_transfer_finetune_ablation"

STRATEGIES = [
    "all_frozen",
    "noise_encoder",
    "prototype_feedback",
    "pcd_adaptation",
    "full",
]

LABELS = {
    "all_frozen": "All Frozen",
    "noise_encoder": "Noise Encoder",
    "prototype_feedback": "Prototype-Feedback",
    "pcd_adaptation": "PCD Adaptation",
    "full": "Full Fine-tuning",
}

METRICS = [
    ("val_gain", "Delta SNR (dB)", True),
    ("val_cc", "CC", True),
    ("val_rmse", "RMSE", False),
    ("val_st_mae", "ST-MAE", False),
]


def load_results() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["suite"].astype(str).eq("freeze")].copy()
    df = df[df["freeze_strategy"].isin(STRATEGIES)].copy()
    if df.empty:
        raise SystemExit("No mining fine-tuning ablation rows found.")

    df["freeze_strategy"] = pd.Categorical(df["freeze_strategy"], STRATEGIES, ordered=True)
    df = df.sort_values("freeze_strategy").reset_index(drop=True)
    df["method"] = df["freeze_strategy"].astype(str).map(LABELS)
    return df


def export_table(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = df[
        [
            "method",
            "val_gain",
            "val_cc",
            "val_rmse",
            "val_st_mae",
            "adaptive_gain",
            "adaptive_cc",
            "effective_steps",
            "early_stop_rate",
        ]
    ].rename(
        columns={
            "method": "Fine-tuning strategy",
            "val_gain": "Delta SNR (dB)",
            "val_cc": "CC",
            "val_rmse": "RMSE",
            "val_st_mae": "ST-MAE",
            "adaptive_gain": "Adaptive Delta SNR (dB)",
            "adaptive_cc": "Adaptive CC",
            "effective_steps": "Effective steps",
            "early_stop_rate": "Early-stop rate",
        }
    )
    table.to_csv(OUT_DIR / "mining_transfer_finetune_ablation_table.csv", index=False, encoding="utf-8-sig")

    rounded = table.copy()
    for col in rounded.columns:
        if col != "Fine-tuning strategy":
            rounded[col] = rounded[col].map(lambda x: f"{float(x):.4f}")
    rounded.to_csv(OUT_DIR / "mining_transfer_finetune_ablation_table_rounded.csv", index=False, encoding="utf-8-sig")
    rounded.to_latex(OUT_DIR / "mining_transfer_finetune_ablation_table.tex", index=False, escape=False)


def plot_metrics(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )

    colors = ["#8c8c8c", "#4c78a8", "#72b7b2", "#d94b63", "#f58518"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4), constrained_layout=True)
    axes = axes.ravel()
    x = range(len(df))

    for ax, (metric, ylabel, higher_is_better) in zip(axes, METRICS):
        values = df[metric].astype(float).to_numpy()
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
        best_idx = int(values.argmax() if higher_is_better else values.argmin())
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(1.6)

        for i, value in enumerate(values):
            label = f"{value:.2f}" if metric == "val_gain" else f"{value:.4f}"
            offset = 0.012 * (max(values) - min(values) + 1e-6)
            ax.text(i, value + offset, label, ha="center", va="bottom", fontsize=10)

        ax.set_xticks(list(x))
        ax.set_xticklabels(df["method"], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Mining Transfer Fine-tuning Ablation of PCD-Net", fontsize=18, fontweight="bold")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"mining_transfer_finetune_ablation.{ext}", dpi=420 if ext == "png" else None)
    plt.close(fig)


def plot_radar(df: pd.DataFrame) -> None:
    # Normalize metrics to [0, 1], with larger values always meaning better.
    metrics = ["val_gain", "val_cc", "val_rmse", "val_st_mae"]
    labels = ["Delta SNR", "CC", "RMSE", "ST-MAE"]
    values = df[metrics].astype(float).copy()
    values["val_rmse"] = -values["val_rmse"]
    values["val_st_mae"] = -values["val_st_mae"]
    norm = (values - values.min()) / (values.max() - values.min() + 1e-12)

    angles = [n / float(len(labels)) * 2.0 * 3.141592653589793 for n in range(len(labels))]
    angles += angles[:1]

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = plt.subplot(111, polar=True)
    colors = ["#8c8c8c", "#4c78a8", "#72b7b2", "#d94b63", "#f58518"]
    for idx, row in norm.iterrows():
        vals = row.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.0, label=df.loc[idx, "method"], color=colors[idx])
        ax.fill(angles, vals, alpha=0.10, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Normalized Mining Transfer Performance", fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.36, 1.13), frameon=False)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"mining_transfer_finetune_ablation_radar.{ext}", dpi=420 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    export_table(df)
    plot_metrics(df)
    plot_radar(df)
    print(f"[OK] Mining transfer fine-tuning ablation figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
