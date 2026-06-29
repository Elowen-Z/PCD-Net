# -*- coding: utf-8 -*-
"""Plot non-natural-domain PCD-Net transfer fine-tuning ablation results."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "v7" / "transfer_comparisons" / "nonnat" / "comparison_summary.csv"
OUT_DIR = ROOT / "v7" / "paper_experiments" / "nonnat_transfer_finetune_ablation"

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
        raise SystemExit("No non-natural fine-tuning ablation rows found.")
    df["freeze_strategy"] = pd.Categorical(df["freeze_strategy"], STRATEGIES, ordered=True)
    df = df.sort_values("freeze_strategy").reset_index(drop=True)
    df["method"] = df["freeze_strategy"].astype(str).map(LABELS)
    missing = [strategy for strategy in STRATEGIES if strategy not in set(df["freeze_strategy"].astype(str))]
    if missing:
        print("[WARN] missing strategies:", ",".join(missing))
    return df


def export_table(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
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
    table = df[cols].rename(
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
    table.to_csv(OUT_DIR / "nonnat_transfer_finetune_ablation_table.csv", index=False, encoding="utf-8-sig")
    rounded = table.copy()
    for col in rounded.columns:
        if col != "Fine-tuning strategy":
            rounded[col] = rounded[col].map(lambda x: f"{float(x):.4f}")
    rounded.to_csv(OUT_DIR / "nonnat_transfer_finetune_ablation_table_rounded.csv", index=False, encoding="utf-8-sig")
    rounded.to_latex(OUT_DIR / "nonnat_transfer_finetune_ablation_table.tex", index=False, escape=False)


def plot_metrics(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    colors = ["#8c8c8c", "#4c78a8", "#72b7b2", "#d94b63", "#f58518"][: len(df)]
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
    fig.suptitle("Non-natural Transfer Fine-tuning Ablation of PCD-Net", fontsize=18, fontweight="bold")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"nonnat_transfer_finetune_ablation.{ext}", dpi=420 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    export_table(df)
    plot_metrics(df)
    print(f"[OK] non-natural transfer fine-tuning ablation figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
