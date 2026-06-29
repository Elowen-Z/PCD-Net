# -*- coding: utf-8 -*-
"""Export a single ST-MAE figure for mining transfer fine-tuning ablation."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "v7" / "paper_experiments" / "mining_transfer_finetune_ablation" / "mining_transfer_finetune_ablation_table_rounded.csv"
OUT_DIR = ROOT / "v7" / "paper_experiments" / "mining_transfer_finetune_ablation"


def main() -> None:
    df = pd.read_csv(TABLE)
    df["ST-MAE"] = df["ST-MAE"].astype(float)
    labels = df["Fine-tuning strategy"].tolist()
    values = df["ST-MAE"].to_numpy()
    best_idx = int(values.argmin())

    colors = ["#8c8c8c", "#4c78a8", "#72b7b2", "#d94b63", "#f58518"][: len(df)]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(1.8)

    for idx, value in enumerate(values):
        ax.text(idx, value + 0.006, f"{value:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("ST-MAE")
    ax.set_xlabel("Fine-tuning strategy")
    ax.set_title("ST-MAE of Mining Transfer Fine-tuning Strategies", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=18)
    ax.text(
        best_idx,
        values[best_idx] + 0.025,
        "Best",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#d94b63",
    )

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"mining_transfer_st_mae.{ext}", dpi=420 if ext == "png" else None)
    plt.close(fig)
    print(f"[OK] ST-MAE figure -> {OUT_DIR}")


if __name__ == "__main__":
    main()
