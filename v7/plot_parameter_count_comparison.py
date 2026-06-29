"""Plot parameter-count comparison for appendix.

This script reuses the measured complexity CSV and creates a compact
parameter-only table and figure. Classical signal-processing baselines are
kept in the table but marked as not applicable in the plot.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "Bandpass",
    "Wavelet",
    "DeepDenoiser",
    "DPRNN",
    "Restormer1D",
    "PCD-Net",
]


COLORS = {
    "Bandpass": "#8e5ea2",
    "Wavelet": "#59a14f",
    "DeepDenoiser": "#ff7f0e",
    "DPRNN": "#4e79a7",
    "Restormer1D": "#76b7b2",
    "PCD-Net": "#d94b63",
}


def read_complexity_table(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["Method"]: row["Parameters_M"] for row in rows}


def format_params(value: str) -> str:
    if value in {"", "-"}:
        return "-"
    return f"{float(value):.3f}"


def write_parameter_tables(params: dict[str, str], out_dir: Path) -> list[dict[str, str]]:
    rows = []
    for method in METHOD_ORDER:
        value = format_params(params.get(method, "-"))
        rows.append({"Method": method, "Parameters_M": value})

    csv_path = out_dir / "parameter_count_comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Method", "Parameters_M"])
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| Method | Parameters (M) |",
        "|---|---:|",
    ]
    for row in rows:
        md_lines.append(f"| {row['Method']} | {row['Parameters_M']} |")
    (out_dir / "parameter_count_comparison.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    tex_lines = [
        r"\begin{tabular}{lr}",
        r"\hline",
        r"Method & Parameters (M) \\",
        r"\hline",
    ]
    for row in rows:
        tex_lines.append(f"{row['Method']} & {row['Parameters_M']} \\\\")
    tex_lines.extend([r"\hline", r"\end{tabular}"])
    (out_dir / "parameter_count_comparison.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )
    return rows


def plot_parameter_counts(rows: list[dict[str, str]], out_dir: Path) -> None:
    labels = [row["Method"] for row in rows]
    values = [
        np.nan if row["Parameters_M"] == "-" else float(row["Parameters_M"])
        for row in rows
    ]
    plot_values = [0.0 if np.isnan(value) else value for value in values]
    colors = [COLORS[label] for label in labels]

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 14,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, plot_values, color=colors, edgecolor="black", linewidth=0.8)

    for bar, value, label in zip(bars, values, labels):
        if np.isnan(value):
            bar.set_facecolor("#dddddd")
            bar.set_hatch("//")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.08,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.06,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold" if label == "PCD-Net" else "normal",
                color="#c43b52" if label == "PCD-Net" else "black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Parameters (M)")
    ax.set_title("Model Parameter Count Comparison", fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(0, max(v for v in plot_values) * 1.22)
    ax.text(
        0.01,
        0.97,
        "Classical signal-processing methods have no trainable parameters.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#555555",
    )

    fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"parameter_count_comparison.{suffix}", dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default=(
            "v7/paper_experiments/appendix/model_complexity_latency_comparison/"
            "model_complexity_latency_table.csv"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="v7/paper_experiments/appendix/model_parameter_comparison",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = read_complexity_table(input_csv)
    rows = write_parameter_tables(params, out_dir)
    plot_parameter_counts(rows, out_dir)
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
