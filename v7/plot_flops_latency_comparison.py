"""Plot FLOPs and inference-latency comparison for appendix."""

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


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value or value == "-":
        return None
    if value.startswith("<"):
        return float(value[1:])
    return float(value)


def fmt_value(value: str, decimals: int = 3) -> str:
    value = value.strip()
    if not value or value == "-":
        return "-"
    if value.startswith("<"):
        return value
    return f"{float(value):.{decimals}f}"


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["Method"]: row for row in rows}


def write_tables(data: dict[str, dict[str, str]], out_dir: Path) -> list[dict[str, str]]:
    rows = []
    for method in METHOD_ORDER:
        source = data[method]
        rows.append(
            {
                "Method": method,
                "FLOPs_G": fmt_value(source["FLOPs_G"], 3),
                "Latency_ms": fmt_value(source["Latency_ms"], 3),
                "Throughput_samples_per_s": fmt_value(
                    source["Throughput_samples_per_s"], 3
                ),
            }
        )

    csv_path = out_dir / "flops_latency_comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Method",
                "FLOPs_G",
                "Latency_ms",
                "Throughput_samples_per_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| Method | FLOPs (G) | Latency (ms/sample) | Throughput (samples/s) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['Method']} | {row['FLOPs_G']} | {row['Latency_ms']} | "
            f"{row['Throughput_samples_per_s']} |"
        )
    (out_dir / "flops_latency_comparison.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    tex_lines = [
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Method & FLOPs (G) & Latency (ms/sample) & Throughput (samples/s) \\",
        r"\hline",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['Method']} & {row['FLOPs_G']} & {row['Latency_ms']} & "
            f"{row['Throughput_samples_per_s']} \\\\"
        )
    tex_lines.extend([r"\hline", r"\end{tabular}"])
    (out_dir / "flops_latency_comparison.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )
    return rows


def draw_panel(
    ax,
    labels: list[str],
    raw_values: list[float | None],
    ylabel: str,
    title: str,
    na_text_y: float,
) -> None:
    plot_values = [0.0 if value is None else value for value in raw_values]
    colors = [COLORS[label] for label in labels]
    x = np.arange(len(labels))
    bars = ax.bar(x, plot_values, color=colors, edgecolor="black", linewidth=0.8)

    positive_values = [value for value in plot_values if value > 0]
    ymax = max(positive_values) * 1.28 if positive_values else 1.0
    ax.set_ylim(0, ymax)

    for bar, value, label in zip(bars, raw_values, labels):
        if value is None:
            bar.set_facecolor("#dddddd")
            bar.set_hatch("//")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                na_text_y * ymax,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        else:
            text = "<0.001" if 0 < value <= 0.001 and ylabel.startswith("FLOPs") else f"{value:.2f}"
            if ylabel.startswith("FLOPs") and value >= 0.01:
                text = f"{value:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.025,
                text,
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold" if label == "PCD-Net" else "normal",
                color="#c43b52" if label == "PCD-Net" else "black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def plot(rows: list[dict[str, str]], out_dir: Path) -> None:
    labels = [row["Method"] for row in rows]
    flops = [parse_float(row["FLOPs_G"]) for row in rows]
    latency = [parse_float(row["Latency_ms"]) for row in rows]

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 13,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))
    draw_panel(
        axes[0],
        labels,
        flops,
        "FLOPs (G)",
        "Computational Cost",
        na_text_y=0.04,
    )
    draw_panel(
        axes[1],
        labels,
        latency,
        "Latency (ms/sample)",
        "Inference Latency",
        na_text_y=0.04,
    )
    axes[0].text(
        0.01,
        0.97,
        "N/A: non-neural signal-processing baseline.",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.suptitle(
        "FLOPs and Inference-Time Comparison",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"flops_latency_comparison.{suffix}", dpi=300)
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
        default="v7/paper_experiments/appendix/model_flops_latency_comparison",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = read_rows(Path(args.input_csv))
    rows = write_tables(data, out_dir)
    plot(rows, out_dir)
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
