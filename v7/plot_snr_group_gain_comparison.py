"""Plot SNR-improvement comparison across input-SNR groups."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHOD_ORDER = [
    "PCD-Net",
    "DPRNN",
    "Restormer1D",
    "Wavelet",
    "DeepDenoiser",
    "Bandpass",
]

GROUP_ORDER = ["<-5", "-5~0", "0~5", "5~10"]

COLORS = {
    "PCD-Net": "#D94B5F",
    "DPRNN": "#4C78A8",
    "Restormer1D": "#72B7B2",
    "Wavelet": "#59A14F",
    "DeepDenoiser": "#F58518",
    "Bandpass": "#8E5EA2",
}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty csv: {path}")
    return rows


def delta_key(rows: list[dict]) -> str:
    keys = rows[0].keys()
    for key in ("DeltaSNR", "delta_snr", "ΔSNR", "螖SNR"):
        if key in keys:
            return key
    for key in keys:
        lowered = key.lower()
        if "snr" in lowered and key not in {"SNR_in", "SNR_out", "SNR_Group"}:
            return key
    raise KeyError("cannot find delta-SNR column")


def normalize_model_name(name: str) -> str:
    return "PCD-Net" if name == "V7" else name


def build_matrix(rows: list[dict], key: str) -> dict[str, list[float]]:
    lookup = {}
    for row in rows:
        model = normalize_model_name(row["Model"])
        group = row["SNR_Group"]
        lookup[(model, group)] = float(row[key])
    return {
        model: [lookup.get((model, group), float("nan")) for group in GROUP_ORDER]
        for model in METHOD_ORDER
    }


def plot_line(matrix: dict[str, list[float]], out_png: Path, out_pdf: Path, out_svg: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(GROUP_ORDER))
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.3), dpi=240)
    for model in METHOD_ORDER:
        values = matrix[model]
        lw = 2.7 if model == "PCD-Net" else 1.8
        marker_size = 7.2 if model == "PCD-Net" else 5.6
        zorder = 5 if model == "PCD-Net" else 3
        ax.plot(
            x,
            values,
            marker="o",
            markersize=marker_size,
            linewidth=lw,
            color=COLORS[model],
            label=model,
            zorder=zorder,
        )
        if model == "PCD-Net":
            for xi, value in zip(x, values):
                ax.text(
                    xi,
                    value + 0.45,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color=COLORS[model],
                )
    ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, fontsize=11)
    ax.set_xlabel("Input SNR Range (dB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("SNR Improvement (dB)", fontsize=12, fontweight="bold")
    ax.set_title(
        "SNR Improvement across Input-SNR Ranges",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.38)
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        fontsize=9,
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bar(matrix: dict[str, list[float]], out_png: Path, out_pdf: Path, out_svg: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(GROUP_ORDER))
    width = 0.13
    offsets = (np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=240)
    for index, model in enumerate(METHOD_ORDER):
        ax.bar(
            x + offsets[index],
            matrix[model],
            width=width,
            color=COLORS[model],
            label=model,
            alpha=0.92,
            edgecolor="black",
            linewidth=0.35,
            hatch="//" if model == "PCD-Net" else None,
        )
    ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, fontsize=11)
    ax.set_xlabel("Input SNR Range (dB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("SNR Improvement (dB)", fontsize=12, fontweight="bold")
    ax.set_title(
        "SNR Improvement of Different Denoising Methods under Input-SNR Groups",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.38)
    ax.legend(loc="upper right", ncol=3, fontsize=9, frameon=True, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def write_clean_csv(matrix: dict[str, list[float]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", *GROUP_ORDER])
        for method in METHOD_ORDER:
            writer.writerow([method, *[f"{value:.4f}" for value in matrix[method]]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="v7/paper_experiments/model_comparison_stead/per_group_compare.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="v7/paper_experiments/model_comparison_stead",
    )
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    key = delta_key(rows)
    matrix = build_matrix(rows, key)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_clean_csv(matrix, out_dir / "snr_group_gain_table.csv")
    plot_line(
        matrix,
        out_dir / "snr_group_gain_line.png",
        out_dir / "snr_group_gain_line.pdf",
        out_dir / "snr_group_gain_line.svg",
    )
    plot_grouped_bar(
        matrix,
        out_dir / "snr_group_gain_bar.png",
        out_dir / "snr_group_gain_bar.pdf",
        out_dir / "snr_group_gain_bar.svg",
    )
    print(f"wrote {out_dir / 'snr_group_gain_table.csv'}")
    print(f"wrote {out_dir / 'snr_group_gain_line.png'}")
    print(f"wrote {out_dir / 'snr_group_gain_bar.png'}")


if __name__ == "__main__":
    main()
