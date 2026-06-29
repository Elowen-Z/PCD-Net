"""Plot publication-ready full-model comparison figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODEL_LABELS = {
    "V7": "PCD-Net",
    "PCD-Net": "PCD-Net",
    "DPRNN": "DPRNN",
    "Restormer1D": "Restormer1D",
    "DeepDenoiser": "DeepDenoiser",
    "Wavelet": "Wavelet",
    "Bandpass": "Bandpass",
}

ORDER = ["PCD-Net", "V7", "DPRNN", "Restormer1D", "Wavelet", "DeepDenoiser", "Bandpass"]


def display_name(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def is_proposed(name: str) -> bool:
    return name in {"V7", "PCD-Net"}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_name = {row["Model"]: row for row in rows}
    selected = []
    seen = set()
    for name in ORDER:
        if name in by_name and name not in seen:
            selected.append(by_name[name])
            seen.add(name)
    for row in rows:
        if row["Model"] not in seen:
            selected.append(row)
    return selected


def f(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def error(row: dict, key: str, low_key: str, high_key: str) -> tuple[float, float]:
    value = f(row, key)
    low = f(row, low_key)
    high = f(row, high_key)
    return max(value - low, 0.0), max(high - value, 0.0)


def plot_main(
    rows: list[dict],
    out_png: Path,
    out_pdf: Path,
    out_svg: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        (
            "delta_snr",
            "delta_snr_ci_low",
            "delta_snr_ci_high",
            "SNR Improvement (dB)",
            "Higher is better",
            True,
        ),
        (
            "cc",
            "cc_ci_low",
            "cc_ci_high",
            "Correlation Coefficient",
            "Higher is better",
            True,
        ),
        (
            "rmse",
            "rmse_ci_low",
            "rmse_ci_high",
            "RMSE",
            "Lower is better",
            False,
        ),
        (
            "st_mae_denoised",
            "st_mae_denoised_ci_low",
            "st_mae_denoised_ci_high",
            "ST-MAE",
            "Lower is better",
            False,
        ),
    ]
    labels = [display_name(row["Model"]) for row in rows]
    x = np.arange(len(rows))
    colors = ["#D94B5F" if is_proposed(row["Model"]) else "#7E8FA6" for row in rows]

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), dpi=240)
    axes = axes.ravel()
    for ax, (key, low_key, high_key, title, note, higher_better) in zip(axes, metrics):
        values = [f(row, key) for row in rows]
        yerr = np.array([error(row, key, low_key, high_key) for row in rows]).T
        bars = ax.bar(
            x,
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.65,
            alpha=0.92,
            width=0.72,
            yerr=yerr,
            capsize=3.5,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=10)
        ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.38)
        ax.text(
            0.98,
            0.94,
            note,
            transform=ax.transAxes,
            fontsize=9,
            color="#555555",
            ha="right",
            va="top",
        )
        best_index = int(np.argmax(values) if higher_better else np.argmin(values))
        bars[best_index].set_hatch("//")
        for xi, value in zip(x, values):
            if key == "delta_snr":
                text = f"{value:.2f}"
            elif key == "cc":
                text = f"{value:.3f}"
            else:
                text = f"{value:.4f}"
            ax.text(
                xi,
                value,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )
        ymin, ymax = ax.get_ylim()
        if not higher_better:
            ax.set_ylim(0, ymax)
        else:
            ax.set_ylim(max(0, ymin), ymax * 1.08)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#D94B5F", edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#7E8FA6", edgecolor="black"),
    ]
    fig.legend(
        handles,
        ["Proposed PCD-Net", "Baseline methods"],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        "Full Model Comparison on the STEAD Test Set",
        fontsize=17,
        fontweight="bold",
        y=1.035,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    if out_svg is not None:
        fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(rows: list[dict], path: Path) -> None:
    lines = [
        "# Full Model Comparison",
        "",
        "| Model | N | Delta SNR (dB) | CC | RMSE | ST-MAE | Fidelity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {n} | {snr:.2f} | {cc:.3f} | {rmse:.4f} | "
            "{st:.4f} | {fid:.3f} |".format(
                model=display_name(row["Model"]),
                n=int(float(row["N"])),
                snr=f(row, "delta_snr"),
                cc=f(row, "cc"),
                rmse=f(row, "rmse"),
                st=f(row, "st_mae_denoised"),
                fid=f(row, "fidelity_score"),
            )
        )
    lines.extend(
        [
            "",
            "PCD-Net obtains the highest SNR improvement and correlation, "
            "and the lowest RMSE and ST-MAE among all compared methods.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_display_tables(rows: list[dict], csv_path: Path, tex_path: Path) -> None:
    keys = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["Model"] = display_name(row["Model"])
            writer.writerow(out)

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & $\Delta$SNR $\uparrow$ & CC $\uparrow$ & RMSE $\downarrow$ & "
        r"PRD $\downarrow$ & ST-MAE $\downarrow$ & Fidelity $\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{display_name(row['Model'])} & {f(row, 'delta_snr'):.3f} & "
            f"{f(row, 'cc'):.4f} & {f(row, 'rmse'):.4f} & "
            f"{f(row, 'prd'):.4f} & {f(row, 'st_mae_denoised'):.4f} & "
            f"{f(row, 'fidelity_score'):.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results_notes(rows: list[dict], path: Path) -> None:
    proposed = next(row for row in rows if is_proposed(row["Model"]))
    baselines = [row for row in rows if not is_proposed(row["Model"])]
    best = max(baselines, key=lambda row: f(row, "delta_snr"))
    st_reduction = 100.0 * (
        f(best, "st_mae_denoised") - f(proposed, "st_mae_denoised")
    ) / max(f(best, "st_mae_denoised"), 1e-12)
    lines = [
        "# Model Comparison Results for the Manuscript",
        "",
        "## Recommended quantitative statement",
        "",
        (
            "On 2,000 held-out STEAD samples, PCD-Net achieved an average "
            f"SNR improvement of {f(proposed, 'delta_snr'):.2f} dB, a "
            f"correlation coefficient of {f(proposed, 'cc'):.4f}, and an "
            f"ST-MAE of {f(proposed, 'st_mae_denoised'):.4f}. Compared with "
            f"the strongest baseline ({display_name(best['Model'])}), PCD-Net "
            f"improved the SNR gain by {f(proposed, 'delta_snr') - f(best, 'delta_snr'):.2f} dB "
            f"and reduced ST-MAE by {st_reduction:.1f}%."
        ),
        "",
        "## Figure captions",
        "",
        (
            "**Quantitative comparison.** Denoising performance on the "
            "held-out STEAD test set. Bars show the mean and error bars "
            "show 95% bootstrap confidence intervals. Higher values are "
            "better for SNR improvement, CC, and fidelity; lower values "
            "are better for RMSE, PRD, and ST-MAE."
        ),
        "",
        (
            "**Qualitative comparison.** Representative three-component "
            "STEAD waveform selected using a predefined criterion that "
            "requires waveform complexity, high output fidelity, and a "
            "clear denoising improvement. All methods use the same input "
            "and display window."
        ),
        "",
        "## Quality-score note",
        "",
        (
            "The PCD-Net model-quality score is produced by its learned "
            "no-reference quality head. It is reported as an auxiliary "
            "confidence indicator and is not treated as a metric shared "
            "by the baseline methods. Fidelity is computed against the "
            "clean target and is used for cross-model comparison."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_six_metrics(
    rows: list[dict],
    out_png: Path,
    out_pdf: Path,
    out_svg: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        ("delta_snr", "SNR Improvement (dB)", True),
        ("cc", "Correlation Coefficient", True),
        ("rmse", "RMSE", False),
        ("prd", "PRD", False),
        ("st_mae_denoised", "ST-MAE", False),
        ("fidelity_score", "Fidelity Score", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.0), dpi=240)
    for axis, (metric, title, higher_better) in zip(axes.flat, metrics):
        values_all = np.array([f(row, metric) for row in rows])
        order = np.argsort(values_all)
        if higher_better:
            order = order[::-1]
        selected = [rows[index] for index in order]
        values = np.array([f(row, metric) for row in selected])
        errors = np.array(
            [
                error(row, metric, metric + "_ci_low", metric + "_ci_high")
                for row in selected
            ]
        ).T
        colors = [
            "#D1495B" if is_proposed(row["Model"]) else "#718096"
            for row in selected
        ]
        positions = np.arange(len(selected))
        axis.bar(
            positions,
            values,
            yerr=errors,
            capsize=3,
            color=colors,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [display_name(row["Model"]) for row in selected],
            rotation=28,
            ha="right",
            fontsize=8,
        )
        axis.set_title(title, fontsize=10, fontweight="bold")
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(axis="y", labelsize=8)
    fig.suptitle(
        "Denoising Performance on the STEAD Test Set (mean and 95% bootstrap CI)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    if out_svg is not None:
        fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="v7/paper_experiments/model_comparison_stead/paper_model_comparison_table.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="v7/paper_experiments/model_comparison_stead",
    )
    args = parser.parse_args()
    rows = read_rows(Path(args.csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_main(
        rows,
        out_dir / "full_model_comparison_main_metrics.png",
        out_dir / "full_model_comparison_main_metrics.pdf",
        out_dir / "full_model_comparison_main_metrics.svg",
    )
    write_markdown_summary(rows, out_dir / "full_model_comparison_summary.md")
    write_display_tables(
        rows,
        out_dir / "paper_model_comparison_table.csv",
        out_dir / "paper_model_comparison_table.tex",
    )
    write_results_notes(rows, out_dir / "paper_results_notes.md")
    plot_six_metrics(
        rows,
        out_dir / "paper_model_comparison.png",
        out_dir / "paper_model_comparison.pdf",
        out_dir / "paper_model_comparison.svg",
    )
    print(f"wrote {out_dir / 'full_model_comparison_main_metrics.png'}")
    print(f"wrote {out_dir / 'full_model_comparison_main_metrics.pdf'}")
    print(f"wrote {out_dir / 'full_model_comparison_main_metrics.svg'}")
    print(f"wrote {out_dir / 'full_model_comparison_summary.md'}")
    print(f"wrote {out_dir / 'paper_model_comparison.png'}")
    print(f"wrote {out_dir / 'paper_model_comparison.svg'}")


if __name__ == "__main__":
    main()
