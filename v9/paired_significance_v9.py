"""Paired significance testing for the PCD-Net V9 model comparison.

Reads the ``per_model_<Name>.csv`` files written by ``v9/compare_v9.py``.  Those
files are paired by construction: one shared ``EvalDataset`` instance supplies
the same noisy waveform, the same sampled input SNR and the same conditioning
window for every method, in the same row order.  The script verifies the pairing
(equal ``N`` and an identical ``snr_in`` column) before testing, then reports,
for every baseline and every metric:

* paired mean difference (proposed - baseline)
* percentile bootstrap 95% confidence interval of that difference
* two-sided Wilcoxon signed-rank p-value
* Holm-Bonferroni corrected p-value across baselines

Usage
-----
python -m v9.paired_significance_v9 \
    --input_dir v9/paper_experiments/model_comparison_stead_snr_m15_p10
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# metric -> (higher_is_better, decimals for the LaTeX table)
METRIC_SPECS = {
    "delta_snr": (True, 3, r"$\Delta$SNR (dB)"),
    "cc": (True, 4, "CC"),
    "rmse": (False, 4, "RMSE"),
    "prd": (False, 4, "PRD"),
    "st_mae_denoised": (False, 4, "ST-MAE"),
    "fidelity_score": (True, 4, "Fidelity"),
}

PAIRING_KEY = "snr_in"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired significance test for the PCD-Net V9 comparison"
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--proposed", default="PCD-Net")
    parser.add_argument(
        "--baselines",
        default=None,
        help="comma separated list; defaults to every per_model_*.csv found",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output_prefix",
        default="pairwise_significance",
        help="base name for the .csv and .tex outputs inside --input_dir",
    )
    return parser


def bootstrap_ci(values: np.ndarray, seed: int, repetitions: int):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        means[index] = rng.choice(values, size=values.size, replace=True).mean()
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def load_frames(input_dir: Path, proposed: str, baselines: list[str]):
    frames = {}
    for name in [proposed, *baselines]:
        path = input_dir / f"per_model_{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"per-model file not found: {path}")
        frames[name] = pd.read_csv(path)
    return frames


def verify_pairing(frames: dict[str, pd.DataFrame], proposed: str) -> int:
    reference = frames[proposed]
    size = len(reference)
    for name, frame in frames.items():
        if len(frame) != size:
            raise RuntimeError(
                f"{name} has {len(frame)} rows but {proposed} has {size}; "
                "the comparison is not paired"
            )
        deviation = np.abs(
            frame[PAIRING_KEY].to_numpy() - reference[PAIRING_KEY].to_numpy()
        )
        if deviation.size and float(deviation.max()) > 1e-6:
            raise RuntimeError(
                f"{name} does not share the {PAIRING_KEY} vector of {proposed} "
                "(max deviation "
                f"{float(deviation.max()):.3e}); the comparison is not paired"
            )
    print(
        f"[pairing] verified: {len(frames)} methods, N={size}, identical "
        f"'{PAIRING_KEY}' vector",
        flush=True,
    )
    return size


def stars(p_value: float) -> str:
    if not math.isfinite(p_value):
        return ""
    if p_value < 1e-3:
        return r"$^{***}$"
    if p_value < 1e-2:
        return r"$^{**}$"
    if p_value < 0.05:
        return r"$^{*}$"
    return ""


def analyze(frames, proposed: str, baselines: list[str], args):
    reference = frames[proposed]
    rows = []
    for metric, (higher_better, _, _) in METRIC_SPECS.items():
        metric_pvalues = []
        metric_indexes = []
        for baseline in baselines:
            other = frames[baseline]
            diff = (
                reference[metric].to_numpy(dtype=np.float64)
                - other[metric].to_numpy(dtype=np.float64)
            )
            finite = np.isfinite(diff)
            diff = diff[finite]
            mean_diff, ci_low, ci_high = bootstrap_ci(
                diff, args.seed, args.repetitions
            )
            win_rate = float((diff > 0).mean()) if diff.size else float("nan")
            try:
                if diff.size and np.any(diff != 0):
                    p_value = float(
                        stats.wilcoxon(diff, alternative="two-sided").pvalue
                    )
                else:
                    p_value = 1.0
            except ValueError:
                p_value = 1.0
            better = (mean_diff > 0) if higher_better else (mean_diff < 0)
            rows.append(
                {
                    "metric": metric,
                    "baseline": baseline,
                    "n": int(diff.size),
                    "proposed_mean": float(
                        np.nanmean(reference[metric].to_numpy(dtype=np.float64))
                    ),
                    "baseline_mean": float(
                        np.nanmean(other[metric].to_numpy(dtype=np.float64))
                    ),
                    "mean_diff_proposed_minus_baseline": mean_diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wilcoxon_p": p_value,
                    "proposed_win_rate": win_rate,
                    "higher_is_better": higher_better,
                    "proposed_better": bool(better),
                }
            )
            metric_pvalues.append(p_value)
            metric_indexes.append(len(rows) - 1)
        for index, adjusted in zip(
            metric_indexes, holm_adjust(metric_pvalues)
        ):
            rows[index]["holm_p"] = adjusted
            rows[index]["significant_holm"] = bool(adjusted < args.alpha)
    return pd.DataFrame(rows)


def write_latex(table: pd.DataFrame, baselines: list[str], path: Path) -> None:
    header_names = {key: spec[2] for key, spec in METRIC_SPECS.items()}
    lines = [
        r"\begin{tabular}{l" + "c" * len(METRIC_SPECS) + "}",
        r"\toprule",
        "Baseline & "
        + " & ".join(header_names[key] for key in METRIC_SPECS)
        + r" \\",
        r"\midrule",
    ]
    for baseline in baselines:
        cells = []
        for metric in METRIC_SPECS:
            row = table[(table["metric"] == metric) & (table["baseline"] == baseline)]
            if row.empty:
                cells.append("--")
                continue
            row = row.iloc[0]
            cells.append(f"{row['mean_diff_proposed_minus_baseline']:+.3f}"
                         + stars(float(row["holm_p"])))
        lines.append(f"{baseline} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    p_lines = [
        r"\begin{tabular}{l" + "c" * len(METRIC_SPECS) + "}",
        r"\toprule",
        "Baseline & "
        + " & ".join(header_names[key] for key in METRIC_SPECS)
        + r" \\",
        r"\midrule",
    ]
    for baseline in baselines:
        cells = []
        for metric in METRIC_SPECS:
            row = table[(table["metric"] == metric) & (table["baseline"] == baseline)]
            if row.empty:
                cells.append("--")
                continue
            value = float(row.iloc[0]["holm_p"])
            cells.append("<0.001" if value < 1e-3 else f"{value:.3f}")
        p_lines.append(f"{baseline} & " + " & ".join(cells) + r" \\")
    p_lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.with_name(path.stem + "_pvalues.tex").write_text(
        "\n".join(p_lines) + "\n", encoding="utf-8"
    )


def write_markdown(table: pd.DataFrame, proposed: str, path: Path) -> None:
    lines = [
        f"# Paired significance: {proposed} vs baselines",
        "",
        "Mean paired difference (`proposed - baseline`), 95% bootstrap CI and "
        "Holm-Bonferroni corrected Wilcoxon signed-rank p-value. For RMSE / PRD / "
        "ST-MAE a negative difference favours the proposed model.",
        "",
    ]
    for metric, (higher_better, decimals, label) in METRIC_SPECS.items():
        sub = table[table["metric"] == metric]
        if sub.empty:
            continue
        direction = "higher is better" if higher_better else "lower is better"
        lines.append(f"## {label} ({direction})")
        lines.append("")
        lines.append(
            "| Baseline | diff | 95% CI | win rate | Holm p | significant |"
        )
        lines.append("|---|---:|---|---:|---:|:--:|")
        for _, row in sub.iterrows():
            lines.append(
                f"| {row['baseline']} | "
                f"{row['mean_diff_proposed_minus_baseline']:+.4f} | "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
                f"{row['proposed_win_rate']:.3f} | "
                f"{row['holm_p']:.3g} | "
                f"{'yes' if row['significant_holm'] else 'no'} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input_dir is not a directory: {input_dir}")

    available = sorted(
        path.stem.replace("per_model_", "")
        for path in input_dir.glob("per_model_*.csv")
    )
    if not available:
        raise FileNotFoundError(f"no per_model_*.csv found in {input_dir}")
    if args.proposed not in available:
        raise FileNotFoundError(
            f"{args.proposed} not found in {input_dir}; available: {available}"
        )
    if args.baselines:
        baselines = [
            item.strip() for item in args.baselines.split(",") if item.strip()
        ]
    else:
        baselines = [name for name in available if name != args.proposed]
    unknown = [name for name in baselines if name not in available]
    if unknown:
        raise ValueError(f"baselines not found: {unknown}; available: {available}")

    print(f"[input]     {input_dir}")
    print(f"[proposed]  {args.proposed}")
    print(f"[baselines] {baselines}")

    frames = load_frames(input_dir, args.proposed, baselines)
    verify_pairing(frames, args.proposed)

    table = analyze(frames, args.proposed, baselines, args)
    csv_path = input_dir / f"{args.output_prefix}.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    tex_path = input_dir / f"{args.output_prefix}.tex"
    write_latex(table, baselines, tex_path)
    write_markdown(table, args.proposed, input_dir / f"{args.output_prefix}.md")

    print()
    for metric, (higher_better, _, label) in METRIC_SPECS.items():
        sub = table[table["metric"] == metric]
        if sub.empty:
            continue
        print(f"  {label} ({'higher' if higher_better else 'lower'} is better)")
        for _, row in sub.iterrows():
            flag = "OK" if row["proposed_better"] else "!!"
            print(
                f"    [{flag}] {row['baseline']:<14} "
                f"diff={row['mean_diff_proposed_minus_baseline']:+.4f} "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] "
                f"win={row['proposed_win_rate']:.3f} "
                f"p_holm={row['holm_p']:.3g} "
                f"{'significant' if row['significant_holm'] else 'not-significant'}"
            )

    print(f"\n[done] {csv_path}")
    print(f"[done] {tex_path}")
    print(f"[done] {tex_path.with_name(tex_path.stem + '_pvalues.tex')}")
    print(f"[done] {input_dir / (args.output_prefix + '.md')}")


if __name__ == "__main__":
    main()
