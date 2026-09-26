"""Inference-time routing-temperature sweep and joint K/tau sensitivity plot.

The temperature sweep reuses one trained checkpoint and one deterministic
evaluation set. Prototype-count results are supplied as a CSV because changing
K requires independently trained checkpoints and must not be digitized from a
rendered figure for publication metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from v5.evaluate_transfer_v5 import EvalDataset
from v7.train_v7 import configure_cuda_backend
from v9.compare_v9 import build_group_spec, build_v9_denoiser, evaluate_method


DEFAULT_CHECKPOINT = "v9/experiments/snr_m15_p10_seed0/best_model_v9.pth"
DEFAULT_OUTPUT = "v9/paper_experiments/routing_sensitivity_v9"
METRIC_COLUMNS = {
    "DeltaSNR_dB": ("delta_snr", "ΔSNR (dB)", True),
    "CC": ("cc", "CC", True),
    "RMSE": ("rmse", "RMSE", False),
    "ST_MAE": ("st_mae_denoised", "ST-MAE", False),
}


def sha1_16(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("routing temperatures must be positive")
    return math.log(math.expm1(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--event_h5", default="E:/p_wave/data/chunk2.hdf5")
    parser.add_argument("--event_csv", default="v5/v7_seed0_val.csv")
    parser.add_argument("--noise_h5", default="E:/p_wave/data/chunk1.hdf5")
    parser.add_argument("--noise_csv", default="E:/p_wave/data/chunk1.csv")
    parser.add_argument("--context_source", choices=("pre_event", "paired_noise"), default="pre_event")
    parser.add_argument("--snr_min_db", type=float, default=-15.0)
    parser.add_argument("--snr_max_db", type=float, default=10.0)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--signal_len", type=int, default=6000)
    parser.add_argument("--cond_len", type=int, default=400)
    parser.add_argument("--tau_values", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 1.0])
    parser.add_argument("--k_results_csv", default="", help="CSV with K, DeltaSNR_dB, CC, RMSE, ST_MAE")
    parser.add_argument("--tau_results_csv", default="", help="existing temperature-sweep summary CSV")
    parser.add_argument("--plot_only", action="store_true", help="plot existing K and tau CSVs without inference")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--cuda_safe", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def canonicalize_k_results(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    aliases = {column.lower().replace(" ", "_"): column for column in frame.columns}
    expected = {
        "K": ("k", "num_prototypes", "prototype_number"),
        "DeltaSNR_dB": ("deltasnr_db", "delta_snr_db", "deltasnr", "delta_snr"),
        "CC": ("cc",),
        "RMSE": ("rmse",),
        "ST_MAE": ("st_mae", "st-mae"),
    }
    rename = {}
    for target, options in expected.items():
        found = next((aliases[key] for key in options if key in aliases), None)
        if found is None:
            raise ValueError(f"K result CSV is missing {target}; found columns: {list(frame.columns)}")
        rename[found] = target
    frame = frame.rename(columns=rename)[list(expected)]
    frame = frame.apply(pd.to_numeric, errors="raise")
    frame = frame.sort_values("K")
    if frame["K"].duplicated().any():
        raise ValueError("K result CSV must contain one summary row per prototype count")
    return frame


def write_joint_plot_overlay(k_csv: str, tau_csv: Path, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    k_frame = canonicalize_k_results(k_csv)
    tau_frame = pd.read_csv(tau_csv).sort_values("tau")
    learned_tau = float(tau_frame["learned_checkpoint_tau"].iloc[0])
    tau_plot = tau_frame.copy()
    close_to_learned = (tau_plot["tau"] - learned_tau).abs() < 0.01
    tau_plot = tau_plot.loc[~(close_to_learned & ((tau_plot["tau"] - learned_tau).abs() > 1e-8))]
    figure, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), constrained_layout=True)
    colors = {"k": "#3274a1", "tau": "#e1812c"}

    for col, (column, (_sample_column, label, higher_is_better)) in enumerate(METRIC_COLUMNS.items()):
        axis = axes[col]
        axis.plot(k_frame["K"], k_frame[column], marker="o", linewidth=1.8,
                  markersize=5, color=colors["k"], label="K sweep (V7 Top-M)")
        tau_axis = axis.twiny()
        tau_axis.plot(tau_plot["tau"], tau_plot[column], marker="s", linewidth=1.8,
                      markersize=4.5, color=colors["tau"], label="τ sweep (V9 inference)")
        if 16 in set(k_frame["K"].tolist()):
            baseline = k_frame.loc[k_frame["K"] == 16].iloc[0]
            axis.scatter([16], [baseline[column]], color="#c43c39", s=45, zorder=3,
                         label="K=16")
        axis.set_xlabel("Prototype number K (bottom axis)")
        tau_axis.set_xlabel("τ", labelpad=1)
        axis.set_ylabel(label)
        axis.set_title(label, pad=22)
        axis.set_xticks(k_frame["K"].tolist())
        tau_axis.set_xticks(tau_plot["tau"].tolist())
        tau_axis.set_xticklabels([f"{value:.2f}" for value in tau_plot["tau"]])
        axis.grid(True, linestyle=":", alpha=0.55)
        direction = "higher is better" if higher_is_better else "lower is better"
        axis.text(0.98, 0.03, direction, transform=axis.transAxes,
                  ha="right", va="bottom", fontsize=8, color="#555555")

    legend_handles = [
        Line2D([0], [0], color=colors["k"], marker="o", label="K sweep (V7 Top-M)"),
        Line2D([0], [0], color=colors["tau"], marker="s", label="τ sweep (V9 inference)"),
        Line2D([0], [0], color="#c43c39", marker="o", linestyle="None", label="K=16"),
    ]
    figure.legend(handles=legend_handles, loc="upper center", ncol=3,
                  frameon=False, bbox_to_anchor=(0.5, 1.12))
    output_path = output_dir / "routing_sensitivity_combined.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "routing_sensitivity_combined.pdf", bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_joint_plot(k_csv: str, tau_csv: Path, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k_frame = canonicalize_k_results(k_csv)
    tau_frame = pd.read_csv(tau_csv).sort_values("tau")
    learned_tau = float(tau_frame["learned_checkpoint_tau"].iloc[0])
    close_to_learned = (tau_frame["tau"] - learned_tau).abs() < 0.01
    tau_plot = tau_frame.loc[
        ~(close_to_learned & ((tau_frame["tau"] - learned_tau).abs() > 1e-8))
    ].copy()

    figure, axes = plt.subplots(2, 4, figsize=(13.2, 6.0))
    blue = "#3274a1"
    for col, (column, (_sample_column, title, _higher_is_better)) in enumerate(METRIC_COLUMNS.items()):
        for row, (frame, x_column, x_label) in enumerate(
            ((k_frame, "K", "K"), (tau_plot, "tau", chr(0x03C4)))
        ):
            axis = axes[row, col]
            axis.plot(frame[x_column], frame[column], marker="o", linewidth=1.8,
                      markersize=5, color=blue)
            if row == 0 and 16 in set(frame["K"].tolist()):
                point = frame.loc[frame["K"] == 16].iloc[0]
                axis.scatter([16], [point[column]], color="#c43c39", s=45, zorder=3)
            elif row == 1:
                point = frame.iloc[(frame["tau"] - learned_tau).abs().argmin()]
                axis.scatter([point["tau"]], [point[column]], color="#c43c39", s=45, zorder=3)
            axis.set_title(title, fontweight="bold", pad=5)
            axis.set_xlabel(x_label)
            axis.grid(True, linestyle=":", alpha=0.55)
            if x_column == "K":
                axis.set_xticks(frame["K"].tolist())
            else:
                axis.set_xticks(frame["tau"].tolist())
                axis.set_xticklabels([f"{value:.1f}" for value in frame["tau"]])

    figure.text(0.018, 0.73, "Prototype number K", rotation=90,
                va="center", ha="center", fontsize=10)
    figure.text(0.018, 0.28, f"Routing temperature {chr(0x03C4)}", rotation=90,
                va="center", ha="center", fontsize=10)
    figure.subplots_adjust(left=0.10, right=0.99, bottom=0.09, top=0.96,
                           hspace=0.42, wspace=0.34)
    output_path = output_dir / "routing_sensitivity_combined.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "routing_sensitivity_combined.pdf", bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        if not args.k_results_csv or not args.tau_results_csv:
            raise ValueError("--plot_only requires both --k_results_csv and --tau_results_csv")
        plot_path = write_joint_plot(
            args.k_results_csv, Path(args.tau_results_csv), output_dir
        )
        print(f"[saved] combined figure: {plot_path}", flush=True)
        return
    if args.snr_min_db >= args.snr_max_db:
        raise ValueError("snr_min_db must be smaller than snr_max_db")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    tau_values = sorted(set(float(value) for value in args.tau_values))
    if any(value <= 0 for value in tau_values):
        raise ValueError("all tau_values must be positive")

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    configure_cuda_backend(args.cuda_safe and device.type == "cuda")
    dataset = EvalDataset(
        event_h5_path=args.event_h5,
        event_csv_path=args.event_csv,
        noise_h5_path=args.noise_h5,
        noise_csv_path=args.noise_csv,
        signal_len=args.signal_len,
        cond_len=args.cond_len,
        snr_db_range=(args.snr_min_db, args.snr_max_db),
        noise_boost=1.0,
        max_samples=args.max_samples,
        seed=args.seed,
        context_source=args.context_source,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    labels, _, assign_group = build_group_spec(None)
    del labels
    denoiser = build_v9_denoiser(args.checkpoint, device)
    router = denoiser.model.router
    original_tau = float(router.temperature.detach().cpu())
    config = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("config", {})
    tau_values = sorted(set(tau_values + [original_tau]))
    print(f"[device] {device} cuda_safe={args.cuda_safe}", flush=True)
    print(f"[dataset] N={len(dataset)}; learned checkpoint tau={original_tau:.6g}", flush=True)

    summaries = []
    reference_snr = None
    for tau in tau_values:
        with torch.no_grad():
            router.raw_temperature.fill_(inverse_softplus(tau))
        denoiser.name = f"tau={tau:g}"
        records, snr_seen, elapsed = evaluate_method(
            denoiser, loader, device,
            {"fs": 100.0, "sta_len": 0.5, "lta_len": 10.0,
             "stalta_thr": 2.0, "pick_tol": 50},
            assign_group, expected_snr_in=reference_snr,
        )
        if reference_snr is None:
            reference_snr = snr_seen
        frame = pd.DataFrame(records)
        summary = {
            "tau": tau,
            "learned_checkpoint_tau": original_tau,
            "N": len(frame),
            "DeltaSNR_dB": float(frame["delta_snr"].mean()),
            "CC": float(frame["cc"].mean()),
            "RMSE": float(frame["rmse"].mean()),
            "ST_MAE": float(frame["st_mae_denoised"].mean()),
            "time_sec": elapsed,
        }
        summaries.append(summary)
        frame.to_csv(output_dir / f"per_sample_tau_{tau:g}.csv", index=False, encoding="utf-8-sig")
        print(summary, flush=True)

    tau_path = output_dir / "routing_temperature_sensitivity.csv"
    pd.DataFrame(summaries).to_csv(tau_path, index=False, encoding="utf-8-sig")
    audit = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha1_16": sha1_16(args.checkpoint),
        "checkpoint_config": config,
        "learned_checkpoint_tau": original_tau,
        "inference_tau_values": tau_values,
        "protocol": {
            "event_h5": args.event_h5, "event_csv": args.event_csv,
            "noise_h5": args.noise_h5, "noise_csv": args.noise_csv,
            "context_source": args.context_source,
            "snr_db": [args.snr_min_db, args.snr_max_db],
            "seed": args.seed, "samples": len(dataset),
            "batch_size": args.batch_size, "device": str(device),
        },
    }
    (output_dir / "routing_temperature_sensitivity_audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )

    if args.k_results_csv:
        plot_path = write_joint_plot(args.k_results_csv, tau_path, output_dir)
        print(f"[saved] combined figure: {plot_path}", flush=True)
    else:
        print("[note] τ sweep finished; pass --k_results_csv to generate the combined K/τ figure.", flush=True)
    print(f"[saved] {tau_path}", flush=True)


if __name__ == "__main__":
    main()
