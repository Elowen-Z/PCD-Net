# -*- coding: utf-8 -*-
"""Waveform visualization for mining transfer fine-tuning strategies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v5.evaluate_transfer_v5 import cc_fn, rmse_fn, snr_db, st_mae_mean
from v7.model_v7 import NoiseAwareDenoiserV7
from v7.transfer_staged_v7 import TARGETS, make_dataset


TRANSFER_ROOT = Path("v7/transfer_comparisons/mining")
OUT_DIR = Path("v7/paper_experiments/mining_transfer_finetune_ablation")
SAMPLE_RATE = 100

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
CHANNELS = [("Z", 2), ("N", 1), ("E", 0)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_index", type=int, default=-1)
    parser.add_argument("--scan_samples", type=int, default=120)
    parser.add_argument("--pre_seconds", type=float, default=0.5)
    parser.add_argument("--post_seconds", type=float, default=5.5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--output_prefix", default="mining_transfer_finetune")
    parser.add_argument(
        "--view",
        choices=["waveform", "envelope"],
        default="waveform",
        help="Plot raw signed waveforms or smoothed absolute-amplitude envelopes.",
    )
    parser.add_argument("--no_title", action="store_true", help="Remove the figure-level title.")
    parser.add_argument("--time_unit", choices=["s", "ms"], default="s")
    parser.add_argument(
        "--metric_label",
        choices=["gain_cc", "st_mae", "all"],
        default="gain_cc",
        help="Metric text shown in each row.",
    )
    return parser.parse_args()


def checkpoint_for(strategy: str) -> Path:
    if strategy == "all_frozen":
        path = Path("v7/checkpoints_feedback_stead_seed0/best_model_v7.pth")
    else:
        path = TRANSFER_ROOT / f"freeze_{strategy}" / "best_transfer_v7.pth"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def val_csv_for(strategy: str = "pcd_adaptation") -> Path:
    path = TRANSFER_ROOT / f"freeze_{strategy}" / "mining_val.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_model(checkpoint_path: Path, device: torch.device) -> NoiseAwareDenoiserV7:
    model = NoiseAwareDenoiserV7(
        in_ch=3,
        z_dim=128,
        cond_len=400,
        num_prototypes=16,
        top_m=4,
        num_heads=4,
        n_refine=3,
        base_ch=32,
        vq_temperature=0.3,
        use_prototypes=True,
        use_sparse_selection=True,
        use_cross_attn=True,
        use_quality_head=True,
        use_residual_feedback=True,
        adaptive_inference=True,
        stop_threshold=0.95,
        min_refine_steps=1,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_dataset():
    val_csv = val_csv_for()
    return (
        make_dataset(TARGETS["mining"], str(val_csv), seed=0, evaluation=True),
        pd.read_csv(val_csv, low_memory=False),
    )


def crop(arrays: list[np.ndarray], item: dict, pre_seconds: float, post_seconds: float):
    p_onset = int(item.get("p_onset", arrays[0].shape[-1] // 2))
    start = max(0, p_onset - int(round(pre_seconds * SAMPLE_RATE)))
    stop = min(arrays[0].shape[-1], p_onset + int(round(post_seconds * SAMPLE_RATE)))
    return [a[:, start:stop] for a in arrays], start, stop, p_onset


@torch.inference_mode()
def predict(model: NoiseAwareDenoiserV7, item: dict, device: torch.device):
    x = item["x"].unsqueeze(0).to(device)
    z = item["z_cond"].unsqueeze(0).to(device)
    output = model(x, z, adaptive_stop=False)
    pred = output[0][0].detach().cpu().numpy().astype(np.float32)
    quality = float(output[1][0].detach().cpu())
    return pred, quality


def metrics(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> dict[str, float]:
    clean_f = clean.reshape(-1).astype(np.float64)
    noisy_f = noisy.reshape(-1).astype(np.float64)
    den_f = denoised.reshape(-1).astype(np.float64)
    snr_in = float(snr_db(clean_f, noisy_f))
    snr_out = float(snr_db(clean_f, den_f))
    return {
        "input_snr_db": snr_in,
        "output_snr_db": snr_out,
        "gain_db": snr_out - snr_in,
        "cc": float(cc_fn(clean_f, den_f)),
        "rmse": float(rmse_fn(clean_f, den_f)),
        "st_mae": float(st_mae_mean(clean_f, den_f, SAMPLE_RATE)),
    }


def choose_sample(dataset, model, device, scan_samples: int, pre_seconds: float, post_seconds: float):
    best = None
    limit = min(scan_samples, len(dataset))
    for index in range(limit):
        item = dataset[index]
        if float(item.get("has_target", 1.0)) < 0.5:
            continue
        clean = item["y_clean"].numpy().astype(np.float32)
        noisy = item["x"].numpy().astype(np.float32)
        pred, quality = predict(model, item, device)
        (clean_w, noisy_w, pred_w), start, stop, _ = crop(
            [clean, noisy, pred], item, pre_seconds, post_seconds
        )
        m = metrics(clean_w, noisy_w, pred_w)
        signal_rms = float(np.sqrt(np.mean(clean_w**2)))
        pred_rms = float(np.sqrt(np.mean(pred_w**2)))
        noisy_rms = float(np.sqrt(np.mean(noisy_w**2)))
        residual_rms = float(np.sqrt(np.mean((noisy_w - pred_w) ** 2)))
        density = float(np.sqrt(np.mean(np.diff(clean_w, axis=-1) ** 2)) / max(signal_rms, 1e-8))
        rms_ratio = pred_rms / max(signal_rms, 1e-8)
        if not (0.35 <= rms_ratio <= 1.35):
            continue
        if not (6.0 <= m["gain_db"] <= 18.0 and 0.45 <= m["cc"] <= 0.95):
            continue
        score = (
            1.4 * m["gain_db"]
            + 4.0 * m["cc"]
            + 1.2 * min(residual_rms / max(noisy_rms, 1e-8), 1.0)
            + 0.8 * min(density, 1.5)
            - 3.0 * abs(rms_ratio - 0.8)
            - 0.25 * abs(m["input_snr_db"])
        )
        if best is None or score > best["score"]:
            best = {
                "index": index,
                "item": item,
                "metrics": m,
                "quality": quality,
                "score": float(score),
                "window_start": start,
                "window_stop": stop,
            }
    if best is None:
        raise RuntimeError("No valid mining sample found.")
    return best


def plot_waveform_grid(
    trace_name: str,
    clean: np.ndarray,
    noisy: np.ndarray,
    predictions: dict[str, np.ndarray],
    rows: list[dict[str, float | str]],
    output_base: Path,
    view: str = "waveform",
    no_title: bool = False,
    time_unit: str = "s",
    metric_label: str = "gain_cc",
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    fig, axes = plt.subplots(
        len(STRATEGIES),
        len(CHANNELS),
        figsize=(12.6, 7.6),
        sharex=True,
        constrained_layout=True,
    )
    t = np.arange(noisy.shape[-1]) / SAMPLE_RATE
    if time_unit == "ms":
        t = t * 1000.0
    noisy_color = "#6cc36c"
    den_color = "#ff8c1a"

    def display(values: np.ndarray) -> np.ndarray:
        if view == "waveform":
            return values
        envelope = np.abs(values)
        kernel = np.ones(9, dtype=np.float32) / 9.0
        return np.stack([np.convolve(ch, kernel, mode="same") for ch in envelope]).astype(np.float32)

    for r, strategy in enumerate(STRATEGIES):
        denoised = predictions[strategy]
        row = next(item for item in rows if item["strategy"] == strategy)
        for c, (channel_name, channel_index) in enumerate(CHANNELS):
            ax = axes[r, c]
            n = display(noisy)[channel_index]
            d = display(denoised)[channel_index]
            if view == "waveform":
                ymax = max(float(np.max(np.abs(n))), float(np.max(np.abs(d))), 1e-6)
                ymin = -1.08 * ymax
                ymax = 1.08 * ymax
            else:
                ymax = max(float(np.max(n)), float(np.max(d)), 1e-6)
                ymin = -0.03 * ymax
                ymax = 1.10 * ymax
            ax.plot(t, n, color=noisy_color, lw=0.55, alpha=0.62, label="Noisy")
            ax.plot(t, d, color=den_color, lw=0.75, alpha=0.95, label="Denoised")
            ax.axhline(0, color="#888888", lw=0.35, alpha=0.35)
            ax.set_ylim(ymin, ymax)
            ax.grid(alpha=0.17, linestyle="--", linewidth=0.35)
            if r == 0:
                ax.set_title(f"Channel {channel_name}", fontweight="bold")
            if c == 0:
                ax.set_ylabel(LABELS[strategy], fontweight="bold")
            else:
                ax.set_yticklabels([])
            if r == len(STRATEGIES) - 1:
                ax.set_xlabel("Time [ms]" if time_unit == "ms" else "Time [s]")
            if c == 2:
                if metric_label == "st_mae":
                    metric_text = f"ST-MAE {row['st_mae']:.4f}"
                elif metric_label == "all":
                    metric_text = (
                        f"Gain {row['gain_db']:+.2f} dB | "
                        f"CC {row['cc']:.3f} | ST-MAE {row['st_mae']:.4f}"
                    )
                else:
                    metric_text = f"Gain {row['gain_db']:+.2f} dB | CC {row['cc']:.3f}"
                ax.text(
                    0.98,
                    0.08,
                    metric_text,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.5),
                )
            if r == 0 and c == 2:
                ax.legend(loc="upper right", frameon=True)

    if not no_title:
        fig.suptitle(
            f"Mining Transfer Fine-tuning {'Envelope' if view == 'envelope' else 'Waveform'} Visualization | {trace_name}",
            fontsize=14,
            fontweight="bold",
        )
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output_base.with_suffix(f".{ext}"), dpi=450 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.matmul.allow_tf32 = False

    dataset, frame = load_dataset()
    selector_model = build_model(checkpoint_for("pcd_adaptation"), device)
    if args.sample_index >= 0:
        selected = {"index": args.sample_index, "item": dataset[args.sample_index]}
    else:
        selected = choose_sample(
            dataset,
            selector_model,
            device,
            args.scan_samples,
            args.pre_seconds,
            args.post_seconds,
        )
    del selector_model

    index = int(selected["index"])
    item = selected["item"]
    trace_name = (
        str(frame.iloc[index]["trace_name"])
        if index < len(frame) and "trace_name" in frame.columns
        else f"mining_val_{index:04d}"
    )

    clean = item["y_clean"].numpy().astype(np.float32)
    noisy = item["x"].numpy().astype(np.float32)
    cropped_clean_noisy, start, stop, p_onset = crop(
        [clean, noisy], item, args.pre_seconds, args.post_seconds
    )
    clean_w, noisy_w = cropped_clean_noisy

    predictions: dict[str, np.ndarray] = {}
    rows: list[dict[str, float | str]] = []
    for strategy in STRATEGIES:
        model = build_model(checkpoint_for(strategy), device)
        pred, quality = predict(model, item, device)
        pred_w = crop([pred], item, args.pre_seconds, args.post_seconds)[0][0]
        predictions[strategy] = pred_w
        rows.append(
            {
                "strategy": strategy,
                "label": LABELS[strategy],
                "quality": quality,
                **metrics(clean_w, noisy_w, pred_w),
            }
        )
        del model

    output_base = out_dir / "mining_transfer_finetune_waveform_grid"
    output_base = out_dir / (
        f"{args.output_prefix}_envelope_grid"
        if args.view == "envelope"
        else f"{args.output_prefix}_waveform_grid"
    )
    plot_waveform_grid(
        trace_name,
        clean_w,
        noisy_w,
        predictions,
        rows,
        output_base,
        view=args.view,
        no_title=args.no_title,
        time_unit=args.time_unit,
        metric_label=args.metric_label,
    )

    table = pd.DataFrame(rows)
    table.insert(0, "trace_name", trace_name)
    table.insert(1, "sample_index", index)
    table.insert(2, "window_start", start)
    table.insert(3, "window_stop", stop)
    table.insert(4, "p_onset", p_onset)
    table.to_csv(out_dir / "mining_transfer_finetune_waveform_table.csv", index=False, encoding="utf-8-sig")
    table.round(4).to_csv(
        out_dir / "mining_transfer_finetune_waveform_table_rounded.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (out_dir / "mining_transfer_finetune_waveform_selection.json").write_text(
        json.dumps(
            {
                "trace_name": trace_name,
                "sample_index": index,
                "window_start": start,
                "window_stop": stop,
                "p_onset": p_onset,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] waveform -> {output_base}.png/.svg/.pdf")
    print(f"[OK] table -> {out_dir / 'mining_transfer_finetune_waveform_table_rounded.csv'}")


if __name__ == "__main__":
    main()
