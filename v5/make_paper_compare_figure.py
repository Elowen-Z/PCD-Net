# -*- coding: utf-8 -*-
"""
Generate a publication-ready comparison figure with zoomed waveform regions.

Output:
- SVG: vector figure for papers
- PNG: high-DPI raster backup

Example:
  D:/app/anaconda/anaconda/envs/EarthquakeDetection/python.exe \
    v5/make_paper_compare_figure.py \
    --trace B087.PB_20110811072730_EV --snr 0
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
for p in [ROOT, THIS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from v5.evaluate_transfer_v5 import EvalDataset, snr_db, cc_fn
from v5.compare_models import CONFIG, build_v6, build_baseline
from v3.baeslines.restormer1d import Restormer1D
from v3.baeslines.deep_denoiser import DeepDenoiser
from v3.baeslines.dprnn import DPRNN
from v3.baeslines.traditional_denoise import wavelet_denoise, butterworth_bandpass


def make_single_row_csv(trace_name: str, event_csv: str, out_csv: str) -> str:
    df = pd.read_csv(event_csv, low_memory=False)
    hit = df[df["trace_name"].astype(str) == trace_name]
    if len(hit) == 0:
        raise ValueError(f"trace not found in csv: {trace_name}")
    hit.iloc[[0]].to_csv(out_csv, index=False)
    return out_csv


def to_3t(arr: np.ndarray) -> np.ndarray:
    """Convert model output to [3, T]."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"unexpected output shape: {a.shape}")
    if a.shape[0] == 3:
        return a.astype(np.float32)
    if a.shape[1] == 3:
        return a.T.astype(np.float32)
    raise ValueError(f"cannot convert output to [3,T], shape={a.shape}")


def run_model(model_name: str, noisy: np.ndarray, z_cond: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        if model_name == "PCD-Net":
            model = build_v6(CONFIG["ckpt_v6"], device).to(device).eval()
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            zc = torch.from_numpy(z_cond).unsqueeze(0).to(device)
            out = model(x, zc)
            pred = out[0] if isinstance(out, tuple) else out
            return to_3t(pred[0].cpu().numpy())

        if model_name == "Restormer1D":
            model = build_baseline(Restormer1D, CONFIG["ckpt_restormer1d"], device).to(device).eval()
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            pred = model(x)
            pred = pred[0] if isinstance(pred, tuple) else pred
            return to_3t(pred.detach().cpu().numpy())

        if model_name == "DPRNN":
            model = build_baseline(DPRNN, CONFIG["ckpt_dprnn"], device).to(device).eval()
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            pred = model(x)
            pred = pred[0] if isinstance(pred, tuple) else pred
            return to_3t(pred.detach().cpu().numpy())

        if model_name == "DeepDenoiser":
            model = build_baseline(DeepDenoiser, CONFIG["ckpt_deepdenoiser"], device).to(device).eval()
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            pred = model(x)
            pred = pred[0] if isinstance(pred, tuple) else pred
            return to_3t(pred.detach().cpu().numpy())

        if model_name == "Wavelet":
            return to_3t(wavelet_denoise(noisy, wavelet="db4", level=6, threshold_mode="soft", threshold_scale=1.0))

        if model_name == "Bandpass":
            return to_3t(butterworth_bandpass(noisy, fs=float(CONFIG.get("fs", 100)), order=6, adaptive=True))

        raise ValueError(f"unknown model: {model_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="B087.PB_20110811072730_EV")
    ap.add_argument("--snr", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--channel", choices=["E", "N", "Z"], default="Z")
    ap.add_argument("--zoom_width", type=int, default=900,
                    help="Number of samples in zoomed window")
    ap.add_argument("--out_svg", default=None)
    ap.add_argument("--out_png", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_cfg = CONFIG["ds_natural"]

    full_meta = ds_cfg["event_csv"].replace("chunk2_val.csv", "chunk2.csv")
    if not os.path.exists(full_meta):
        full_meta = ds_cfg["event_csv"]

    tmp_csv = os.path.join("v5", "_tmp_single_trace_paper.csv")
    make_single_row_csv(args.trace, full_meta, tmp_csv)

    ds = EvalDataset(
        event_h5_path=ds_cfg["event_h5"],
        event_csv_path=tmp_csv,
        noise_h5_path=ds_cfg["noise_h5"],
        noise_csv_path=ds_cfg["noise_csv"],
        signal_len=CONFIG.get("signal_len", 6000),
        cond_len=CONFIG.get("cond_len", 400),
        snr_db_range=(args.snr, args.snr),
        seed=args.seed,
    )

    sample = ds[0]
    clean = sample["clean"].numpy()
    noisy = sample["noisy"].numpy()
    z_cond = sample["z_cond"].numpy()
    p_onset = int(sample["p_onset"])

    ch_map = {"E": 0, "N": 1, "Z": 2}
    ch_idx = ch_map[args.channel]

    models = ["PCD-Net", "Restormer1D", "DPRNN", "DeepDenoiser", "Wavelet", "Bandpass"]
    preds = {}
    metrics = {}

    snr_in = snr_db(clean, noisy)

    for m in models:
        pred = run_model(m, noisy, z_cond, device)
        preds[m] = pred
        snr_out = snr_db(clean, pred)
        cc = cc_fn(clean.reshape(-1), pred.reshape(-1))
        metrics[m] = {
            "snr_out": snr_out,
            "gain": snr_out - snr_in,
            "cc": cc,
        }
        print(f"[METRIC] {m:12s} | ΔSNR={snr_out - snr_in:+.2f} dB | CC={cc:.3f}")

    T = clean.shape[-1]
    t = np.arange(T)
    half = max(50, args.zoom_width // 2)
    z0 = max(0, p_onset - half)
    z1 = min(T, p_onset + half)

    fig, axes = plt.subplots(len(models), 2, figsize=(14, 16), sharex="col")
    fig.patch.set_facecolor("white")

    color_denoised = {
        "PCD-Net": "#005f73",
        "Restormer1D": "#0a9396",
        "DPRNN": "#94d2bd",
        "DeepDenoiser": "#ee9b00",
        "Wavelet": "#ca6702",
        "Bandpass": "#bb3e03",
    }

    for i, m in enumerate(models):
        ax_l = axes[i, 0]
        ax_r = axes[i, 1]

        c = clean[ch_idx]
        n = noisy[ch_idx]
        d = preds[m][ch_idx]

        ymax = max(np.max(np.abs(c)), np.max(np.abs(n)), np.max(np.abs(d)), 1e-6)

        # Full view
        ax_l.plot(t, n, color="#9aa0a6", lw=0.7, alpha=0.65, label="Noisy" if i == 0 else None)
        ax_l.plot(t, c, color="#111111", lw=1.2, alpha=0.95, label="Clean" if i == 0 else None)
        ax_l.plot(t, d, color=color_denoised[m], lw=1.0, alpha=0.95, label="Denoised" if i == 0 else None)
        ax_l.axvspan(z0, z1, color="#ffd166", alpha=0.22)
        ax_l.set_ylim(-ymax, ymax)
        ax_l.grid(alpha=0.22, linestyle="--", linewidth=0.5)
        ax_l.text(
            0.01, 0.88,
            f"{m}   ΔSNR={metrics[m]['gain']:+.2f} dB   CC={metrics[m]['cc']:.3f}",
            transform=ax_l.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
        )

        # Zoom view
        tz = np.arange(z0, z1)
        ax_r.plot(tz, n[z0:z1], color="#9aa0a6", lw=0.8, alpha=0.65)
        ax_r.plot(tz, c[z0:z1], color="#111111", lw=1.35, alpha=0.95)
        ax_r.plot(tz, d[z0:z1], color=color_denoised[m], lw=1.1, alpha=0.98)
        ax_r.set_ylim(-ymax, ymax)
        ax_r.grid(alpha=0.22, linestyle="--", linewidth=0.5)

        if i == 0:
            ax_l.set_title(f"Full Waveform ({args.channel} channel)", fontsize=12, pad=8)
            ax_r.set_title(f"Zoomed Waveform (samples {z0}:{z1})", fontsize=12, pad=8)

        ax_l.set_ylabel("Amplitude", fontsize=9)

    axes[-1, 0].set_xlabel("Sample Index", fontsize=10)
    axes[-1, 1].set_xlabel("Sample Index", fontsize=10)

    # Global legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.995))

    fig.suptitle(
        f"Single-Trace Denoising Comparison ({args.trace})\n"
        f"Input SNR = {snr_in:.2f} dB, highlighted region is used for visual zoom",
        fontsize=13,
        y=0.999,
    )
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.985])

    out_dir = os.path.join("v5", "trace_figs")
    os.makedirs(out_dir, exist_ok=True)
    out_svg = args.out_svg or os.path.join(out_dir, f"paper_compare_{args.trace}_snr{args.snr:.0f}_{args.channel}.svg")
    out_png = args.out_png or os.path.join(out_dir, f"paper_compare_{args.trace}_snr{args.snr:.0f}_{args.channel}.png")

    fig.savefig(out_svg, format="svg")
    fig.savefig(out_png, format="png", dpi=600)
    plt.close(fig)

    print(f"[OK] SVG -> {out_svg}")
    print(f"[OK] PNG -> {out_png}")

    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)


if __name__ == "__main__":
    main()
