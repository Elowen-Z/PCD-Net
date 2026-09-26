"""Unified input-SNR controlled comparison for PCD-Net V9.

Benchmarks the following methods on one shared STEAD hold-out split with a
fixed input-SNR interval (default ``-15 ... 10`` dB, sampled uniformly in dB):

    Bandpass, Wavelet, DeepDenoiser, DPRNN, Restormer1D, ProtoCondTCN, PCD-Net

Design notes
------------
* Only one :class:`EvalDataset` instance is built and reused for every method.
  Its per-index RNG (``default_rng(seed + idx)``) makes the noisy waveform, the
  sampled input SNR and the P-arrival identical for all methods, so the
  comparison is paired sample by sample.
* ``--context_source pre_event`` (default) builds the conditioning window from
  the P-arrival-preceding background of the same trace.  This is the protocol
  PCD-Net V9 and ProtoCondTCN are trained with.  ``paired_noise`` reproduces the
  legacy ``v5/compare_models.py`` behaviour (conditioning window taken from the
  chunk1 noise file).
* Methods that do not consume a conditioning window (Bandpass, Wavelet,
  DeepDenoiser, DPRNN, Restormer1D) simply ignore it.
* The output CSV schema matches ``v5/compare_models.py`` so the existing
  aggregation and plotting scripts keep working:

    per_model_<Name>.csv, global_compare.csv, per_group_compare.csv,
    paper_model_comparison_table.csv/.tex, compare_bar.png, summary.txt

  plus two protocol records:

    run_config.json      exact settings, paths and checkpoint digests
    test_set_audit.json  per-method N / mean SNR_in / SNR-group counts

Usage
-----
python -m v9.compare_v9 --max_samples 2000 \
    --ckpt_v9 v9/experiments/snr_m15_p10_seed0/best_model_v9.pth \
    --ckpt_deepdenoiser ... --ckpt_dprnn ... --ckpt_restormer1d ... \
    --ckpt_proto_cond_tcn ... \
    --save_dir v9/paper_experiments/model_comparison_stead_snr_m15_p10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from v5.evaluate_transfer_v5 import (
    EPS,
    SNR_BINS,
    SNR_LABELS,
    EvalDataset,
    assign_snr_group,
    cc_fn,
    prd_fn,
    rmse_fn,
    snr_db,
    st_mae_mean,
    stalta_pick,
)
from v9.train_v9 import DEFAULT_CONFIG as V9_DEFAULT_CONFIG
from v9.train_v9 import build_model as build_v9_model

try:  # PyWavelets is only needed for the two classical baselines.
    from v3.baeslines.traditional_denoise import (
        butterworth_bandpass,
        wavelet_denoise,
    )

    TRADITIONAL_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment guard
    butterworth_bandpass = None
    wavelet_denoise = None
    TRADITIONAL_IMPORT_ERROR = str(exc)


# ============================================================
#  协议常量
# ============================================================
METHOD_ORDER = [
    "PCD-Net",
    "ProtoCondTCN",
    "GDiffuSE",
    "Restormer1D",
    "CDiffuSE",
    "DPRNN",
    "DeepDenoiser",
    "DnCNN",
    "Wavelet",
    "Bandpass",
]

DEFAULT_EVENT_H5 = "E:/p_wave/data/chunk2.hdf5"
DEFAULT_EVENT_CSV = "v5/v7_seed0_val.csv"
DEFAULT_NOISE_H5 = "E:/p_wave/data/chunk1.hdf5"
DEFAULT_NOISE_CSV = "E:/p_wave/data/chunk1.csv"

DEFAULT_CKPT_V9 = "v9/experiments/snr_m15_p10_seed0/best_model_v9.pth"
DEFAULT_CKPT_DEEPDENOISER = (
    "v3/baeslines/checkpoints_snr_m15_p10/deep_denoiser/best_model.pth"
)
DEFAULT_CKPT_DPRNN = (
    "v3/baeslines/checkpoints_snr_m15_p10/dprnn/best_model.pth"
)
DEFAULT_CKPT_RESTORMER1D = (
    "v3/baeslines/checkpoints_snr_m15_p10/restormer1d/best_model.pth"
)
DEFAULT_CKPT_PROTO_COND_TCN = (
    "v7/retrain_snr_m15_p10/ProtoCondTCN_seed0/best_proto_cond_tcn.pth"
)
DEFAULT_CKPT_DNCNN = (
    "v3/baeslines/checkpoints_snr_m15_p10/dncnn/best_model.pth"
)
DEFAULT_CKPT_GDiffuSE = (
    "v7/retrain_stead_v7_20260702_ablation_scale/GDiffuSEStyle_seed0/"
    "best_gdiffuse_style.pth"
)
DEFAULT_CKPT_CDiffuSE = (
    "v7/retrain_stead_v7_20260702_ablation_scale/CDiffuSEStyle_seed0/"
    "best_cdiffuse_style.pth"
)

MODEL_INFO = {
    "Bandpass": ("classical filtering baseline with an adaptive pass band", False),
    "Wavelet": ("classical db4 soft-thresholding baseline", False),
    "DnCNN": ("residual-learning convolutional denoiser (classic CNN baseline)", False),
    "DeepDenoiser": ("convolutional U-Net waveform denoiser", False),
    "DPRNN": ("dual-path recurrent network for long sequences", False),
    "CDiffuSE": ("conditional diffusion baseline, uses the background condition", True),
    "Restormer1D": ("transformer-style one-dimensional restoration baseline", False),
    "GDiffuSE": ("guided diffusion baseline, uses the background condition", True),
    "ProtoCondTCN": ("prototype-conditioned temporal convolutional baseline", True),
    "PCD-Net": ("proposed prototype-conditioned cross-attention network", True),
}

# Inference settings for the diffusion-style baselines.  The defaults below are
# the operating points selected on the validation split (see
# v7/paper_experiments/diffusion_param_scan): the plain ancestral loop collapses
# on this task, and the residual blend dominates the achievable Delta-SNR.
DEFAULT_GD_MODE = "posterior_residual"
DEFAULT_GD_SAMPLE_STEPS = 50
DEFAULT_GD_GUIDANCE = 0.15
DEFAULT_GD_BLEND = 0.30
DEFAULT_CD_SAMPLE_STEPS = 50
DEFAULT_CD_BLEND = 0.20
DEFAULT_DIFFUSION_STEPS = 50


# ============================================================
#  统一推理接口
# ============================================================
class Denoiser:
    """Every method is exposed as ``denoise(noisy, z_cond) -> (pred, quality)``."""

    name = "Denoiser"
    uses_condition = False

    def denoise(self, noisy: torch.Tensor, z_cond: torch.Tensor):
        raise NotImplementedError


class TorchDenoiser(Denoiser):
    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        device: torch.device,
        use_zcond: bool,
        returns_quality: bool = False,
        accepts_adaptive_stop: bool = False,
    ):
        self.name = name
        self.model = model.to(device).eval()
        self.device = device
        self.use_zcond = bool(use_zcond)
        self.returns_quality = bool(returns_quality)
        self.accepts_adaptive_stop = bool(accepts_adaptive_stop)
        self.uses_condition = self.use_zcond

    @torch.no_grad()
    def denoise(self, noisy, z_cond):
        if self.use_zcond:
            if self.accepts_adaptive_stop:
                out = self.model(noisy, z_cond, adaptive_stop=False)
            else:
                out = self.model(noisy, z_cond)
        else:
            out = self.model(noisy)
        pred = out[0] if isinstance(out, tuple) else out
        quality = None
        if self.returns_quality and isinstance(out, tuple) and len(out) > 1:
            quality = out[1]
            if torch.is_tensor(quality):
                quality = quality.detach().float().reshape(quality.size(0), -1)[:, 0]
        return pred, quality


class TraditionalDenoiser(Denoiser):
    """Numpy ``[3, T] -> [3, T]`` callable, applied per sample."""

    def __init__(self, name: str, func):
        self.name = name
        self.func = func

    def denoise(self, noisy, z_cond):
        batch = noisy.detach().cpu().numpy()
        out = np.stack([self.func(batch[i]) for i in range(batch.shape[0])], axis=0)
        return torch.from_numpy(out).to(noisy.device).float(), None


class FunctionalDenoiser(Denoiser):
    """Wrap an arbitrary ``fn(noisy, z_cond) -> pred`` callable."""

    def __init__(self, name: str, fn, uses_condition: bool = False):
        self.name = name
        self.fn = fn
        self.uses_condition = bool(uses_condition)

    @torch.no_grad()
    def denoise(self, noisy, z_cond):
        return self.fn(noisy, z_cond), None


# ============================================================
#  模型构建
# ============================================================
def _sha1(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _load_state_dict(path: str, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("config", {})
    return checkpoint, {}


def build_v9_denoiser(ckpt_path: str, device: torch.device) -> Denoiser:
    state, config = _load_state_dict(ckpt_path, device)
    merged = {**V9_DEFAULT_CONFIG, **config}
    model = build_v9_model(merged)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(
        f"  [cfg] PCD-Net ckpt snr_range={tuple(merged.get('snr_range', ()))} "
        f"snr_sampling={merged.get('snr_sampling')} "
        f"context_source={merged.get('context_source')} "
        f"n_refine={merged.get('n_refine')} "
        f"train_samples={merged.get('train_samples')} "
        f"epochs={merged.get('epochs')}",
        flush=True,
    )
    return TorchDenoiser(
        "PCD-Net", model, device, use_zcond=True, returns_quality=True,
        accepts_adaptive_stop=True,
    )


def build_proto_cond_tcn_denoiser(ckpt_path: str, device: torch.device) -> Denoiser:
    from v7.proto_cond_tcn_baseline import ProtoCondTCNDenoiser

    state, config = _load_state_dict(ckpt_path, device)
    model = ProtoCondTCNDenoiser(
        hidden=int(config.get("hidden", 64)),
        cond_dim=int(config.get("cond_dim", 128)),
        num_prototypes=int(config.get("num_prototypes", 16)),
        top_m=int(config.get("top_m", 4)),
        layers=int(config.get("layers", 8)),
        residual_scale=float(config.get("residual_scale", 1.0)),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return TorchDenoiser(
        "ProtoCondTCN", model, device, use_zcond=True, returns_quality=False
    )


def build_baseline_denoiser(
    name: str, ckpt_path: str, device: torch.device
) -> Denoiser:
    from v3.baeslines.deep_denoiser import DeepDenoiser
    from v3.baeslines.dncnn import DnCNN
    from v3.baeslines.dprnn import DPRNN
    from v3.baeslines.restormer1d import Restormer1D

    registry = {
        "DeepDenoiser": DeepDenoiser,
        "DPRNN": DPRNN,
        "Restormer1D": Restormer1D,
        "DnCNN": DnCNN,
    }
    model = registry[name](in_ch=3)
    state, _ = _load_state_dict(ckpt_path, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return TorchDenoiser(name, model, device, use_zcond=False)


def build_diffusion_denoiser(
    name: str, ckpt_path: str, device: torch.device, settings: dict
) -> Denoiser:
    """Wrap the diffusion-style baselines behind the shared denoise contract.

    The inference recipe mirrors the scripts' own ``evaluate`` path, with the
    operating point selected on the validation split.
    """
    from argparse import Namespace

    arch = Namespace(hidden=32, time_dim=128, layers=8)
    if name == "GDiffuSE":
        from v7.gdiffuse_style_baseline import (
            build_schedule,
            enhance_batch,
            enhance_batch_residual,
            load_model,
        )

        model = load_model(Path(ckpt_path), arch, device)
        schedule = build_schedule(settings["diffusion_steps"], device)

        def run(noisy, z_cond):
            if settings["gd_mode"] == "ancestral":
                return enhance_batch(
                    model,
                    noisy,
                    z_cond,
                    schedule,
                    settings["diffusion_steps"],
                    settings["gd_sample_steps"],
                    settings["gd_guidance"],
                )
            return enhance_batch_residual(
                model,
                noisy,
                z_cond,
                schedule,
                settings["diffusion_steps"],
                settings["gd_sample_steps"],
                settings["gd_guidance"],
                settings["gd_blend"],
            )

        return FunctionalDenoiser(name, run, uses_condition=True)

    if name == "CDiffuSE":
        from v7.cdiffuse_style_baseline import (
            build_schedule,
            enhance_batch,
            load_model,
        )

        model = load_model(Path(ckpt_path), arch, device)
        schedule = build_schedule(settings["diffusion_steps"], device)

        def run(noisy, z_cond):
            return enhance_batch(
                model,
                noisy,
                z_cond,
                schedule,
                settings["diffusion_steps"],
                settings["cd_sample_steps"],
                settings["cd_blend"],
            )

        return FunctionalDenoiser(name, run, uses_condition=True)

    raise ValueError(f"unsupported diffusion baseline: {name}")


def build_traditional_denoiser(name: str, fs: float) -> Denoiser:
    if TRADITIONAL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyWavelets is required for the classical baselines "
            f"(import error: {TRADITIONAL_IMPORT_ERROR}). Install it with "
            "`python -m pip install PyWavelets`."
        )
    if name == "Bandpass":
        return TraditionalDenoiser(
            "Bandpass",
            lambda wave: butterworth_bandpass(
                wave, fs=fs, adaptive=True, order=6
            ),
        )
    return TraditionalDenoiser(
        "Wavelet",
        lambda wave: wavelet_denoise(
            wave, wavelet="db4", threshold_mode="soft", threshold_scale=1.0
        ),
    )


# ============================================================
#  SNR 分组
# ============================================================
def build_group_spec(spec: str | None):
    """Return ``(labels, edges, assign)`` for the input-SNR grouping.

    Without ``spec`` the paper bins from ``v5.evaluate_transfer_v5`` are used
    (``<-5 / -5~0 / 0~5 / 5~10 / ...``).  ``--snr_groups -15,-10,-5,0,5,10``
    gives a finer low-SNR resolution.
    """
    if not spec:
        return list(SNR_LABELS), list(SNR_BINS), assign_snr_group

    edges = [float(value) for value in spec.split(",")]
    if len(edges) < 2 or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("--snr_groups must be an increasing list of edges")
    labels = [f"{low:g}~{high:g}" for low, high in zip(edges, edges[1:])]

    def assign(value: float) -> str:
        if value < edges[0]:
            return labels[0]
        if value >= edges[-1]:
            return labels[-1]
        for index in range(len(edges) - 1):
            if edges[index] <= value < edges[index + 1]:
                return labels[index]
        return labels[-1]

    return labels, edges, assign


# ============================================================
#  评估
# ============================================================
def evaluate_method(
    denoiser: Denoiser,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    assign_group,
    expected_snr_in: list[float] | None = None,
):
    """Run one method over the shared loader and return per-sample records."""
    records: list[dict] = []
    pick_suc_noisy: list[bool] = []
    pick_suc_denoise: list[bool] = []
    snr_seen: list[float] = []
    skipped = 0
    started = time.time()

    pick_kwargs = dict(
        fs=cfg["fs"],
        sta_len=cfg["sta_len"],
        lta_len=cfg["lta_len"],
        threshold=cfg["stalta_thr"],
    )

    for batch in loader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        z_cond = batch["z_cond"].to(device)
        nominal_snr = batch["snr_db"]
        p_onset = batch["p_onset"]

        if not all(torch.isfinite(t).all() for t in (noisy, clean, z_cond)):
            skipped += int(noisy.size(0))
            continue

        try:
            pred, quality = denoiser.denoise(noisy, z_cond)
        except Exception as exc:  # pragma: no cover - defensive
            skipped += int(noisy.size(0))
            print(f"    [WARN] {denoiser.name} forward failed: {exc}", flush=True)
            continue
        if not torch.isfinite(pred).all():
            skipped += int(noisy.size(0))
            continue

        quality_np = None
        if quality is not None:
            quality_np = quality.detach().float().cpu().numpy().reshape(-1)

        for index in range(noisy.size(0)):
            x_np = noisy[index].detach().cpu().numpy()
            y_np = clean[index].detach().cpu().numpy()
            p_np = pred[index].detach().cpu().numpy()
            p_true = int(p_onset[index].item())
            snr_nominal = float(nominal_snr[index].item())

            y_all = y_np.astype(np.float64).reshape(-1)
            x_all = x_np.astype(np.float64).reshape(-1)
            p_all = p_np.astype(np.float64).reshape(-1)

            snr_out = snr_db(y_all, p_all)
            snr_realized = snr_db(y_all, x_all)
            relative_mse = float(
                np.sum((p_all - y_all) ** 2) / (np.sum(y_all ** 2) + EPS)
            )

            peak = float(np.abs(p_np).max())
            p_norm = p_np / peak if peak > 1e-10 else p_np
            picks = {}
            for tag, wave, store in (
                ("noisy", x_np, pick_suc_noisy),
                ("denoise", p_norm, pick_suc_denoise),
            ):
                picked = stalta_pick(wave, **pick_kwargs)
                success = picked >= 0 and abs(picked - p_true) <= cfg["pick_tol"]
                store.append(success)
                picks[tag] = bool(success)

            records.append(
                {
                    "snr_in": snr_nominal,
                    "snr_in_realized": snr_realized,
                    "snr_out": snr_out,
                    "delta_snr": snr_out - snr_nominal,
                    "delta_snr_realized": snr_out - snr_realized,
                    "cc": cc_fn(y_all, p_all),
                    "rmse": rmse_fn(y_all, p_all),
                    "prd": prd_fn(y_all, p_all),
                    "st_mae_noisy": st_mae_mean(y_all, x_all, cfg["fs"]),
                    "st_mae_denoised": st_mae_mean(y_all, p_all, cfg["fs"]),
                    "model_quality": (
                        float(quality_np[index]) if quality_np is not None else float("nan")
                    ),
                    "fidelity_score": float(np.exp(-relative_mse)),
                    "pick_suc_noisy": picks["noisy"],
                    "pick_suc_denoise": picks["denoise"],
                    "snr_group": assign_group(snr_nominal),
                }
            )
            snr_seen.append(snr_nominal)

        if len(records) % 400 == 0 and records:
            print(f"    [{denoiser.name}] n={len(records)}", flush=True)

    elapsed = time.time() - started

    if expected_snr_in is not None:
        if len(expected_snr_in) != len(snr_seen) or not np.allclose(
            expected_snr_in, snr_seen, atol=1e-6
        ):
            raise RuntimeError(
                f"{denoiser.name} did not see the reference test set "
                "(input SNR mismatch); the paired comparison would be invalid."
            )

    print(
        f"    {denoiser.name:>13s} | n={len(records):4d} | skip={skipped} | "
        f"{elapsed:.1f}s",
        flush=True,
    )
    return records, snr_seen, elapsed


# ============================================================
#  汇总
# ============================================================
def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def summarize(records, model_name, dt, labels):
    df = pd.DataFrame(records)
    if df.empty:
        return None, None
    row = {
        "Model": model_name,
        "N": len(df),
        "ΔSNR(dB)": df["delta_snr"].mean(),
        "SNR_in(dB)": df["snr_in"].mean(),
        "SNR_out(dB)": df["snr_out"].mean(),
        "CC": df["cc"].mean(),
        "RMSE": df["rmse"].mean(),
        "PRD": df["prd"].mean(),
        "ST-MAE(noisy)": df["st_mae_noisy"].mean(),
        "ST-MAE(denoised)": df["st_mae_denoised"].mean(),
        "Model_quality": df["model_quality"].mean(),
        "Fidelity_score": df["fidelity_score"].mean(),
        "Pick_succ_noisy": _mean(df["pick_suc_noisy"]),
        "Pick_succ_denoise": _mean(df["pick_suc_denoise"]),
        "time_sec": round(dt, 1),
    }
    quality_rows = df.dropna(subset=["model_quality"])
    row["Quality_fidelity_corr"] = float("nan")
    if (
        len(quality_rows) > 1
        and quality_rows["model_quality"].std() > 0
        and quality_rows["fidelity_score"].std() > 0
    ):
        row["Quality_fidelity_corr"] = float(
            quality_rows["model_quality"].corr(quality_rows["fidelity_score"])
        )

    group_rows = []
    for label in labels:
        sub = df[df["snr_group"] == label]
        if sub.empty:
            continue
        group_rows.append(
            {
                "Model": model_name,
                "SNR_Group": label,
                "N": len(sub),
                "SNR_in": sub["snr_in"].mean(),
                "SNR_out": sub["snr_out"].mean(),
                "ΔSNR": sub["delta_snr"].mean(),
                "CC": sub["cc"].mean(),
                "RMSE": sub["rmse"].mean(),
                "PRD": sub["prd"].mean(),
                "ST-MAE": sub["st_mae_denoised"].mean(),
                "Model_quality": sub["model_quality"].mean(),
                "Fidelity_score": sub["fidelity_score"].mean(),
            }
        )
    return row, group_rows


def _bootstrap_ci(values, seed=42, repetitions=2000):
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


METRIC_SPECS = [
    ("delta_snr", "SNR improvement (dB)", True),
    ("cc", "correlation coefficient", True),
    ("rmse", "RMSE", False),
    ("prd", "PRD", False),
    ("st_mae_denoised", "ST-MAE", False),
    ("fidelity_score", "fidelity score", True),
]


def write_publication_table(out_dir: Path, method_order):
    """Bootstrap CIs + LaTeX table, same layout as v5/compare_models.py."""
    rows = []
    for model in method_order:
        path = out_dir / f"per_model_{model}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        row = {"Model": model, "N": len(frame)}
        for metric, _, _ in METRIC_SPECS:
            mean, low, high = _bootstrap_ci(frame[metric].to_numpy())
            row[metric] = mean
            row[metric + "_ci_low"] = low
            row[metric + "_ci_high"] = high
        quality = frame["model_quality"].dropna()
        row["model_quality"] = (
            float(quality.mean()) if not quality.empty else float("nan")
        )
        pair = frame[["model_quality", "fidelity_score"]].dropna()
        row["quality_fidelity_corr"] = (
            float(pair.corr().iloc[0, 1]) if len(pair) > 1 else float("nan")
        )
        rows.append(row)
    if not rows:
        return None

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "paper_model_comparison_table.csv", index=False,
                 encoding="utf-8-sig")

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & $\Delta$SNR $\uparrow$ & CC $\uparrow$ & RMSE $\downarrow$ & "
        r"PRD $\downarrow$ & ST-MAE $\downarrow$ & Fidelity $\uparrow$ \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['Model']} & {row['delta_snr']:.3f} & {row['cc']:.4f} & "
            f"{row['rmse']:.4f} & {row['prd']:.4f} & "
            f"{row['st_mae_denoised']:.4f} & {row['fidelity_score']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "paper_model_comparison_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return table


def plot_bar(global_df: pd.DataFrame, output_path: Path, proposed: str) -> None:
    if global_df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Model comparison on the shared hold-out split (proposed: {proposed})",
        fontsize=13,
        fontweight="bold",
    )
    panels = [
        ("ΔSNR(dB)", "ΔSNR (dB, higher better)", True),
        ("CC", "Correlation (higher better)", True),
        ("RMSE", "RMSE (lower better)", False),
        ("Fidelity_score", "Fidelity score (higher better)", True),
        ("ST-MAE(denoised)", "ST-MAE (lower better)", False),
        ("Pick_succ_denoise", "P pick success rate", True),
    ]
    for ax, (column, title, higher_better) in zip(axes.flat, panels):
        order = global_df.sort_values(column, ascending=not higher_better)
        colors = [
            "#E34A5F" if model == proposed else "#7f7f7f" for model in order["Model"]
        ]
        bars = ax.barh(order["Model"], order[column], color=colors, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        for bar, value in zip(bars, order[column]):
            if not np.isfinite(value):
                continue
            text = f"  {value:+.3f}" if column == "ΔSNR(dB)" else f"  {value:.3f}"
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                text,
                va="center",
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {output_path}", flush=True)


def write_summary(
    out_dir: Path,
    global_df: pd.DataFrame,
    group_df: pd.DataFrame,
    cfg: dict,
    group_labels: list[str],
) -> None:
    lines = []
    lines.append("PCD-Net V9 unified comparison — shared STEAD hold-out split")
    lines.append(f"event_csv        : {cfg['event_csv']}")
    lines.append(
        f"input SNR        : [{cfg['snr_min_db']}, {cfg['snr_max_db']}] dB "
        f"(uniform in dB)"
    )
    lines.append(f"context source   : {cfg['context_source']}")
    lines.append(f"condition length : {cfg['cond_len']} samples")
    lines.append(f"noise boost      : {cfg['noise_boost']}")
    lines.append(f"samples per method: {cfg['max_samples']}")
    lines.append("-" * 118)
    header = (
        f"{'Model':>14} | {'ΔSNR':>8} | {'CC':>6} | {'RMSE':>7} | "
        f"{'ST-MAE':>8} | {'Fidelity':>8} | {'Quality':>7} | "
        f"{'Q-F corr':>8} | {'Pick':>5} | {'sec':>6}"
    )
    lines.append(header)
    lines.append("-" * 118)
    for _, row in global_df.iterrows():
        quality = (
            f"{row['Model_quality']:.3f}"
            if np.isfinite(row["Model_quality"])
            else "N/A"
        )
        quality_corr = (
            f"{row['Quality_fidelity_corr']:.3f}"
            if np.isfinite(row["Quality_fidelity_corr"])
            else "N/A"
        )
        lines.append(
            f"{row['Model']:>14} | {row['ΔSNR(dB)']:>+8.3f} | "
            f"{row['CC']:>6.4f} | {row['RMSE']:>7.4f} | "
            f"{row['ST-MAE(denoised)']:>8.4f} | {row['Fidelity_score']:>8.4f} | "
            f"{quality:>7} | {quality_corr:>8} | "
            f"{row['Pick_succ_denoise']:>5.3f} | {row['time_sec']:>6.1f}"
        )

    if not group_df.empty:
        lines.append("")
        lines.append("ΔSNR by input-SNR group (dB)")
        lines.append("-" * 118)
        pivot = group_df.pivot(index="Model", columns="SNR_Group", values="ΔSNR")
        pivot = pivot.reindex(columns=[l for l in group_labels if l in pivot.columns])
        counts = group_df.pivot(index="Model", columns="SNR_Group", values="N")
        lines.append("N per group:")
        lines.append(counts.to_string())
        lines.append("")
        lines.append(pivot.round(3).to_string())

    text = "\n".join(lines)
    print("\n" + text, flush=True)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8-sig")


# ============================================================
#  CLI
# ============================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified input-SNR controlled comparison for PCD-Net V9"
    )
    parser.add_argument("--event_h5", default=DEFAULT_EVENT_H5)
    parser.add_argument(
        "--event_csv",
        default=DEFAULT_EVENT_CSV,
        help="shared hold-out split; v5/v7_seed0_val.csv is the seed-0 STEAD split",
    )
    parser.add_argument("--noise_h5", default=DEFAULT_NOISE_H5)
    parser.add_argument("--noise_csv", default=DEFAULT_NOISE_CSV)
    parser.add_argument(
        "--context_source",
        choices=["pre_event", "paired_noise"],
        default="pre_event",
        help="pre_event matches the PCD-Net V9 training protocol",
    )
    parser.add_argument("--snr_min_db", type=float, default=-15.0)
    parser.add_argument("--snr_max_db", type=float, default=10.0)
    parser.add_argument(
        "--snr_groups",
        default=None,
        help="comma separated SNR bin edges, e.g. -15,-10,-5,0,5,10",
    )
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--signal_len", type=int, default=6000)
    parser.add_argument("--cond_len", type=int, default=400)
    parser.add_argument("--noise_boost", type=float, default=1.0)
    parser.add_argument("--st_window", type=int, default=100)
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--sta_len", type=float, default=0.5)
    parser.add_argument("--lta_len", type=float, default=10.0)
    parser.add_argument("--stalta_thr", type=float, default=2.0)
    parser.add_argument("--pick_tol", type=int, default=50)
    parser.add_argument("--methods", default=",".join(METHOD_ORDER))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cuda_safe", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse an existing per_model_<Name>.csv with the expected row count",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="skip methods whose checkpoint is missing instead of aborting",
    )
    parser.add_argument("--ckpt_v9", default=DEFAULT_CKPT_V9)
    parser.add_argument("--ckpt_deepdenoiser", default=DEFAULT_CKPT_DEEPDENOISER)
    parser.add_argument("--ckpt_dprnn", default=DEFAULT_CKPT_DPRNN)
    parser.add_argument("--ckpt_restormer1d", default=DEFAULT_CKPT_RESTORMER1D)
    parser.add_argument("--ckpt_proto_cond_tcn", default=DEFAULT_CKPT_PROTO_COND_TCN)
    parser.add_argument("--ckpt_dncnn", default=DEFAULT_CKPT_DNCNN)
    parser.add_argument("--ckpt_gdiffuse", default=DEFAULT_CKPT_GDiffuSE)
    parser.add_argument("--ckpt_cdiffuse", default=DEFAULT_CKPT_CDiffuSE)
    # Diffusion inference operating points (validation-selected).
    parser.add_argument("--diffusion_steps", type=int, default=DEFAULT_DIFFUSION_STEPS)
    parser.add_argument(
        "--gd_mode",
        choices=["posterior_residual", "ancestral"],
        default=DEFAULT_GD_MODE,
    )
    parser.add_argument("--gd_sample_steps", type=int, default=DEFAULT_GD_SAMPLE_STEPS)
    parser.add_argument("--gd_guidance", type=float, default=DEFAULT_GD_GUIDANCE)
    parser.add_argument("--gd_blend", type=float, default=DEFAULT_GD_BLEND)
    parser.add_argument("--cd_sample_steps", type=int, default=DEFAULT_CD_SAMPLE_STEPS)
    parser.add_argument("--cd_blend", type=float, default=DEFAULT_CD_BLEND)
    parser.add_argument("--save_dir", required=True)
    return parser


# ============================================================
#  main
# ============================================================
def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.cuda_safe and device.type == "cuda":
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.snr_min_db >= args.snr_max_db:
        raise ValueError("--snr_min_db must be smaller than --snr_max_db")
    group_labels, group_edges, assign_group = build_group_spec(args.snr_groups)

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [item for item in methods if item not in MODEL_INFO]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; candidates {list(MODEL_INFO)}")

    ckpt_by_method = {
        "PCD-Net": args.ckpt_v9,
        "ProtoCondTCN": args.ckpt_proto_cond_tcn,
        "DeepDenoiser": args.ckpt_deepdenoiser,
        "DPRNN": args.ckpt_dprnn,
        "Restormer1D": args.ckpt_restormer1d,
        "DnCNN": args.ckpt_dncnn,
        "GDiffuSE": args.ckpt_gdiffuse,
        "CDiffuSE": args.ckpt_cdiffuse,
    }
    diffusion_settings = {
        "diffusion_steps": args.diffusion_steps,
        "gd_mode": args.gd_mode,
        "gd_sample_steps": args.gd_sample_steps,
        "gd_guidance": args.gd_guidance,
        "gd_blend": args.gd_blend,
        "cd_sample_steps": args.cd_sample_steps,
        "cd_blend": args.cd_blend,
    }

    print("=" * 80)
    print("  PCD-Net V9 unified comparison")
    print("=" * 80)
    print(f"  device         : {device}")
    print(f"  hold-out csv   : {args.event_csv}")
    print(f"  input SNR      : [{args.snr_min_db}, {args.snr_max_db}] dB, uniform in dB")
    print(f"  context source : {args.context_source}")
    print(f"  max samples    : {args.max_samples}")
    print(f"  SNR groups     : {group_labels}")
    print(f"  methods        : {methods}")
    print(f"  output         : {out_dir}")

    for key in ("event_h5", "event_csv", "noise_h5", "noise_csv"):
        value = getattr(args, key)
        if not os.path.exists(value):
            raise FileNotFoundError(f"--{key} not found: {value}")

    missing = [
        (name, ckpt_by_method[name])
        for name in methods
        if name in ckpt_by_method and not os.path.exists(ckpt_by_method[name])
    ]
    if missing:
        for name, path in missing:
            print(f"  [MISSING] {name}: {path}")
        if not args.allow_missing:
            raise FileNotFoundError(
                "checkpoints are missing; pass --allow_missing to skip them"
            )

    cfg = {
        "event_h5": args.event_h5,
        "event_csv": args.event_csv,
        "noise_h5": args.noise_h5,
        "noise_csv": args.noise_csv,
        "context_source": args.context_source,
        "snr_min_db": args.snr_min_db,
        "snr_max_db": args.snr_max_db,
        "max_samples": args.max_samples,
        "signal_len": args.signal_len,
        "cond_len": args.cond_len,
        "noise_boost": args.noise_boost,
        "st_window": args.st_window,
        "fs": args.fs,
        "sta_len": args.sta_len,
        "lta_len": args.lta_len,
        "stalta_thr": args.stalta_thr,
        "pick_tol": args.pick_tol,
        "seed": args.seed,
        **diffusion_settings,
    }

    # One dataset instance for every method: identical noisy waveform, identical
    # sampled input SNR and identical conditioning window at every index.
    dataset = EvalDataset(
        event_h5_path=args.event_h5,
        event_csv_path=args.event_csv,
        noise_h5_path=args.noise_h5,
        noise_csv_path=args.noise_csv,
        signal_len=args.signal_len,
        cond_len=args.cond_len,
        snr_db_range=(args.snr_min_db, args.snr_max_db),
        noise_boost=args.noise_boost,
        max_samples=args.max_samples,
        seed=args.seed,
        context_source=args.context_source,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    expected_n = len(dataset)
    print(f"  test samples   : {expected_n} (paired across methods)\n")

    global_rows, group_rows = [], []
    reference_snr: list[float] | None = None
    evaluated_order: list[str] = []
    audit = {
        "protocol": cfg,
        "snr_groups": {"labels": group_labels, "edges": group_edges},
        "n_test_samples": expected_n,
        "methods": {},
    }

    for name in methods:
        per_model_path = out_dir / f"per_model_{name}.csv"
        records = None
        snr_seen: list[float] | None = None
        elapsed = float("nan")

        if args.resume and per_model_path.exists():
            frame = pd.read_csv(per_model_path)
            if len(frame) == expected_n:
                records = frame.to_dict("records")
                snr_seen = frame["snr_in"].tolist()
                print(f"  [resume] {name} <- {per_model_path}", flush=True)
            else:
                print(
                    f"  [resume] {name}: {len(frame)} rows != {expected_n}; recomputing",
                    flush=True,
                )

        if records is None:
            ckpt = ckpt_by_method.get(name)
            if ckpt is not None and not os.path.exists(ckpt):
                print(f"  [SKIP] {name}: checkpoint missing ({ckpt})", flush=True)
                continue
            print(f"[method] {name}", flush=True)
            try:
                if name == "PCD-Net":
                    denoiser = build_v9_denoiser(ckpt, device)
                elif name == "ProtoCondTCN":
                    denoiser = build_proto_cond_tcn_denoiser(ckpt, device)
                elif name in ("GDiffuSE", "CDiffuSE"):
                    denoiser = build_diffusion_denoiser(
                        name, ckpt, device, diffusion_settings
                    )
                elif name in ("Bandpass", "Wavelet"):
                    denoiser = build_traditional_denoiser(name, args.fs)
                else:
                    denoiser = build_baseline_denoiser(name, ckpt, device)
            except Exception as exc:
                if not args.allow_missing:
                    raise
                print(f"  [SKIP] {name}: {exc}", flush=True)
                continue

            records, snr_seen, elapsed = evaluate_method(
                denoiser, loader, device, cfg, assign_group, reference_snr
            )
            pd.DataFrame(records).to_csv(per_model_path, index=False, encoding="utf-8")
            del denoiser
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if reference_snr is None:
            reference_snr = list(snr_seen)
        elif len(snr_seen) != len(reference_snr) or not np.allclose(
            reference_snr, snr_seen, atol=1e-6
        ):
            raise RuntimeError(
                f"{name} did not see the reference test set; the paired "
                "comparison would be invalid"
            )

        row, groups = summarize(records, name, elapsed, group_labels)
        if row is None:
            continue
        global_rows.append(row)
        group_rows.extend(groups)
        evaluated_order.append(name)

        frame = pd.DataFrame(records)
        audit["methods"][name] = {
            "N": int(len(frame)),
            "snr_in_mean": float(frame["snr_in"].mean()),
            "snr_in_min": float(frame["snr_in"].min()),
            "snr_in_max": float(frame["snr_in"].max()),
            "snr_group_counts": {
                label: int((frame["snr_group"] == label).sum())
                for label in group_labels
            },
            "checkpoint": ckpt_by_method.get(name),
            "checkpoint_sha1_16": _sha1(ckpt_by_method.get(name)),
            "uses_condition": bool(MODEL_INFO[name][1]),
        }

    if not global_rows:
        raise RuntimeError("no method was evaluated successfully")

    global_df = pd.DataFrame(global_rows)
    global_df = global_df.sort_values("ΔSNR(dB)", ascending=False).reset_index(drop=True)
    group_df = pd.DataFrame(group_rows)

    rounded = global_df.copy()
    for column in (
        "ΔSNR(dB)", "SNR_in(dB)", "SNR_out(dB)", "CC", "RMSE", "PRD",
        "ST-MAE(noisy)", "ST-MAE(denoised)", "Model_quality", "Fidelity_score",
        "Quality_fidelity_corr", "Pick_succ_noisy", "Pick_succ_denoise",
    ):
        if column in rounded.columns:
            rounded[column] = rounded[column].round(4)
    rounded.to_csv(out_dir / "global_compare.csv", index=False, encoding="utf-8-sig")
    if not group_df.empty:
        group_df.round(4).to_csv(
            out_dir / "per_group_compare.csv", index=False, encoding="utf-8-sig"
        )

    write_publication_table(out_dir, evaluated_order)
    plot_bar(global_df, out_dir / "compare_bar.png", "PCD-Net")
    write_summary(out_dir, global_df, group_df, cfg, group_labels)

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "device": str(device),
                "resolved_group_labels": group_labels,
                "n_test_samples": expected_n,
                "cwd": os.getcwd(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "test_set_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[done] results written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
