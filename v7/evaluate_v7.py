"""Evaluate a trained Feedback PCD-Net V7 checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from v3.dataset_v3 import STEADDatasetV3
from v5.dataset_mining import MiningDatasetV6
from v5.train_v6 import cc_batch, compute_snr_batch, split_dataset
from v7.model_v7 import NoiseAwareDenoiserV7


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_type", choices=["stead", "mining"], default="stead")
    parser.add_argument("--event_h5", required=True)
    parser.add_argument("--event_csv", required=True)
    parser.add_argument("--noise_h5", required=True)
    parser.add_argument("--noise_csv", required=True)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adaptive_stop", action="store_true")
    parser.add_argument("--st_window", type=int, default=100)
    parser.add_argument("--output", default=None)
    return parser


def make_dataset(args, checkpoint_config):
    _, val_csv = split_dataset(
        args.event_csv,
        0.1,
        args.seed,
        prefix=f"v7_eval_seed{args.seed}",
    )
    dataset_class = (
        MiningDatasetV6
        if args.dataset_type == "mining"
        else STEADDatasetV3
    )
    kwargs = {
        "event_h5_path": args.event_h5,
        "event_csv_path": val_csv,
        "noise_h5_path": args.noise_h5,
        "noise_csv_path": args.noise_csv,
        "signal_len": checkpoint_config.get("signal_len", 6000),
        "cond_len": checkpoint_config.get("cond_len", 400),
        "snr_range": tuple(checkpoint_config.get("snr_range", (0.1, 20.0))),
        "clean_prob": 0.0,
        "part_b_ratio": 0.0,
        "seed": args.seed,
    }
    if args.dataset_type == "mining":
        kwargs["eval_mode"] = True
    dataset = dataset_class(**kwargs)
    if args.max_samples > 0 and len(dataset) > args.max_samples:
        indices = np.random.default_rng(args.seed).choice(
            len(dataset), size=args.max_samples, replace=False
        )
        dataset = Subset(dataset, indices.tolist())
    return dataset


def build_model(checkpoint, device):
    config = checkpoint.get("config", {})
    variant = config.get("variant", "full")
    model = NoiseAwareDenoiserV7(
        in_ch=3,
        z_dim=config.get("z_dim", 128),
        cond_len=config.get("cond_len", 400),
        num_prototypes=config.get("num_prototypes", 16),
        top_m=config.get("top_m", 4),
        num_heads=config.get("num_heads", 4),
        n_refine=config.get("n_refine", 3),
        base_ch=config.get("base_ch", 32),
        vq_temperature=config.get("vq_temperature", 0.3),
        use_prototypes=config.get(
            "use_prototypes", variant != "no_prototypes"
        ),
        use_sparse_selection=config.get(
            "use_sparse_selection", variant != "no_sparse"
        ),
        use_cross_attn=config.get(
            "use_cross_attn", variant != "no_cross_attn"
        ),
        use_quality_head=config.get(
            "use_quality_head", variant != "no_quality"
        ),
        use_residual_feedback=config.get(
            "use_residual_feedback", variant != "no_feedback"
        ),
        adaptive_inference=config.get("adaptive_inference", True),
        stop_threshold=config.get("stop_threshold", 0.95),
        min_refine_steps=config.get("min_refine_steps", 1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config


@torch.inference_mode()
def evaluate(model, loader, device, adaptive_stop, st_window):
    gains, correlations, rmses, prds = [], [], [], []
    st_maes, event_st_maes = [], []
    quality_values, fidelity_values = [], []
    step_values, stopped_values = [], []
    selected_mass = []
    processed = 0
    started = time.perf_counter()

    for batch_index, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        target = batch["y_clean"].to(device, non_blocking=True)
        condition = batch["z_cond"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True)
        supervised = batch["has_target"].bool().to(device)
        if not supervised.any():
            continue

        pred, quality, _, aux = model(
            x, condition, adaptive_stop=adaptive_stop
        )
        pred = pred[supervised]
        target = target[supervised]
        noisy = x[supervised]
        mask = mask[supervised]

        snr_in = compute_snr_batch(target, noisy - target, mask)
        snr_out = compute_snr_batch(target, pred - target, mask)
        gains.extend((snr_out - snr_in).cpu().tolist())
        correlations.extend(cc_batch(pred, target).cpu().tolist())

        error = pred - target
        rmse = torch.sqrt(torch.mean(error ** 2, dim=(1, 2)))
        reference_rms = torch.sqrt(
            torch.mean(target ** 2, dim=(1, 2))
        ).clamp_min(1e-8)
        prd = 100.0 * rmse / reference_rms
        rmses.extend(rmse.cpu().tolist())
        prds.extend(prd.cpu().tolist())

        absolute_error = error.abs().mean(dim=1)
        length = absolute_error.size(-1)
        padded_length = (
            (length + st_window - 1) // st_window
        ) * st_window
        if padded_length != length:
            absolute_error = torch.nn.functional.pad(
                absolute_error, (0, padded_length - length)
            )
            mask_for_windows = torch.nn.functional.pad(
                mask, (0, padded_length - length)
            )
        else:
            mask_for_windows = mask
        window_error = absolute_error.unfold(-1, st_window, st_window).mean(-1)
        window_mask = mask_for_windows.unfold(
            -1, st_window, st_window
        ).mean(-1)
        st_maes.extend(window_error.mean(-1).cpu().tolist())
        event_weights = (window_mask > 0).float()
        event_st = (window_error * event_weights).sum(-1) / event_weights.sum(
            -1
        ).clamp_min(1.0)
        event_st_maes.extend(event_st.cpu().tolist())

        fidelity = torch.exp(-((rmse / reference_rms) ** 2))
        quality_values.extend(quality[supervised].squeeze(-1).cpu().tolist())
        fidelity_values.extend(fidelity.cpu().tolist())
        step_values.extend(
            aux["effective_steps"][supervised].cpu().tolist()
        )
        stopped_values.extend(
            aux["stopped_early"][supervised].float().cpu().tolist()
        )
        selected_mass.extend(
            aux["selected_mass"][supervised].cpu().tolist()
        )
        processed += int(supervised.sum())
        if batch_index % 25 == 0:
            print(
                f"[eval {processed}/{len(loader.dataset)}] "
                f"gain={np.mean(gains):+.2f} CC={np.mean(correlations):.4f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    quality_corr = float("nan")
    if len(quality_values) > 1 and np.std(quality_values) > 0:
        quality_corr = float(
            np.corrcoef(quality_values, fidelity_values)[0, 1]
        )
    return {
        "samples": processed,
        "adaptive_stop": adaptive_stop,
        "delta_snr_db": float(np.mean(gains)),
        "cc": float(np.mean(correlations)),
        "rmse": float(np.mean(rmses)),
        "prd_percent": float(np.mean(prds)),
        "st_mae": float(np.mean(st_maes)),
        "event_st_mae": float(np.mean(event_st_maes)),
        "st_window_samples": st_window,
        "quality_corr": quality_corr,
        "selected_mass": float(np.mean(selected_mass)),
        "effective_steps": float(np.mean(step_values)),
        "early_stop_rate": float(np.mean(stopped_values)),
        "elapsed_seconds": elapsed,
        "samples_per_second": processed / max(elapsed, 1e-8),
    }


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, config = build_model(checkpoint, device)
    dataset = make_dataset(args, config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    print(
        f"[checkpoint] epoch={checkpoint.get('epoch')} "
        f"CC={checkpoint.get('metrics', {}).get('val_cc')} "
        f"samples={len(dataset)} device={device}",
        flush=True,
    )
    metrics = evaluate(
        model, loader, device, args.adaptive_stop, args.st_window
    )
    print(json.dumps(metrics, indent=2, allow_nan=True), flush=True)
    output = args.output or str(
        Path(args.checkpoint).with_name("evaluation_v7.json")
    )
    Path(output).write_text(
        json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(f"[saved] {output}", flush=True)


if __name__ == "__main__":
    main()
