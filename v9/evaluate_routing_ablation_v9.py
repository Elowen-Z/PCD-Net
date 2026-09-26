"""Paired evaluation of the V9 soft-routing ablation.

The full and uniform-routing checkpoints are evaluated on one deterministic
``EvalDataset`` instance.  This keeps the clean waveform, injected noise,
input SNR, and pre-event context identical for both variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from v5.evaluate_transfer_v5 import EvalDataset
from v9.compare_v9 import build_group_spec, build_v9_denoiser, evaluate_method
from v7.train_v7 import configure_cuda_backend


VARIANTS = (
    ("Full soft routing", True),
    ("w/o routing (uniform alpha)", False),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full_checkpoint", required=True)
    parser.add_argument("--no_routing_checkpoint", required=True)
    parser.add_argument("--no_prototype_checkpoint")
    parser.add_argument("--weighted_concat_checkpoint")
    parser.add_argument("--event_h5", default="E:/p_wave/data/chunk2.hdf5")
    parser.add_argument("--event_csv", default="v5/v7_seed0_val.csv")
    parser.add_argument("--noise_h5", default="E:/p_wave/data/chunk1.hdf5")
    parser.add_argument("--noise_csv", default="E:/p_wave/data/chunk1.csv")
    parser.add_argument("--snr_min_db", type=float, default=-15.0)
    parser.add_argument("--snr_max_db", type=float, default=10.0)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--signal_len", type=int, default=6000)
    parser.add_argument("--cond_len", type=int, default=400)
    parser.add_argument("--allow_protocol_mismatch", action="store_true")
    parser.add_argument("--cuda_safe", action="store_true",
                        help="disable cuDNN and TF32 for GPU compatibility")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output_dir", default="v9/paper_experiments/ablation_v9_seed0/routing")
    return parser


def checkpoint_config(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    config.setdefault("prototype_interaction", "cross_attention")
    return config


def sha1_16(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def verify_checkpoints(
    checkpoints: dict[str, str], allow_protocol_mismatch: bool = False
) -> dict[str, dict]:
    configs = {name: checkpoint_config(path) for name, path in checkpoints.items()}
    full = configs["Full PCD-Net"]
    if not full.get("use_routing", True) or not full.get("use_prototypes", True):
        raise ValueError("the full checkpoint must use prototypes and soft routing")
    comparable_keys = (
        "context_source", "snr_range", "snr_sampling", "clean_prob",
        "num_prototypes", "base_ch", "n_refine", "train_samples",
        "val_samples", "epochs", "batch_size",
    )
    for name, config in configs.items():
        mismatches = {
            key: (full.get(key), config.get(key))
            for key in comparable_keys
            if full.get(key) != config.get(key)
        }
        if mismatches and not allow_protocol_mismatch:
            details = ", ".join(
                f"{key}: full={values[0]!r}, variant={values[1]!r}"
                for key, values in mismatches.items()
            )
            raise ValueError(f"{name} protocol is not comparable: {details}")
        if name == "w/o routing (uniform alpha)" and config.get("use_routing", True):
            raise ValueError("no-routing checkpoint has use_routing=True")
        if name == "w/o prototype memory" and config.get("use_prototypes", True):
            raise ValueError("no-prototype checkpoint has use_prototypes=True")
        if name == "weighted-sum + concat fusion" and config.get("prototype_interaction") != "weighted_concat":
            raise ValueError("weighted-concat checkpoint has the wrong interaction mode")
        if mismatches:
            print(f"[warning] protocol mismatch allowed for smoke test ({name}): {mismatches}", flush=True)
    return configs


def metric_summary(records: list[dict], variant: str) -> dict:
    frame = pd.DataFrame(records)
    return {
        "Variant": variant,
        "N": len(frame),
        "DeltaSNR_dB": float(frame["delta_snr"].mean()),
        "CC": float(frame["cc"].mean()),
        "RMSE": float(frame["rmse"].mean()),
        "ST_MAE": float(frame["st_mae_denoised"].mean()),
    }


def write_latex(frame: pd.DataFrame, path: Path) -> None:
    formatted = frame.copy()
    formatted["DeltaSNR_dB"] = formatted["DeltaSNR_dB"].map(lambda x: f"{x:.4f}")
    for column in ("CC", "RMSE", "ST_MAE"):
        formatted[column] = formatted[column].map(lambda x: f"{x:.4f}")
    formatted = formatted.rename(columns={
        "DeltaSNR_dB": r"$\Delta$SNR (dB)",
        "ST_MAE": "ST-MAE",
    })
    path.write_text(
        formatted.to_latex(index=False, escape=False), encoding="utf-8"
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.snr_min_db >= args.snr_max_db:
        raise ValueError("snr_min_db must be smaller than snr_max_db")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {"Full PCD-Net": args.full_checkpoint,
                   "w/o routing (uniform alpha)": args.no_routing_checkpoint}
    if args.no_prototype_checkpoint:
        checkpoints["w/o prototype memory"] = args.no_prototype_checkpoint
    if args.weighted_concat_checkpoint:
        checkpoints["weighted-sum + concat fusion"] = args.weighted_concat_checkpoint
    checkpoint_configs = verify_checkpoints(
        checkpoints, allow_protocol_mismatch=args.allow_protocol_mismatch
    )
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    configure_cuda_backend(args.cuda_safe and device.type == "cuda")
    print(f"[device] {device} cuda_safe={args.cuda_safe}", flush=True)
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
        context_source="pre_event",
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    labels, _, assign_group = build_group_spec(None)
    summaries = []
    reference_snr = None
    for variant, checkpoint_path in checkpoints.items():
        print(f"[variant] {variant}", flush=True)
        denoiser = build_v9_denoiser(checkpoint_path, device)
        denoiser.name = variant
        records, snr_seen, elapsed = evaluate_method(
            denoiser,
            loader,
            device,
            {
                "fs": 100.0, "sta_len": 0.5, "lta_len": 10.0,
                "stalta_thr": 2.0, "pick_tol": 50,
            },
            assign_group,
            expected_snr_in=reference_snr,
        )
        if reference_snr is None:
            reference_snr = snr_seen
        slug = "".join(char.lower() if char.isalnum() else "_" for char in variant).strip("_")
        pd.DataFrame(records).to_csv(
            output_dir / f"per_sample_{slug}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        row = metric_summary(records, variant)
        row["time_sec"] = elapsed
        summaries.append(row)
        del denoiser
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame(summaries)
    table.to_csv(output_dir / "routing_ablation_table.csv", index=False, encoding="utf-8-sig")
    write_latex(table.drop(columns="time_sec"), output_dir / "routing_ablation_table.tex")
    audit = {
        "protocol": {
            "event_h5": args.event_h5,
            "event_csv": args.event_csv,
            "noise_h5": args.noise_h5,
            "noise_csv": args.noise_csv,
            "context_source": "pre_event",
            "input_snr_db": [args.snr_min_db, args.snr_max_db],
            "seed": args.seed,
            "samples": len(dataset),
        },
        "checkpoints": {
            name: {"path": path, "sha1_16": sha1_16(path), "config": checkpoint_configs[name]}
            for name, path in checkpoints.items()
        },
    }
    (output_dir / "routing_ablation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False), flush=True)
    print(f"[saved] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
