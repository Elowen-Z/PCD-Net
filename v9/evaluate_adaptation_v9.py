"""Paired target-domain evaluation for V9 adaptation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from v5.evaluate_transfer_v5 import EvalDataset
from v9.compare_v9 import build_group_spec, build_v9_denoiser, evaluate_method


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="NAME=PATH; repeat for each adaptation strategy")
    parser.add_argument("--event_h5", required=True)
    parser.add_argument("--event_csv", required=True,
                        help="held-out target-domain test CSV")
    parser.add_argument("--noise_h5", required=True)
    parser.add_argument("--noise_csv", required=True)
    parser.add_argument("--context_source", choices=("pre_event", "paired_noise"), default="pre_event")
    parser.add_argument("--snr_min_db", type=float, default=-15.0)
    parser.add_argument("--snr_max_db", type=float, default=10.0)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--signal_len", type=int, default=6000)
    parser.add_argument("--cond_len", type=int, default=400)
    parser.add_argument("--output_dir", required=True)
    return parser


def sha1_16(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def main() -> None:
    args = build_parser().parse_args()
    checkpoints = {}
    for value in args.checkpoint:
        if "=" not in value:
            raise ValueError("each --checkpoint must have the form NAME=PATH")
        name, path = value.split("=", 1)
        if name in checkpoints:
            raise ValueError(f"duplicate checkpoint name: {name}")
        checkpoints[name] = path
    if args.snr_min_db >= args.snr_max_db:
        raise ValueError("snr_min_db must be smaller than snr_max_db")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EvalDataset(
        event_h5_path=args.event_h5,
        event_csv_path=args.event_csv,
        noise_h5_path=args.noise_h5,
        noise_csv_path=args.noise_csv,
        signal_len=args.signal_len,
        cond_len=args.cond_len,
        snr_db_range=(args.snr_min_db, args.snr_max_db),
        max_samples=args.max_samples,
        seed=args.seed,
        context_source=args.context_source,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    _, _, assign_group = build_group_spec(None)
    rows = []
    audit = {"protocol": vars(args), "samples": len(dataset), "checkpoints": {}}
    reference_snr = None
    for name, path in checkpoints.items():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        print(f"[evaluate] {name}", flush=True)
        denoiser = build_v9_denoiser(path, device)
        denoiser.name = name
        records, snr_seen, elapsed = evaluate_method(
            denoiser, loader, device,
            {"fs": 100.0, "sta_len": 0.5, "lta_len": 10.0,
             "stalta_thr": 2.0, "pick_tol": 50},
            assign_group, expected_snr_in=reference_snr,
        )
        if reference_snr is None:
            reference_snr = snr_seen
        per_sample_path = output_dir / f"per_sample_{len(rows)}.csv"
        pd.DataFrame(records).to_csv(per_sample_path, index=False, encoding="utf-8-sig")
        frame = pd.DataFrame(records)
        rows.append({
            "Strategy": name,
            "N": len(frame),
            "DeltaSNR_dB": float(frame["delta_snr"].mean()),
            "CC": float(frame["cc"].mean()),
            "RMSE": float(frame["rmse"].mean()),
            "ST_MAE": float(frame["st_mae_denoised"].mean()),
            "time_sec": elapsed,
        })
        audit["checkpoints"][name] = {"path": path, "sha1_16": sha1_16(path)}
        del denoiser
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "adaptation_ablation_table.csv", index=False, encoding="utf-8-sig")
    tex = table.drop(columns="time_sec").rename(
        columns={"DeltaSNR_dB": r"$\Delta$SNR (dB)", "ST_MAE": "ST-MAE"}
    )
    (output_dir / "adaptation_ablation_table.tex").write_text(
        tex.to_latex(index=False, escape=False, float_format="%.4f"), encoding="utf-8"
    )
    (output_dir / "adaptation_ablation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False), flush=True)
    print(f"[saved] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
