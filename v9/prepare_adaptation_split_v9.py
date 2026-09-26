"""Create a reproducible target train-pool and held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--dataset_type", choices=("mining", "stead"), default="mining")
    args = parser.parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    frame = pd.read_csv(args.input_csv, low_memory=False)
    if args.dataset_type == "mining":
        required = {"sampling_rate", "p_arrival_sample"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"mining CSV is missing columns: {sorted(missing)}")
        sampling_rate = pd.to_numeric(frame["sampling_rate"], errors="coerce")
        p_arrival = pd.to_numeric(frame["p_arrival_sample"], errors="coerce")
        frame = frame[sampling_rate.isin((50.0, 100.0)) & (p_arrival >= 0)].reset_index(drop=True)
    elif "p_arrival_sample" in frame.columns:
        frame = frame[pd.to_numeric(frame["p_arrival_sample"], errors="coerce") >= 0].reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("target CSV must contain at least 10 rows")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(frame))
    test_count = max(1, int(round(len(frame) * args.test_fraction)))
    test_indices = order[:test_count]
    train_indices = order[test_count:]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"target_train_pool_seed{args.seed}.csv"
    test_path = output_dir / f"target_test_seed{args.seed}.csv"
    frame.iloc[train_indices].to_csv(train_path, index=False)
    frame.iloc[test_indices].to_csv(test_path, index=False)
    print(f"train_pool={len(train_indices)} -> {train_path}")
    print(f"held_out_test={len(test_indices)} -> {test_path}")


if __name__ == "__main__":
    main()
