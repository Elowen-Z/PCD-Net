# -*- coding: utf-8 -*-
"""Envelope waveform visualization for non-natural transfer fine-tuning."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import v7.plot_mining_transfer_finetune_waveform as base
from v7.transfer_staged_v7 import TARGETS, make_dataset


NONNAT_ROOT = Path("v7/transfer_comparisons/nonnat")
NONNAT_OUT = Path("v7/paper_experiments/nonnat_transfer_finetune_ablation")


def val_csv_for(strategy: str = "pcd_adaptation") -> Path:
    path = NONNAT_ROOT / f"freeze_{strategy}" / "nonnat_val.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_dataset():
    val_csv = val_csv_for()
    return (
        make_dataset(TARGETS["nonnat"], str(val_csv), seed=0, evaluation=True),
        pd.read_csv(val_csv, low_memory=False),
    )


def main() -> None:
    base.TRANSFER_ROOT = NONNAT_ROOT
    base.OUT_DIR = NONNAT_OUT
    base.val_csv_for = val_csv_for
    base.load_dataset = load_dataset
    base.main()


if __name__ == "__main__":
    main()
