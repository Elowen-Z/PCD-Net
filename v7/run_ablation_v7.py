"""Launch the paper-required V7 ablations into v7/exp_runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "full": [],
    # Paper-core ablations aligned with the three claimed innovations.
    "no_proto_sparse": [
        "--no_prototypes",
        "--no_sparse_selection",
        "--alpha_sparse",
        "0",
        "--alpha_balance",
        "0",
    ],
    "no_cross_feedback": ["--no_cross_attn", "--no_residual_feedback"],
    "no_quality_adaptive": [
        "--no_quality_head",
        "--alpha_quality",
        "0",
        "--no_adaptive_inference",
    ],
    "no_prototype": ["--no_prototypes", "--alpha_sparse", "0", "--alpha_balance", "0"],
    "no_sparse_selection": ["--no_sparse_selection"],
    "no_sparse_loss": ["--alpha_sparse", "0"],
    "no_balance_loss": ["--alpha_balance", "0"],
    "no_cross_attn": ["--no_cross_attn"],
    "no_quality": ["--no_quality_head", "--alpha_quality", "0"],
    "no_feedback": ["--no_residual_feedback"],
    "no_adaptive_stop": ["--no_adaptive_inference"],
    "no_intermediate": ["--alpha_intermediate", "0"],
    "no_refine": ["--n_refine", "0"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--dataset_type", choices=["stead", "mining"], default="stead")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_m", type=int, default=4)
    parser.add_argument("--init_from")
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args, passthrough = parser.parse_known_args()

    selected = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in selected if name not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")

    for name in selected:
        output = Path("v7/exp_runs/ablation_v7") / name
        history_path = output / "history.json"
        last_path = output / "last_model_v7.pth"
        completed_epoch = 0
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
                if history:
                    completed_epoch = max(int(row.get("epoch", 0)) for row in history)
            except Exception:
                completed_epoch = 0
        if args.skip_completed and completed_epoch >= args.epochs:
            print(
                f"[skip] {name}: completed epoch {completed_epoch}/{args.epochs}"
            )
            continue

        command = [
            sys.executable,
            "-m",
            "v7.train_v7",
            "--dataset_type",
            args.dataset_type,
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--top_m",
            str(args.top_m),
            "--save_dir",
            str(output),
            *VARIANTS[name],
            *passthrough,
        ]
        if args.resume_existing and last_path.exists() and completed_epoch < args.epochs:
            command.extend(["--resume_from", str(last_path)])
        elif args.init_from:
            command.extend(["--init_from", args.init_from])
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
