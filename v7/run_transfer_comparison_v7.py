"""Run V7 transfer ablations and aggregate their target-domain metrics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


INNOVATION_VARIANTS = [
    "full",
    "no_feedback",
    "no_sparse",
    "no_prototypes",
    "no_cross_attn",
    "no_quality",
]

FREEZE_STRATEGIES = [
    "noise_encoder",
    "signal_encoder",
    "signal_decoder",
    "prototype_feedback",
    "pcd_adaptation",
    "decoder_only",
    "feedback_decoder",
    "signal_backbone",
    "full",
]


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["mining", "nonnat"], required=True)
    parser.add_argument(
        "--suite", choices=["innovation", "freeze", "all"], default="all"
    )
    parser.add_argument(
        "--source_ckpt",
        default="v7/checkpoints_feedback_stead_seed0/best_model_v7.pth",
    )
    parser.add_argument("--output_root", default="v7/transfer_comparisons")
    parser.add_argument("--stage2_epochs", type=int, default=5)
    parser.add_argument("--stage3_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--checkpoint_every_batches", type=int, default=500)
    parser.add_argument(
        "--freeze_strategies",
        default=None,
        help=(
            "Comma-separated Stage-3 fine-tuning strategies for suite=freeze. "
            "Defaults to the built-in full list."
        ),
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def experiment_matrix(suite, freeze_strategies=None):
    experiments = []
    if suite in ("innovation", "all"):
        experiments.extend(
            {
                "name": f"innovation_{variant}",
                "variant": variant,
                "freeze_strategy": "signal_backbone",
                "suite": "innovation",
            }
            for variant in INNOVATION_VARIANTS
        )
    if suite in ("freeze", "all"):
        strategies = freeze_strategies or FREEZE_STRATEGIES
        experiments.extend(
            {
                "name": f"freeze_{strategy}",
                "variant": "full",
                "freeze_strategy": strategy,
                "suite": "freeze",
            }
            for strategy in strategies
        )
    return experiments


def completed_stage_epochs(save_dir, stage):
    history_path = save_dir / "transfer_history.json"
    if not history_path.exists():
        return 0
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return max(
        (
            int(row.get("epoch", 0))
            for row in history
            if int(row.get("stage", 0)) == stage
        ),
        default=0,
    )


def main():
    args = build_parser().parse_args()
    root = Path(args.output_root) / args.target
    root.mkdir(parents=True, exist_ok=True)
    requested_freeze = None
    if args.freeze_strategies:
        requested_freeze = [
            item.strip()
            for item in args.freeze_strategies.split(",")
            if item.strip()
        ]
        unknown = sorted(set(requested_freeze) - set(FREEZE_STRATEGIES))
        if unknown:
            raise ValueError(f"unknown freeze strategies: {unknown}")
    experiments = experiment_matrix(args.suite, requested_freeze)
    rows = []

    for summary_path in sorted(root.glob("*/transfer_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        suite = (
            "innovation"
            if summary_path.parent.name.startswith("innovation_")
            else "freeze"
        )
        rows.append({"suite": suite, **summary})

    def write_summary():
        if not rows:
            return
        fields = sorted({key for row in rows for key in row})
        with (root / "comparison_summary.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_summary()

    for experiment in experiments:
        save_dir = root / experiment["name"]
        summary_path = save_dir / "transfer_summary.json"
        if summary_path.exists():
            print(f"[SKIP completed] {experiment['name']}")
            continue
        stage2_checkpoint = save_dir / "best_stage2_v7.pth"
        stage2_last_checkpoint = save_dir / "last_stage2_v7.pth"
        stage2_progress_checkpoint = save_dir / "progress_stage2_v7.pth"
        stage3_checkpoint = save_dir / "last_transfer_v7.pth"

        def build_command():
            command = [
                sys.executable,
                "-u",
                "-m",
                "v7.transfer_staged_v7",
                "--target",
                args.target,
                "--source_ckpt",
                args.source_ckpt,
                "--save_dir",
                str(save_dir),
                "--stage2_epochs",
                str(args.stage2_epochs),
                "--stage3_epochs",
                str(args.stage3_epochs),
                "--batch_size",
                str(min(args.batch_size, 1)),
                "--num_workers",
                str(args.num_workers),
                "--print_every",
                str(args.print_every),
                "--checkpoint_every_batches",
                str(args.checkpoint_every_batches),
                "--seed",
                str(args.seed),
                "--variant",
                experiment["variant"],
                "--freeze_strategy",
                experiment["freeze_strategy"],
                "--cuda_safe",
                "--no_amp",
            ]
            stage2_completed = (
                completed_stage_epochs(save_dir, stage=2)
                >= args.stage2_epochs
            )
            if stage2_completed and stage2_checkpoint.exists():
                command.extend(
                    [
                        "--skip_stage2",
                        "--stage2_ckpt",
                        str(stage2_checkpoint),
                    ]
                )
            elif stage2_progress_checkpoint.exists():
                command.extend(
                    [
                        "--resume_stage2",
                        str(stage2_progress_checkpoint),
                    ]
                )
            elif stage2_last_checkpoint.exists():
                command.extend(
                    ["--resume_stage2", str(stage2_last_checkpoint)]
                )
            if stage2_completed and stage3_checkpoint.exists():
                command.extend(
                    ["--resume_stage3", str(stage3_checkpoint)]
                )
            return command

        command = build_command()
        print("\n" + "=" * 80)
        print(experiment["name"])
        print(" ".join(command))
        if not args.dry_run:
            for attempt in range(args.max_retries + 1):
                command = build_command()
                try:
                    subprocess.run(command, check=True)
                    break
                except subprocess.CalledProcessError:
                    if attempt >= args.max_retries:
                        raise
                    delay = 10
                    print(
                        f"[RETRY {attempt + 1}/{args.max_retries}] "
                        f"process failed; resume in {delay}s"
                    )
                    time.sleep(delay)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append({"suite": experiment["suite"], **summary})
            write_summary()

    if not args.dry_run:
        print(f"\n[DONE] {root / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()
