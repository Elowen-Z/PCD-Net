"""Run target-domain few-shot transfer experiments for PCD-Net."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


STRATEGIES = [
    "noise_encoder",
    "prototype_feedback",
    "pcd_adaptation",
    "full",
]


def parse_list(value: str, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["mining", "nonnat", "both"], default="both")
    parser.add_argument(
        "--source_ckpt",
        default="v7/checkpoints_feedback_stead_seed0/best_model_v7.pth",
    )
    parser.add_argument(
        "--output_root",
        default="v7/paper_experiments/appendix/fewshot_transfer",
    )
    parser.add_argument("--sample_sizes", default="50,100,500,1000")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--stage2_epochs", type=int, default=3)
    parser.add_argument("--stage3_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_source", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--checkpoint_every_batches", type=int, default=500)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def completed(summary_path: Path) -> bool:
    return summary_path.exists()


def read_summary(summary_path: Path, suite_target: str, samples: int, strategy: str) -> dict:
    row = json.loads(summary_path.read_text(encoding="utf-8"))
    row.update(
        {
            "target": suite_target,
            "fewshot_samples": samples,
            "strategy": strategy,
            "save_dir": str(summary_path.parent),
        }
    )
    return row


def write_summary(rows: list[dict], root: Path) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with (root / "fewshot_transfer_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    targets = ["mining", "nonnat"] if args.target == "both" else [args.target]
    sample_sizes = parse_list(args.sample_sizes, int)
    strategies = parse_list(args.strategies)
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown:
        raise ValueError(f"unknown few-shot strategies: {unknown}")

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []

    for summary_path in sorted(root.glob("*/*/*/transfer_summary.json")):
        parts = summary_path.parts
        target = summary_path.parents[2].name
        samples = int(summary_path.parents[1].name.replace("n", ""))
        strategy = summary_path.parent.name
        rows.append(read_summary(summary_path, target, samples, strategy))
    write_summary(rows, root)

    for target in targets:
        for samples in sample_sizes:
            for strategy in strategies:
                save_dir = root / target / f"n{samples}" / strategy
                summary_path = save_dir / "transfer_summary.json"
                if completed(summary_path):
                    print(f"[SKIP completed] {target} n={samples} {strategy}")
                    continue
                save_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "v7.transfer_staged_v7",
                    "--target",
                    target,
                    "--source_ckpt",
                    args.source_ckpt,
                    "--save_dir",
                    str(save_dir),
                    "--stage2_epochs",
                    str(args.stage2_epochs),
                    "--stage3_epochs",
                    str(args.stage3_epochs),
                    "--batch_size",
                    str(args.batch_size),
                    "--num_workers",
                    str(args.num_workers),
                    "--print_every",
                    str(args.print_every),
                    "--checkpoint_every_batches",
                    str(args.checkpoint_every_batches),
                    "--seed",
                    str(args.seed),
                    "--variant",
                    "full",
                    "--freeze_strategy",
                    strategy,
                    "--max_target_train",
                    str(samples),
                    "--max_source",
                    str(args.max_source),
                    "--cuda_safe",
                    "--no_amp",
                ]
                print("\n" + "=" * 80)
                print(f"{target} few-shot n={samples} strategy={strategy}")
                print(" ".join(command))
                if args.dry_run:
                    continue
                for attempt in range(args.max_retries + 1):
                    try:
                        subprocess.run(command, check=True)
                        break
                    except subprocess.CalledProcessError:
                        if attempt >= args.max_retries:
                            raise
                        print(f"[RETRY {attempt + 1}/{args.max_retries}] resume in 10s")
                        time.sleep(10)
                rows.append(read_summary(summary_path, target, samples, strategy))
                write_summary(rows, root)

    if not args.dry_run:
        print(f"\n[DONE] {root / 'fewshot_transfer_summary.csv'}")


if __name__ == "__main__":
    main()
