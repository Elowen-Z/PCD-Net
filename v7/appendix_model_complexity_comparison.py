"""Appendix model complexity and inference-time comparison.

The script measures parameter count and single-sample inference latency for
PCD-Net and the baseline denoisers used in the main comparison experiment.
FLOPs are reported when PyTorch can estimate them; traditional signal
processing methods are marked as not applicable for parameter count/FLOPs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

import torch

from v5.compare_models import build_all_denoisers


METHODS = [
    "V7",
    "DPRNN",
    "Restormer1D",
    "DeepDenoiser",
    "Wavelet",
    "Bandpass",
]

DISPLAY_NAMES = {
    "V7": "PCD-Net",
    "DPRNN": "DPRNN",
    "Restormer1D": "Restormer1D",
    "DeepDenoiser": "DeepDenoiser",
    "Wavelet": "Wavelet",
    "Bandpass": "Bandpass",
}


def configure_cuda_safe() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def count_parameters(denoiser) -> tuple[int | None, int | None]:
    model = getattr(denoiser, "model", None)
    if model is None:
        return None, None
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_latency(
    denoiser,
    noisy: torch.Tensor,
    z_cond: torch.Tensor,
    repetitions: int,
    warmup: int,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        with torch.inference_mode():
            denoiser.denoise(noisy, z_cond)
    synchronize(noisy.device)

    timings = []
    for _ in range(repetitions):
        synchronize(noisy.device)
        started = time.perf_counter()
        with torch.inference_mode():
            denoiser.denoise(noisy, z_cond)
        synchronize(noisy.device)
        timings.append(1000.0 * (time.perf_counter() - started))

    ordered = sorted(timings)
    if len(ordered) >= 20:
        ordered = ordered[2:-2]
    mean = float(sum(ordered) / len(ordered))
    variance = float(sum((x - mean) ** 2 for x in ordered) / len(ordered))
    return mean, math.sqrt(variance), 1000.0 / max(mean, 1e-8)


def estimate_flops(
    denoiser,
    noisy: torch.Tensor,
    z_cond: torch.Tensor,
) -> float | None:
    model = getattr(denoiser, "model", None)
    if model is None:
        return None
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if noisy.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            with torch.inference_mode():
                denoiser.denoise(noisy, z_cond)
        total = sum(event.flops for event in prof.key_averages() if event.flops)
        return float(total) if total else None
    except Exception:
        return None


def fmt_million(value: int | None) -> str:
    return "-" if value is None else f"{value / 1e6:.3f}"


def fmt_gflops(value: float | None) -> str:
    if value is None or value <= 0:
        return "-"
    gflops = value / 1e9
    return "<0.001" if gflops < 0.001 else f"{gflops:.3f}"


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "model_complexity_latency_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "Method",
            "Parameters_M",
            "Trainable_Parameters_M",
            "FLOPs_G",
            "Latency_ms",
            "Latency_std_ms",
            "Throughput_samples_per_s",
            "Device",
            "Input",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| Method | Params (M) | FLOPs (G) | Latency (ms/sample) | Throughput (samples/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {Method} | {Parameters_M} | {FLOPs_G} | {Latency_ms} | {Throughput_samples_per_s} |".format(
                **row
            )
        )
    (out_dir / "model_complexity_latency_table.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    tex_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Method & Params (M) & FLOPs (G) & Latency (ms/sample) & Throughput (samples/s) \\",
        r"\hline",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['Method']} & {row['Parameters_M']} & {row['FLOPs_G']} & "
            f"{row['Latency_ms']} & {row['Throughput_samples_per_s']} \\\\"
        )
    tex_lines.extend([r"\hline", r"\end{tabular}"])
    (out_dir / "model_complexity_latency_table.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--signal_len", type=int, default=6000)
    parser.add_argument("--cond_len", type=int, default=400)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        default="v7/paper_experiments/appendix/model_complexity_latency_comparison",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fast_cuda", action="store_true")
    parser.add_argument("--skip_flops", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.fast_cuda:
        configure_cuda_safe()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    torch.manual_seed(args.seed)

    selected = [item.strip() for item in args.methods.split(",") if item.strip()]
    denoisers = build_all_denoisers(selected, device)
    noisy = torch.randn(1, 3, args.signal_len, device=device)
    z_cond = torch.randn(1, 3, args.cond_len, device=device)
    rows = []

    for method in selected:
        if method not in denoisers:
            print(f"[skip] {method}: not available", flush=True)
            continue
        denoiser = denoisers[method]
        total, trainable = count_parameters(denoiser)
        latency, latency_std, throughput = measure_latency(
            denoiser, noisy, z_cond, args.repetitions, args.warmup
        )
        flops = None if args.skip_flops else estimate_flops(denoiser, noisy, z_cond)
        row = {
            "Method": DISPLAY_NAMES.get(method, method),
            "Parameters_M": fmt_million(total),
            "Trainable_Parameters_M": fmt_million(trainable),
            "FLOPs_G": fmt_gflops(flops),
            "Latency_ms": f"{latency:.2f}",
            "Latency_std_ms": f"{latency_std:.2f}",
            "Throughput_samples_per_s": f"{throughput:.2f}",
            "Device": str(device),
            "Input": f"1x3x{args.signal_len}, cond 1x3x{args.cond_len}",
        }
        rows.append(row)
        print(
            f"[{row['Method']}] params={row['Parameters_M']}M "
            f"flops={row['FLOPs_G']}G latency={row['Latency_ms']} ms",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_dir = Path(args.output_dir)
    write_outputs(rows, out_dir)
    (out_dir / "model_complexity_latency_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print(f"\n[DONE] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
