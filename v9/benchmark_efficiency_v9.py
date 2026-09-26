"""Benchmark PCD-Net V9 and paper baselines under one inference protocol.

Neural-model FLOPs use the explicit convention ``FLOPs = 2 * THOP MACs``.
Latency is end-to-end single-sample inference time, including host-side work for
the classical baselines.  RTF is latency divided by the 60-second input length.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import thop
from thop import profile

from v9.compare_v9 import (
    DEFAULT_CKPT_DEEPDENOISER,
    DEFAULT_CKPT_DPRNN,
    DEFAULT_CKPT_PROTO_COND_TCN,
    DEFAULT_CKPT_RESTORMER1D,
    DEFAULT_CKPT_V9,
    build_baseline_denoiser,
    build_proto_cond_tcn_denoiser,
    build_traditional_denoiser,
    build_v9_denoiser,
)


METHOD_ORDER = [
    "Bandpass",
    "Wavelet",
    "DeepDenoiser",
    "DPRNN",
    "Restormer1D",
    "ProtoCondTCN",
    "PCD-Net",
]


class ProfileModule(nn.Module):
    """Expose each neural denoiser as a two-input module for THOP."""

    def __init__(self, denoiser):
        super().__init__()
        self.model = denoiser.model
        self.use_condition = bool(denoiser.uses_condition)

    def forward(self, noisy: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        output = self.model(noisy, context) if self.use_condition else self.model(noisy)
        return output[0] if isinstance(output, tuple) else output


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(
    denoiser,
    noisy: torch.Tensor,
    context: torch.Tensor,
    warmup: int,
    repetitions: int,
) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            denoiser.denoise(noisy, context)
    synchronize(noisy.device)

    timings = []
    with torch.inference_mode():
        for _ in range(repetitions):
            synchronize(noisy.device)
            started = time.perf_counter()
            denoiser.denoise(noisy, context)
            synchronize(noisy.device)
            timings.append(1000.0 * (time.perf_counter() - started))

    ordered = sorted(timings)
    if len(ordered) >= 20:
        trim = max(1, int(0.05 * len(ordered)))
        ordered = ordered[trim:-trim]
    mean = sum(ordered) / len(ordered)
    variance = sum((value - mean) ** 2 for value in ordered) / len(ordered)
    return float(mean), float(math.sqrt(variance))


def neural_complexity(
    denoiser,
    noisy: torch.Tensor,
    context: torch.Tensor,
) -> tuple[float, float, float]:
    module = ProfileModule(denoiser).eval()
    with torch.inference_mode():
        macs, _ = profile(module, inputs=(noisy, context), verbose=False)
    parameters = sum(parameter.numel() for parameter in module.parameters())
    macs_g = float(macs) / 1e9
    flops_g = 2.0 * macs_g
    return parameters / 1e6, macs_g, flops_g


def build_denoisers(args, device: torch.device) -> dict:
    return {
        "Bandpass": build_traditional_denoiser("Bandpass", args.sample_rate),
        "Wavelet": build_traditional_denoiser("Wavelet", args.sample_rate),
        "DeepDenoiser": build_baseline_denoiser(
            "DeepDenoiser", args.ckpt_deepdenoiser, device
        ),
        "DPRNN": build_baseline_denoiser("DPRNN", args.ckpt_dprnn, device),
        "Restormer1D": build_baseline_denoiser(
            "Restormer1D", args.ckpt_restormer1d, device
        ),
        "ProtoCondTCN": build_proto_cond_tcn_denoiser(
            args.ckpt_proto_cond_tcn, device
        ),
        "PCD-Net": build_v9_denoiser(args.ckpt_v9, device),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & FLOPs & Lat. & Thr. & RTF \\",
        r"       & (G) & (ms) & (samp./s) & ($10^{-4}$) \\",
        r"\midrule",
    ]
    for row in rows:
        method = row["Method"].replace("ProtoCondTCN", "ProtoCond-TCN")
        flops = row["FLOPs_G"] or "--"
        lines.append(
            f"{method} & {flops} & {row['Latency_ms']} & "
            f"{row['Throughput_samples_per_s']} & {row['RTF_1e-4']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_v9", default=DEFAULT_CKPT_V9)
    parser.add_argument("--ckpt_deepdenoiser", default=DEFAULT_CKPT_DEEPDENOISER)
    parser.add_argument("--ckpt_dprnn", default=DEFAULT_CKPT_DPRNN)
    parser.add_argument("--ckpt_restormer1d", default=DEFAULT_CKPT_RESTORMER1D)
    parser.add_argument("--ckpt_proto_cond_tcn", default=DEFAULT_CKPT_PROTO_COND_TCN)
    parser.add_argument("--signal_len", type=int, default=6000)
    parser.add_argument("--cond_len", type=int, default=400)
    parser.add_argument("--sample_rate", type=float, default=100.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--output_dir",
        default="v9/paper_experiments/efficiency_comparison",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    denoisers = build_denoisers(args, device)
    noisy = torch.randn(1, 3, args.signal_len, device=device)
    context = torch.randn(1, 3, args.cond_len, device=device)
    duration_seconds = args.signal_len / args.sample_rate
    rows = []

    for method in METHOD_ORDER:
        denoiser = denoisers[method]
        params_m = macs_g = flops_g = None
        if hasattr(denoiser, "model"):
            params_m, macs_g, flops_g = neural_complexity(denoiser, noisy, context)
        latency_ms, latency_std_ms = measure_latency(
            denoiser, noisy, context, args.warmup, args.repetitions
        )
        throughput = 1000.0 / latency_ms
        rtf = (latency_ms / 1000.0) / duration_seconds
        row = {
            "Method": method,
            "FLOPs_G": "" if flops_g is None else f"{flops_g:.3f}",
            "Latency_ms": f"{latency_ms:.3f}",
            "Throughput_samples_per_s": f"{throughput:.3f}",
            "RTF_1e-4": f"{rtf * 1e4:.3f}",
            "Latency_std_ms": f"{latency_std_ms:.3f}",
            "Params_M": "" if params_m is None else f"{params_m:.3f}",
            "MACs_G": "" if macs_g is None else f"{macs_g:.3f}",
        }
        rows.append(row)
        print(row, flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "efficiency_comparison.csv", rows)
    write_tex(output_dir / "efficiency_comparison.tex", rows)

    config = vars(args).copy()
    config.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "thop": thop.__version__,
            "batch_size": 1,
            "input_duration_seconds": duration_seconds,
            "flops_convention": "FLOPs = 2 * THOP MACs",
            "latency_statistic": "5% trimmed mean after warmup",
        }
    )
    (output_dir / "benchmark_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"[saved] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
