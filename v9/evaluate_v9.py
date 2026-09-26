"""Evaluate a PCD-Net V9 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from v7.evaluate_v7 import build_parser as build_v7_parser, evaluate, make_dataset
from v9.train_v9 import build_model


def main():
    args = build_v7_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    dataset = make_dataset(args, config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    routing = {"entropy": [], "effective": [], "maximum": [], "usage_sum": None, "samples": 0}

    def collect_routing(_module, _inputs, output):
        alpha = output[3]["routing_alpha"].detach().float().cpu()
        entropy = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(-1)
        routing["entropy"].extend(entropy.tolist())
        routing["effective"].extend(torch.exp(entropy).tolist())
        routing["maximum"].extend(alpha.max(-1).values.tolist())
        usage = alpha.sum(0)
        routing["usage_sum"] = usage if routing["usage_sum"] is None else routing["usage_sum"] + usage
        routing["samples"] += alpha.size(0)

    handle = model.register_forward_hook(collect_routing)
    try:
        metrics = evaluate(
            model, loader, device, args.adaptive_stop, args.st_window, args.zero_noise_condition
        )
    finally:
        handle.remove()
    count = max(routing["samples"], 1)
    usage = routing["usage_sum"] / count if routing["usage_sum"] is not None else torch.empty(0)
    metrics.update({
        "model_version": "v9",
        "routing": "soft" if config.get("use_routing", True) else "uniform_ablation",
        "num_prototypes": config.get("num_prototypes", 16),
        "routing_entropy": sum(routing["entropy"]) / max(len(routing["entropy"]), 1),
        "effective_prototypes": sum(routing["effective"]) / max(len(routing["effective"]), 1),
        "max_routing_weight": sum(routing["maximum"]) / max(len(routing["maximum"]), 1),
        "prototype_usage": usage.tolist(),
    })
    output = args.output or str(Path(args.checkpoint).with_name("evaluation_v9.json"))
    Path(output).write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, allow_nan=True), flush=True)
    print(f"[saved] {output}", flush=True)


if __name__ == "__main__":
    main()
