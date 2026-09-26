"""Train PCD-Net V9 with dense soft prototype routing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from v7.loss_v7 import DenoiserLossV7
from v7.train_v7 import (
    DEFAULT_CONFIG as V7_DEFAULT_CONFIG,
    build_parser as build_v7_parser,
    configure_cuda_backend,
    make_loaders,
    set_seed,
    train_epoch,
    validate,
)
from v9.model_v9 import NoiseAwareDenoiserV9


DEFAULT_CONFIG = {
    **V7_DEFAULT_CONFIG,
    "event_h5": "E:/p_wave/data/chunk2.hdf5",
    "event_csv": "E:/p_wave/data/chunk2.csv",
    "noise_h5": "E:/p_wave/data/chunk1.hdf5",
    "noise_csv": "E:/p_wave/data/chunk1.csv",
    "context_source": "pre_event",
    "snr_range": (-15.0, 10.0),
    "snr_sampling": "db",
    "clean_prob": 0.0,
    "routing_temperature": 0.3,
    "routing_strength": 1.0,
    "learnable_routing": True,
    "use_routing": True,
    "prototype_interaction": "cross_attention",
    # V9 is a continuous prototype memory, not a vector-quantization model.
    # Disable the legacy VQ/Top-M objectives unless an experiment explicitly
    # opts into them.
    "alpha_vq_commit": 0.0,
    "alpha_vq_diversity": 0.0,
    "alpha_sparse": 0.0,
    "alpha_balance": 0.0,
    "epochs": 20,
    "batch_size": 4,
    # n_refine counts extra passes. Zero therefore means one restoration pass.
    "n_refine": 0,
    "use_residual_feedback": False,
    "adaptive_inference": False,
    "model_version": "v9",
}


def build_parser():
    parser = build_v7_parser()
    parser.description = "Train PCD-Net V9 (dense soft routing)"
    parser.add_argument("--routing_temperature", type=float)
    parser.add_argument("--routing_strength", type=float)
    parser.add_argument(
        "--prototype_interaction",
        choices=("cross_attention", "weighted_concat"),
    )
    parser.add_argument("--snr_min_db", type=float)
    parser.add_argument("--snr_max_db", type=float)
    parser.add_argument("--fixed_routing", action="store_true")
    parser.add_argument(
        "--no_routing",
        action="store_true",
        help="use uniform alpha; this is the routing ablation",
    )
    return parser


def build_model(cfg: dict) -> NoiseAwareDenoiserV9:
    return NoiseAwareDenoiserV9(
        in_ch=3,
        z_dim=cfg["z_dim"],
        cond_len=cfg["cond_len"],
        num_prototypes=cfg["num_prototypes"],
        num_heads=cfg["num_heads"],
        n_refine=cfg["n_refine"],
        base_ch=cfg["base_ch"],
        routing_temperature=cfg["routing_temperature"],
        routing_strength=cfg["routing_strength"],
        learnable_routing=cfg["learnable_routing"],
        use_routing=cfg["use_routing"],
        use_prototypes=cfg["use_prototypes"],
        use_quality_head=cfg["use_quality_head"],
        use_residual_feedback=cfg["use_residual_feedback"],
        adaptive_inference=cfg["adaptive_inference"],
        stop_threshold=cfg["stop_threshold"],
        min_refine_steps=cfg["min_refine_steps"],
        prototype_interaction=cfg.get("prototype_interaction", "cross_attention"),
    )


def _criterion(cfg):
    keys = (
        "alpha_mse", "alpha_freq", "alpha_grad", "alpha_detect",
        "alpha_vq_commit", "alpha_vq_diversity", "alpha_sparse",
        "alpha_balance", "alpha_quality", "valid_weight", "bg_weight",
        "alpha_intermediate", "quality_temperature",
    )
    return DenoiserLossV7(**{key: cfg[key] for key in keys})


def main():
    args = build_parser().parse_args()
    cfg = dict(DEFAULT_CONFIG)
    cfg["seed"] = args.seed
    for key in (
        "epochs", "batch_size", "lr", "num_workers", "cond_len",
        "num_prototypes", "n_refine", "base_ch", "quality_temperature",
        "alpha_sparse", "alpha_balance", "alpha_quality", "alpha_intermediate",
        "stop_threshold", "min_refine_steps", "print_every", "checkpoint_every",
        "event_h5", "event_csv", "noise_h5", "noise_csv", "raw_h5", "raw_csv",
        "val_batch_size", "val_num_workers", "st_window", "train_samples",
        "val_samples", "context_source", "routing_temperature", "routing_strength",
        "prototype_interaction",
    ):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    cfg["learnable_routing"] = not args.fixed_routing
    if args.snr_min_db is not None or args.snr_max_db is not None:
        low = cfg["snr_range"][0] if args.snr_min_db is None else args.snr_min_db
        high = cfg["snr_range"][1] if args.snr_max_db is None else args.snr_max_db
        if low >= high:
            raise ValueError("snr_min_db must be smaller than snr_max_db")
        cfg["snr_range"] = (float(low), float(high))
        cfg["snr_sampling"] = "db"
    cfg["use_routing"] = not args.no_routing
    cfg["use_prototypes"] = not args.no_prototypes
    cfg["use_quality_head"] = not args.no_quality_head
    # Feedback and adaptive stopping only have meaning when extra refinement
    # passes exist. The V9 default is the paper-aligned single-pass model.
    cfg["use_residual_feedback"] = (
        cfg["n_refine"] > 0 and not args.no_residual_feedback
    )
    cfg["adaptive_inference"] = (
        cfg["n_refine"] > 0 and not args.no_adaptive_inference
    )
    cfg["zero_noise_condition"] = args.zero_noise_condition
    cfg["use_amp"] = cfg["use_amp"] and not args.no_amp
    if args.amp_dtype is not None:
        cfg["amp_dtype"] = args.amp_dtype
    cfg["cuda_safe"] = args.cuda_safe
    cfg["pin_memory"] = not args.no_pin_memory
    cfg["save_dir"] = args.save_dir or f"v9/checkpoints_v9_seed{args.seed}"

    configure_cuda_backend(args.cuda_safe)
    set_seed(cfg["seed"])
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    train_loader, val_loader = make_loaders(cfg, args.dataset_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    if args.init_from and args.resume_from:
        raise ValueError("use only one of --init_from and --resume_from")
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location=device)
        missing, unexpected = model.load_compatible_state_dict(
            checkpoint.get("model_state_dict", checkpoint)
        )
        print(f"[init] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    criterion = _criterion(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg["epochs"], 1))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["use_amp"] and device.type == "cuda")
    history_path = save_dir / "history.json"
    history, best_cc, start_epoch = [], -math.inf, 1
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler.is_enabled() and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        if history:
            best_cc = max(float(row.get("val_cc", -math.inf)) for row in history)

    print(
        f"[model] V9 soft-routing={cfg['use_routing']} K={cfg['num_prototypes']} "
        f"tau={cfg['routing_temperature']} lambda={cfg['routing_strength']} device={device}",
        flush=True,
    )
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        train_loss, detail, skipped = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, cfg,
            scaler, epoch, save_dir / "recovery_model_v9.pth",
        )
        metrics = validate(
            model, val_loader, criterion, device, cfg["st_window"],
            zero_noise_condition=cfg.get("zero_noise_condition", False),
        )
        scheduler.step()
        row = {
            "epoch": epoch, "train_loss": train_loss, "skipped": skipped,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in detail.items()}, **metrics,
        }
        history.append(row)
        history_path.write_text(json.dumps(history, indent=2, allow_nan=True), encoding="utf-8")
        print(
            f"[{epoch:02d}/{cfg['epochs']}] loss={train_loss:.4f} "
            f"gain={metrics['val_gain']:+.2f} CC={metrics['val_cc']:.4f}",
            flush=True,
        )
        state = {
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch, "epoch_complete": True, "metrics": metrics, "config": cfg,
        }
        torch.save(state, save_dir / "last_model_v9.pth")
        if metrics["val_cc"] > best_cc:
            best_cc = metrics["val_cc"]
            torch.save(state, save_dir / "best_model_v9.pth")


if __name__ == "__main__":
    main()
