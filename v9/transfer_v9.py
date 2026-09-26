"""Selective target-domain adaptation for PCD-Net V9.

The environment strategy implements the paper scope exactly:
theta_env = {background encoder, prototype memory, routing parameters}.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from v7.train_v7 import make_loaders, train_epoch, validate
from v9.train_v9 import DEFAULT_CONFIG, _criterion, build_model


def configure_adaptation(model, strategy: str = "environment") -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if strategy == "context_encoder":
        modules = (model.noise_encoder,)
    elif strategy == "environment":
        modules = (model.noise_encoder, model.vq, model.router)
    elif strategy == "decoder_only":
        modules = (
            model.backbone.dec4,
            model.backbone.dec3,
            model.backbone.dec2,
            model.backbone.dec1,
            model.backbone.dec0,
            model.backbone.out_head,
        )
    elif strategy == "environment_decoder":
        modules = (
            model.noise_encoder,
            model.vq,
            model.router,
            model.backbone.dec4,
            model.backbone.dec3,
            model.backbone.dec2,
            model.backbone.dec1,
            model.backbone.dec0,
            model.backbone.out_head,
        )
    elif strategy == "full":
        modules = (model,)
    else:
        raise ValueError(f"unsupported adaptation strategy: {strategy}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in parameters)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    print(
        f"[adaptation] strategy={strategy} trainable={trainable / 1e6:.3f}M/"
        f"{total / 1e6:.3f}M ({100.0 * trainable / total:.2f}%)",
        flush=True,
    )
    for name in names:
        print(f"  [trainable] {name}", flush=True)

    def frozen_modules_in_eval(module, _inputs):
        for child in module.children():
            parameters = list(child.parameters())
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                child.eval()

    model.register_forward_pre_hook(frozen_modules_in_eval)
    return parameters


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_type", choices=["stead", "mining"], default="mining")
    parser.add_argument(
        "--strategy",
        choices=[
            "context_encoder", "environment", "decoder_only",
            "environment_decoder", "full",
        ],
        default="environment",
    )
    parser.add_argument("--event_h5", required=True)
    parser.add_argument("--event_csv", required=True)
    parser.add_argument("--noise_h5", required=True)
    parser.add_argument("--noise_csv", required=True)
    parser.add_argument("--context_source", choices=["pre_event", "paired_noise"], default="pre_event")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_samples", type=int, default=500)
    parser.add_argument("--val_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--no_amp", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = {**DEFAULT_CONFIG, **checkpoint.get("config", {})}
    cfg.update({
        "event_h5": args.event_h5, "event_csv": args.event_csv,
        "noise_h5": args.noise_h5, "noise_csv": args.noise_csv,
        "context_source": args.context_source, "epochs": args.epochs,
        "train_samples": args.train_samples, "val_samples": args.val_samples,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "split_prefix": f"v9_target_{args.dataset_type}_seed{args.seed}",
        "use_amp": cfg.get("use_amp", True) and not args.no_amp,
        "save_dir": args.save_dir,
    })
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    parameters = configure_adaptation(model, args.strategy)
    train_loader, val_loader = make_loaders(cfg, args.dataset_type)
    criterion = _criterion(cfg)
    optimizer = torch.optim.AdamW(parameters, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["use_amp"] and device.type == "cuda")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "adaptation_config.json").write_text(
        json.dumps({**cfg, "strategy": args.strategy}, indent=2, default=str), encoding="utf-8"
    )
    history, best_cc = [], -math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss, detail, skipped = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, cfg,
            scaler, epoch, save_dir / "recovery_adaptation_v9.pth",
        )
        metrics = validate(model, val_loader, criterion, device, cfg["st_window"])
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "skipped": skipped, **metrics}
        history.append(row)
        (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        state = {
            "model_state_dict": model.state_dict(), "epoch": epoch,
            "metrics": metrics, "config": cfg, "adaptation_strategy": args.strategy,
        }
        torch.save(state, save_dir / "last_adapted_v9.pth")
        if metrics["val_cc"] > best_cc:
            best_cc = metrics["val_cc"]
            torch.save(state, save_dir / "best_adapted_v9.pth")
        print(f"[{epoch}/{args.epochs}] loss={train_loss:.4f} CC={metrics['val_cc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
