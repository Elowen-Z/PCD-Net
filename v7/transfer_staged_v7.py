"""Three-stage domain transfer for Feedback PCD-Net V7.

Stage 1 is the completed STEAD pre-training checkpoint.
Stage 2 aligns source and target bottleneck features with GRL while retaining
source/target reconstruction supervision.
Stage 3 selectively fine-tunes the target-domain denoiser while keeping the
background noise encoder and prototype codebook frozen by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from v3.dataset_v3 import STEADDatasetV3
from v5.dataset_mining import MiningDatasetV6
from v5.train_v6 import set_seed
from v7.loss_v7 import DenoiserLossV7
from v7.model_v7 import NoiseAwareDenoiserV7
from v7.train_v7 import validate


SOURCE = {
    "event_h5": "D:/X/p_wave/data/chunk2.hdf5",
    "event_csv": "D:/X/p_wave/data/chunk2.csv",
    "noise_h5": "D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv": "D:/X/p_wave/data/chunk1.csv",
    "dataset_type": "stead",
}

TARGETS = {
    "mining": {
        "event_h5": "D:/X/part2/data/LN_mining.hdf5",
        "event_csv": "D:/X/part2/data/LN_mining.csv",
        "noise_h5": "D:/X/p_wave/data/chunk1.hdf5",
        "noise_csv": "D:/X/p_wave/data/chunk1.csv",
        "dataset_type": "mining",
    },
    "nonnat": {
        "event_h5": "D:/X/p_wave/data/non_naturaldata.hdf5",
        "event_csv": "D:/X/p_wave/data/non_naturaldata.csv",
        "noise_h5": "D:/X/p_wave/data/chunk1.hdf5",
        "noise_csv": "D:/X/p_wave/data/chunk1.csv",
        "dataset_type": "stead",
    },
}


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coefficient: float):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coefficient * grad_output, None


def gradient_reverse(x: torch.Tensor, coefficient: float) -> torch.Tensor:
    return GradientReverse.apply(x, coefficient)


class DomainDiscriminator(nn.Module):
    def __init__(self, channels: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.mean(dim=-1)).squeeze(-1)


def split_csv(
    csv_path: str,
    output_dir: Path,
    prefix: str,
    val_fraction: float,
    seed: int,
    max_train_samples: int = 0,
) -> tuple[str, str]:
    frame = pd.read_csv(csv_path, low_memory=False)
    val = frame.sample(frac=val_fraction, random_state=seed)
    train = frame.drop(val.index)
    if max_train_samples > 0 and len(train) > max_train_samples:
        train = train.sample(n=max_train_samples, random_state=seed)
    train_path = output_dir / f"{prefix}_train.csv"
    val_path = output_dir / f"{prefix}_val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    print(f"[Split:{prefix}] train={len(train)} val={len(val)}")
    return str(train_path), str(val_path)


def sample_source_csv(
    csv_path: str,
    output_dir: Path,
    max_samples: int,
    seed: int,
) -> str:
    frame = pd.read_csv(csv_path, low_memory=False)
    if max_samples > 0 and len(frame) > max_samples:
        frame = frame.sample(n=max_samples, random_state=seed)
    output = output_dir / "source_alignment.csv"
    frame.to_csv(output, index=False)
    print(f"[Source alignment] samples={len(frame)}")
    return str(output)


def make_dataset(config: dict, csv_path: str, seed: int, evaluation: bool):
    dataset_class = (
        MiningDatasetV6
        if config["dataset_type"] == "mining"
        else STEADDatasetV3
    )
    kwargs = {
        "event_h5_path": config["event_h5"],
        "event_csv_path": csv_path,
        "noise_h5_path": config["noise_h5"],
        "noise_csv_path": config["noise_csv"],
        "signal_len": 6000,
        "cond_len": 400,
        "snr_range": (0.1, 20.0),
        "clean_prob": 0.0 if evaluation else 0.1,
        "part_b_ratio": 0.0,
        "seed": seed,
    }
    if config["dataset_type"] == "mining":
        kwargs["eval_mode"] = evaluation
    return dataset_class(**kwargs)


def make_loader(
    dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    drop_last: bool,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def infinite(loader):
    while True:
        yield from loader


def move_batch(
    batch: dict, device: torch.device, non_blocking: bool = False
) -> dict:
    return {
        key: batch[key].to(device, non_blocking=non_blocking)
        for key in ("x", "y_clean", "z_cond", "valid_mask", "has_target")
    }


def task_loss(
    criterion: DenoiserLossV7,
    output,
    batch: dict,
) -> tuple[torch.Tensor, dict]:
    clean, quality, _, aux = output
    return criterion(
        pred=clean,
        target=batch["y_clean"],
        valid_mask=batch["valid_mask"],
        has_target=batch["has_target"],
        det_mask=aux.get("det_mask"),
        quality=quality,
        quality_logits=aux.get("quality_logits"),
        prototype_probs=aux.get("prototype_probs"),
        sparse_probs=aux.get("sparse_probs"),
        vq_commit=aux.get("vq_commit"),
        vq_diversity=aux.get("vq_diversity"),
        refine_history=aux.get("refine_history"),
        quality_logits_history=aux.get("quality_logits_history"),
    )


def freeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze(modules) -> None:
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = True


def configure_stage(
    model: NoiseAwareDenoiserV7,
    stage: int,
    freeze_strategy: str = "signal_backbone",
) -> list:
    freeze_all(model)
    backbone = model.backbone

    if stage == 2:
        modules = [
            model.feedback_updater,
            backbone.bn_conv,
            backbone.bn_xattn,
            backbone.dec4,
            backbone.dec3,
            backbone.dec2,
            backbone.dec1,
            backbone.dec0,
            backbone.out_head,
            backbone.mask_head,
            model.quality_head,
        ]
        description = "feedback updater + bottleneck + decoder + quality head"
    elif stage == 3:
        if freeze_strategy in ("decoder_only", "signal_decoder"):
            modules = [
                backbone.dec4,
                backbone.dec3,
                backbone.dec2,
                backbone.dec1,
                backbone.dec0,
                backbone.out_head,
                backbone.mask_head,
                model.quality_head,
            ]
            description = "signal decoder + output heads + quality head"
        elif freeze_strategy == "noise_encoder":
            modules = [
                model.noise_encoder,
                model.vq,
            ]
            description = "noise encoder + prototype codebook only"
        elif freeze_strategy == "signal_encoder":
            modules = [
                backbone.enc1,
                backbone.enc2,
                backbone.enc3,
                backbone.enc4,
                backbone.enc5,
                backbone.ref1,
                backbone.ref2,
                backbone.ref3,
                backbone.ref4,
                backbone.ref5,
                backbone.bn_conv,
                backbone.bn_xattn,
            ]
            description = "signal encoder + bottleneck conditioning only"
        elif freeze_strategy == "prototype_feedback":
            modules = [
                model.noise_encoder,
                model.vq,
                model.feedback_updater,
                backbone.bn_xattn,
            ]
            description = (
                "noise prototype branch + residual-feedback updater + "
                "prototype-guided cross-attention"
            )
        elif freeze_strategy == "feedback_decoder":
            modules = [
                model.feedback_updater,
                backbone.bn_conv,
                backbone.bn_xattn,
                backbone.dec4,
                backbone.dec3,
                backbone.dec2,
                backbone.dec1,
                backbone.dec0,
                backbone.out_head,
                backbone.mask_head,
                model.quality_head,
            ]
            description = "feedback updater + bottleneck + decoder"
        elif freeze_strategy in ("signal_backbone", "pcd_adaptation"):
            modules = [
                model.feedback_updater,
                backbone,
                model.quality_head,
            ]
            description = (
                "whole signal backbone + feedback updater + quality head; "
                "noise encoder and codebook frozen"
            )
        elif freeze_strategy == "full":
            modules = [model]
            description = "all model parameters"
        else:
            raise ValueError(
                f"unsupported freeze strategy: {freeze_strategy}"
            )
    else:
        raise ValueError(f"unsupported stage: {stage}")

    unfreeze(modules)
    parameters = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in parameters)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[Stage {stage}] trainable={trainable / 1e6:.3f}M/"
        f"{total / 1e6:.3f}M ({100.0 * trainable / total:.1f}%)"
    )
    print(f"[Stage {stage}] {description}")
    return parameters


def build_model(args, device: torch.device) -> NoiseAwareDenoiserV7:
    variant = args.variant
    model = NoiseAwareDenoiserV7(
        in_ch=3,
        z_dim=128,
        cond_len=400,
        num_prototypes=16,
        top_m=args.top_m,
        num_heads=4,
        n_refine=args.n_refine,
        base_ch=32,
        vq_temperature=0.3,
        use_prototypes=variant != "no_prototypes",
        use_sparse_selection=variant != "no_sparse",
        use_cross_attn=variant != "no_cross_attn",
        use_quality_head=variant != "no_quality",
        use_residual_feedback=variant != "no_feedback",
        adaptive_inference=True,
        stop_threshold=args.stop_threshold,
        min_refine_steps=args.min_refine_steps,
    ).to(device)
    checkpoint = torch.load(args.source_ckpt, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_v6_state_dict(state)
    print(
        f"[Load] {args.source_ckpt} missing={len(missing)} "
        f"unexpected={len(unexpected)}"
    )
    return model


def make_criterion(args) -> DenoiserLossV7:
    return DenoiserLossV7(
        alpha_mse=1.0,
        alpha_freq=0.2,
        alpha_grad=0.2,
        alpha_detect=1.0,
        alpha_vq_commit=0.25,
        alpha_vq_diversity=0.3,
        alpha_sparse=0.05,
        alpha_balance=0.02,
        alpha_quality=0.0 if args.variant == "no_quality" else 0.2,
        alpha_intermediate=args.alpha_intermediate,
        valid_weight=3.0,
        bg_weight=0.3,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    discriminator: nn.Module | None,
    optimizer,
    stage: int,
    epoch: int,
    metrics: dict,
    args,
    extra_state: dict | None = None,
) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stage": stage,
        "epoch": epoch,
        "metrics": metrics,
        "config": vars(args),
    }
    if discriminator is not None:
        state["discriminator_state_dict"] = discriminator.state_dict()
    if extra_state:
        state.update(extra_state)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def train_stage2(
    model,
    discriminator,
    source_loader,
    target_loader,
    val_loader,
    criterion,
    device,
    args,
    save_dir,
    history,
    resume_checkpoint=None,
) -> tuple[float, Path]:
    print("\n" + "=" * 72)
    print("STAGE 2: adversarial source-target feature alignment")
    print("=" * 72)
    model_parameters = configure_stage(model, stage=2)
    optimizer = torch.optim.AdamW(
        [
            {"params": model_parameters, "lr": args.lr_stage2},
            {"params": discriminator.parameters(), "lr": args.lr_stage2},
        ],
        weight_decay=args.weight_decay,
    )
    domain_loss = nn.BCEWithLogitsLoss()
    source_iterator = infinite(source_loader)
    total_steps = max(1, args.stage2_epochs * len(target_loader))
    global_step = 0
    best_cc = -math.inf
    best_path = save_dir / "best_stage2_v7.pth"
    amp = args.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch = 1
    resume_batch = 0
    resume_sums = None
    resume_samples = 0
    progress_path = save_dir / "progress_stage2_v7.pth"

    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if "discriminator_state_dict" in checkpoint:
            discriminator.load_state_dict(
                checkpoint["discriminator_state_dict"], strict=True
            )
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        checkpoint_batch = int(checkpoint.get("batch_index", 0))
        if checkpoint_batch > 0:
            start_epoch = checkpoint_epoch
            resume_batch = min(checkpoint_batch, len(target_loader))
            resume_sums = checkpoint.get("partial_sums")
            resume_samples = int(checkpoint.get("partial_samples", 0))
            global_step = int(
                checkpoint.get(
                    "global_step",
                    (start_epoch - 1) * len(target_loader) + resume_batch,
                )
            )
        else:
            start_epoch = checkpoint_epoch + 1
            global_step = (start_epoch - 1) * len(target_loader)
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location="cpu")
            best_cc = float(
                best_checkpoint.get("metrics", {}).get("val_cc", -math.inf)
            )
        print(
            f"[Resume Stage 2] {resume_checkpoint}; "
            f"epoch={start_epoch}, next_batch={resume_batch + 1}, "
            f"best_cc={best_cc:.4f}"
        )

    feature_store = {}

    def capture_feature(_module, _inputs, output):
        feature_store["bottleneck"] = output

    hook = model.backbone.bn_xattn.register_forward_hook(capture_feature)
    try:
        for epoch in range(start_epoch, args.stage2_epochs + 1):
            model.train()
            discriminator.train()
            started = time.time()
            if epoch == start_epoch and resume_sums is not None:
                sums = {
                    key: float(resume_sums.get(key, 0.0))
                    for key in ("total", "target", "source", "domain")
                }
                samples = resume_samples
                epoch_resume_batch = resume_batch
            else:
                sums = {
                    "total": 0.0,
                    "target": 0.0,
                    "source": 0.0,
                    "domain": 0.0,
                }
                samples = 0
                epoch_resume_batch = 0

            for batch_index, target_raw in enumerate(target_loader, start=1):
                if batch_index <= epoch_resume_batch:
                    continue
                global_step += 1
                progress = global_step / total_steps
                grl_coefficient = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0

                target = move_batch(target_raw, device)
                source = move_batch(next(source_iterator), device)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(
                    enabled=amp, dtype=torch.float16
                ):
                    target_output = model(
                        target["x"], target["z_cond"], adaptive_stop=False
                    )
                    target_feature = feature_store["bottleneck"]
                    target_task, _ = task_loss(
                        criterion, target_output, target
                    )

                    source_output = model(
                        source["x"], source["z_cond"], adaptive_stop=False
                    )
                    source_feature = feature_store["bottleneck"]
                    source_task, _ = task_loss(
                        criterion, source_output, source
                    )

                    target_logits = discriminator(
                        gradient_reverse(target_feature, grl_coefficient)
                    )
                    source_logits = discriminator(
                        gradient_reverse(source_feature, grl_coefficient)
                    )
                    alignment = 0.5 * (
                        domain_loss(
                            target_logits, torch.ones_like(target_logits)
                        )
                        + domain_loss(
                            source_logits, torch.zeros_like(source_logits)
                        )
                    )
                    loss = (
                        target_task
                        + args.alpha_source * source_task
                        + args.alpha_domain * alignment
                    )
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model_parameters) + list(discriminator.parameters()),
                    args.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()

                batch_size = target["x"].size(0)
                samples += batch_size
                sums["total"] += float(loss.detach()) * batch_size
                sums["target"] += float(target_task.detach()) * batch_size
                sums["source"] += float(source_task.detach()) * batch_size
                sums["domain"] += float(alignment.detach()) * batch_size

                if args.print_every and batch_index % args.print_every == 0:
                    print(
                        f"  [S2 ep{epoch:02d} batch{batch_index:04d}/"
                        f"{len(target_loader):04d}] loss={float(loss):.4f} "
                        f"domain={float(alignment):.4f} grl={grl_coefficient:.3f}"
                    )

                if (
                    args.checkpoint_every_batches > 0
                    and batch_index % args.checkpoint_every_batches == 0
                ):
                    save_checkpoint(
                        progress_path,
                        model,
                        discriminator,
                        optimizer,
                        2,
                        epoch,
                        {},
                        args,
                        extra_state={
                            "batch_index": batch_index,
                            "partial_sums": sums,
                            "partial_samples": samples,
                            "global_step": global_step,
                        },
                    )
                    print(
                        f"  [CHECKPOINT] Stage 2 epoch {epoch}, "
                        f"batch {batch_index}/{len(target_loader)}"
                    )

            save_checkpoint(
                progress_path,
                model,
                discriminator,
                optimizer,
                2,
                epoch,
                {},
                args,
                extra_state={
                    "batch_index": len(target_loader),
                    "partial_sums": sums,
                    "partial_samples": samples,
                    "global_step": global_step,
                },
            )
            metrics = validate(model, val_loader, criterion, device)
            row = {
                "stage": 2,
                "epoch": epoch,
                **{key: value / max(samples, 1) for key, value in sums.items()},
                **metrics,
                "seconds": time.time() - started,
            }
            history.append(row)
            (save_dir / "transfer_history.json").write_text(
                json.dumps(history, indent=2, allow_nan=True),
                encoding="utf-8",
            )
            print(
                f"[S2 {epoch:02d}/{args.stage2_epochs}] "
                f"loss={row['total']:.4f} domain={row['domain']:.4f} "
                f"gain={metrics['val_gain']:+.2f} "
                f"CC={metrics['val_cc']:.4f} "
                f"A-CC={metrics['adaptive_cc']:.4f} "
                f"({row['seconds']:.0f}s)"
            )
            save_checkpoint(
                save_dir / "last_stage2_v7.pth",
                model,
                discriminator,
                optimizer,
                2,
                epoch,
                metrics,
                args,
            )
            if metrics["val_cc"] > best_cc:
                best_cc = metrics["val_cc"]
                save_checkpoint(
                    best_path,
                    model,
                    discriminator,
                    optimizer,
                    2,
                    epoch,
                    metrics,
                    args,
                )
                print(f"  [SAVE] Stage 2 best CC={best_cc:.4f}")
            if progress_path.exists():
                progress_path.unlink()
            resume_batch = 0
            resume_sums = None
            resume_samples = 0
    finally:
        hook.remove()
    return best_cc, best_path


def train_stage3(
    model,
    target_loader,
    val_loader,
    criterion,
    device,
    args,
    save_dir,
    history,
    resume_checkpoint=None,
) -> tuple[float, Path]:
    print("\n" + "=" * 72)
    print("STAGE 3: selective target-domain fine-tuning")
    print("=" * 72)
    parameters = configure_stage(
        model, stage=3, freeze_strategy=args.freeze_strategy
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr_stage3, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.stage3_epochs, 1)
    )
    best_cc = -math.inf
    best_path = save_dir / "best_transfer_v7.pth"
    amp = args.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch = 1

    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        scheduler.last_epoch = start_epoch - 1
        scheduler._step_count = start_epoch
        scheduler._last_lr = [
            group["lr"] for group in optimizer.param_groups
        ]
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location="cpu")
            best_cc = float(
                best_checkpoint.get("metrics", {}).get(
                    "val_cc", -math.inf
                )
            )
        print(
            f"[Stage 3 resume] checkpoint={resume_checkpoint} "
            f"next_epoch={start_epoch} best_cc={best_cc:.4f}"
        )

    for epoch in range(start_epoch, args.stage3_epochs + 1):
        model.train()
        started = time.time()
        loss_sum = 0.0
        samples = 0

        for batch_index, raw_batch in enumerate(target_loader, start=1):
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=amp, dtype=torch.float16
            ):
                output = model(
                    batch["x"], batch["z_cond"], adaptive_stop=False
                )
                loss, _ = task_loss(criterion, output, batch)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_size = batch["x"].size(0)
            samples += batch_size
            loss_sum += float(loss.detach()) * batch_size
            if args.print_every and batch_index % args.print_every == 0:
                print(
                    f"  [S3 ep{epoch:02d} batch{batch_index:04d}/"
                    f"{len(target_loader):04d}] loss={float(loss):.4f}"
                )

        scheduler.step()
        metrics = validate(model, val_loader, criterion, device)
        row = {
            "stage": 3,
            "epoch": epoch,
            "total": loss_sum / max(samples, 1),
            **metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(row)
        (save_dir / "transfer_history.json").write_text(
            json.dumps(history, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        print(
            f"[S3 {epoch:02d}/{args.stage3_epochs}] "
            f"loss={row['total']:.4f} gain={metrics['val_gain']:+.2f} "
            f"CC={metrics['val_cc']:.4f} "
            f"A-CC={metrics['adaptive_cc']:.4f} "
            f"steps={metrics['effective_steps']:.2f} "
            f"({row['seconds']:.0f}s)"
        )
        save_checkpoint(
            save_dir / "last_transfer_v7.pth",
            model,
            None,
            optimizer,
            3,
            epoch,
            metrics,
            args,
        )
        if metrics["val_cc"] > best_cc:
            best_cc = metrics["val_cc"]
            save_checkpoint(
                best_path,
                model,
                None,
                optimizer,
                3,
                epoch,
                metrics,
                args,
            )
            print(f"  [SAVE] Stage 3 best CC={best_cc:.4f}")
    return best_cc, best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=sorted(TARGETS), default="mining")
    parser.add_argument(
        "--source_ckpt",
        default="v7/checkpoints_feedback_stead_seed0/best_model_v7.pth",
    )
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage3_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr_stage2", type=float, default=5e-5)
    parser.add_argument("--lr_stage3", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha_domain", type=float, default=0.2)
    parser.add_argument("--alpha_source", type=float, default=0.2)
    parser.add_argument("--alpha_intermediate", type=float, default=0.1)
    parser.add_argument("--max_source", type=int, default=8000)
    parser.add_argument(
        "--max_target_train",
        type=int,
        default=0,
        help=(
            "Limit target-domain training samples after the validation split. "
            "0 means using all target training samples."
        ),
    )
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--top_m", type=int, default=4)
    parser.add_argument("--n_refine", type=int, default=3)
    parser.add_argument("--stop_threshold", type=float, default=0.95)
    parser.add_argument("--min_refine_steps", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--checkpoint_every_batches", type=int, default=500)
    parser.add_argument(
        "--variant",
        choices=[
            "full",
            "no_feedback",
            "no_sparse",
            "no_prototypes",
            "no_cross_attn",
            "no_quality",
        ],
        default="full",
    )
    parser.add_argument(
        "--freeze_strategy",
        choices=[
            "decoder_only",
            "noise_encoder",
            "signal_encoder",
            "signal_decoder",
            "prototype_feedback",
            "pcd_adaptation",
            "feedback_decoder",
            "signal_backbone",
            "full",
        ],
        default="signal_backbone",
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--cuda_safe", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_stage2", action="store_true")
    parser.add_argument("--stage2_ckpt", default=None)
    parser.add_argument("--resume_stage2", default=None)
    parser.add_argument("--resume_stage3", default=None)
    parser.add_argument("--target_event_h5", default=None)
    parser.add_argument("--target_event_csv", default=None)
    parser.add_argument("--target_noise_h5", default=None)
    parser.add_argument("--target_noise_csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.use_amp = not args.no_amp
    if args.cuda_safe and torch.cuda.is_available():
        args.use_amp = False
        args.batch_size = 1
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        print(
            "[CUDA safe] FP32, batch_size=1, cuDNN, pinned memory, "
            "and fused SDP kernels disabled."
        )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(
        args.save_dir or f"v7/checkpoints_{args.target}_transfer_v7"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "transfer_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print(
        f"[Info] device={device} target={args.target} "
        f"variant={args.variant} freeze={args.freeze_strategy} "
        f"save={save_dir}"
    )

    target_config = dict(TARGETS[args.target])
    for key, value in (
        ("event_h5", args.target_event_h5),
        ("event_csv", args.target_event_csv),
        ("noise_h5", args.target_noise_h5),
        ("noise_csv", args.target_noise_csv),
    ):
        if value:
            target_config[key] = value

    target_train_csv, target_val_csv = split_csv(
        target_config["event_csv"],
        save_dir,
        args.target,
        args.val_fraction,
        args.seed,
        args.max_target_train,
    )
    source_csv = sample_source_csv(
        SOURCE["event_csv"], save_dir, args.max_source, args.seed
    )

    target_train = make_dataset(
        target_config, target_train_csv, args.seed, evaluation=False
    )
    target_val = make_dataset(
        target_config, target_val_csv, args.seed, evaluation=True
    )
    source_train = make_dataset(
        SOURCE, source_csv, args.seed, evaluation=False
    )
    target_loader = make_loader(
        target_train,
        args.batch_size,
        args.num_workers,
        True,
        True,
        pin_memory=not args.cuda_safe,
    )
    val_loader = make_loader(
        target_val,
        args.batch_size,
        args.num_workers,
        False,
        False,
        pin_memory=not args.cuda_safe,
    )
    source_loader = make_loader(
        source_train,
        args.batch_size,
        args.num_workers,
        True,
        True,
        pin_memory=not args.cuda_safe,
    )

    model = build_model(args, device)
    discriminator = DomainDiscriminator(channels=32 * 8).to(device)
    criterion = make_criterion(args)
    history_path = save_dir / "transfer_history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.exists()
        else []
    )

    if args.skip_stage2:
        stage2_path = Path(
            args.stage2_ckpt or save_dir / "best_stage2_v7.pth"
        )
        checkpoint = torch.load(stage2_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"[Load] skipped Stage 2; loaded {stage2_path}")
    else:
        _, stage2_path = train_stage2(
            model,
            discriminator,
            source_loader,
            target_loader,
            val_loader,
            criterion,
            device,
            args,
            save_dir,
            history,
            resume_checkpoint=args.resume_stage2,
        )
        checkpoint = torch.load(stage2_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"[Load] Stage 2 best checkpoint: {stage2_path}")

    best_cc, best_path = train_stage3(
        model,
        target_loader,
        val_loader,
        criterion,
        device,
        args,
        save_dir,
        history,
        resume_checkpoint=args.resume_stage3,
    )
    print("\n[DONE]")
    print(f"Best target CC: {best_cc:.4f}")
    print(f"Best checkpoint: {best_path}")
    best_checkpoint = torch.load(best_path, map_location="cpu")
    summary = {
        "target": args.target,
        "variant": args.variant,
        "freeze_strategy": args.freeze_strategy,
        "max_target_train": args.max_target_train,
        "target_train_samples": len(target_train),
        "target_val_samples": len(target_val),
        "best_cc": best_cc,
        "best_checkpoint": str(best_path),
        **best_checkpoint.get("metrics", {}),
    }
    (save_dir / "transfer_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
