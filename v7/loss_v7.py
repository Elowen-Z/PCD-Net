"""Losses for the paper-aligned PCD-Net V7."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoiserLossV7(nn.Module):
    def __init__(
        self,
        alpha_mse: float = 1.0,
        alpha_freq: float = 0.20,
        alpha_grad: float = 0.20,
        alpha_detect: float = 1.0,
        alpha_vq_commit: float = 0.25,
        alpha_vq_diversity: float = 0.30,
        alpha_sparse: float = 0.05,
        alpha_balance: float = 0.02,
        alpha_quality: float = 0.20,
        alpha_intermediate: float = 0.10,
        valid_weight: float = 3.0,
        bg_weight: float = 0.3,
        quality_temperature: float = 1.0,
    ):
        super().__init__()
        self.a_mse = alpha_mse
        self.a_freq = alpha_freq
        self.a_grad = alpha_grad
        self.a_det = alpha_detect
        self.a_vqc = alpha_vq_commit
        self.a_vqd = alpha_vq_diversity
        self.a_sparse = alpha_sparse
        self.a_balance = alpha_balance
        self.a_quality = alpha_quality
        self.a_intermediate = alpha_intermediate
        self.w_valid = valid_weight
        self.w_bg = bg_weight
        self.quality_temperature = quality_temperature

    @staticmethod
    def _weighted_mse(pred, target, weight):
        w = weight.unsqueeze(1)
        return (((pred - target) ** 2) * w).sum() / (
            w.sum() * pred.size(1) + 1e-8
        )

    @staticmethod
    def _freq_mse(pred, target):
        return F.mse_loss(
            torch.fft.rfft(pred, dim=-1).abs(),
            torch.fft.rfft(target, dim=-1).abs(),
        )

    @staticmethod
    def _grad_mse(pred, target):
        return F.mse_loss(
            pred[..., 1:] - pred[..., :-1],
            target[..., 1:] - target[..., :-1],
        )

    @staticmethod
    def _entropy(probs):
        probs = probs.clamp_min(1e-8)
        return -(probs * probs.log()).sum(dim=-1)

    def _quality_target(self, pred, target, valid_mask):
        """Map detached normalized reconstruction error to [0,1]."""
        w = valid_mask.unsqueeze(1)
        err = (((pred.detach() - target) ** 2) * w).sum(dim=(1, 2))
        power = ((target ** 2) * w).sum(dim=(1, 2)).clamp_min(1e-8)
        nmse = err / power
        return torch.exp(-nmse / self.quality_temperature).clamp(0.0, 1.0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        has_target: torch.Tensor,
        det_mask: Optional[torch.Tensor] = None,
        quality: Optional[torch.Tensor] = None,
        quality_logits: Optional[torch.Tensor] = None,
        prototype_probs: Optional[torch.Tensor] = None,
        sparse_probs: Optional[torch.Tensor] = None,
        vq_commit: Optional[torch.Tensor] = None,
        vq_diversity: Optional[torch.Tensor] = None,
        refine_history=None,
        quality_logits_history=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = pred.new_zeros(())
        detail: Dict[str, float] = {}

        # STEAD/mining supervised runs have has_target=1 for every sample.
        # Avoid a CUDA boolean reduction/indexing path here; on some Windows
        # CUDA builds asynchronous kernel faults surface at `sup.any()`.
        all_supervised = bool(torch.all(has_target.detach().cpu() > 0.5))
        has_supervised = all_supervised or bool(
            torch.any(has_target.detach().cpu() > 0.5)
        )
        sup = None if all_supervised else has_target.bool()

        if has_supervised:
            if all_supervised:
                p = pred
                t = target
                vm = valid_mask.float()
            else:
                p = pred[sup]
                t = target[sup]
                vm = valid_mask[sup].float()
            weight = vm * self.w_valid + (1.0 - vm) * self.w_bg

            l_mse = self._weighted_mse(p, t, weight)
            l_freq = self._freq_mse(p, t)
            l_grad = self._grad_mse(p, t)
            loss = (
                loss
                + self.a_mse * l_mse
                + self.a_freq * l_freq
                + self.a_grad * l_grad
            )
            detail.update(
                mse=float(l_mse.detach()),
                freq=float(l_freq.detach()),
                grad=float(l_grad.detach()),
            )

            if self.a_intermediate > 0 and refine_history:
                intermediate = refine_history[:-1]
                if intermediate:
                    step_losses = [
                        self._weighted_mse(
                            step_pred if all_supervised else step_pred[sup],
                            t,
                            weight,
                        )
                        for step_pred in intermediate
                    ]
                    l_intermediate = torch.stack(step_losses).mean()
                    loss = loss + self.a_intermediate * l_intermediate
                    detail["intermediate"] = float(l_intermediate.detach())

            if self.a_quality > 0 and quality_logits_history:
                quality_losses = []
                quality_targets = []
                for step_pred, step_logits in zip(
                    refine_history, quality_logits_history
                ):
                    step_pred_sup = (
                        step_pred if all_supervised else step_pred[sup]
                    )
                    step_logits_sup = (
                        step_logits if all_supervised else step_logits[sup]
                    )
                    q_target = self._quality_target(step_pred_sup, t, vm)
                    q_logits = step_logits_sup.squeeze(-1).float()
                    quality_losses.append(
                        F.binary_cross_entropy_with_logits(
                            q_logits, q_target.float()
                        )
                    )
                    quality_targets.append(q_target.mean())
                l_quality = torch.stack(quality_losses).mean()
                loss = loss + self.a_quality * l_quality
                detail["quality"] = float(l_quality.detach())
                detail["quality_target"] = float(
                    torch.stack(quality_targets).mean().detach()
                )
            elif self.a_quality > 0 and (
                quality_logits is not None or quality is not None
            ):
                q_target = self._quality_target(p, t, vm)
                if quality_logits is not None:
                    q_logits_tensor = (
                        quality_logits
                        if all_supervised
                        else quality_logits[sup]
                    )
                    q_logits = q_logits_tensor.squeeze(-1).float()
                    l_quality = F.binary_cross_entropy_with_logits(
                        q_logits, q_target.float()
                    )
                else:
                    q_tensor = quality if all_supervised else quality[sup]
                    q = q_tensor.squeeze(-1).float().clamp(
                        1e-6, 1.0 - 1e-6
                    )
                    with torch.cuda.amp.autocast(enabled=False):
                        l_quality = F.binary_cross_entropy(
                            q, q_target.float()
                        )
                loss = loss + self.a_quality * l_quality
                detail["quality"] = float(l_quality.detach())
                detail["quality_target"] = float(q_target.mean().detach())

        if self.a_det > 0 and det_mask is not None:
            d = det_mask.squeeze(1).float().clamp(1e-6, 1.0 - 1e-6)
            # UNetBackboneV6 exposes probabilities rather than pre-sigmoid
            # logits, so this BCE must run outside CUDA autocast.
            with torch.cuda.amp.autocast(enabled=False):
                l_det = F.binary_cross_entropy(d, valid_mask.float())
            loss = loss + self.a_det * l_det
            detail["detect_bce"] = float(l_det.detach())

        dense_probs = prototype_probs
        if dense_probs is not None and dense_probs.size(1) > 1:
            if self.a_sparse > 0:
                l_sparse = self._entropy(dense_probs).mean()
                loss = loss + self.a_sparse * l_sparse
                detail["sparse_entropy"] = float(l_sparse.detach())

            if self.a_balance > 0:
                mean_usage = dense_probs.mean(dim=0)
                uniform = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
                l_balance = F.kl_div(
                    mean_usage.clamp_min(1e-8).log(),
                    uniform,
                    reduction="sum",
                )
                loss = loss + self.a_balance * l_balance
                detail["prototype_balance"] = float(l_balance.detach())

        if sparse_probs is not None:
            detail["selected_entropy"] = float(
                self._entropy(sparse_probs).mean().detach()
            )

        if self.a_vqc > 0 and vq_commit is not None:
            loss = loss + self.a_vqc * vq_commit
            detail["vq_commit"] = float(vq_commit.detach())
        if self.a_vqd > 0 and vq_diversity is not None:
            loss = loss + self.a_vqd * vq_diversity
            detail["vq_diversity"] = float(vq_diversity.detach())

        detail["total"] = float(loss.detach())
        return loss, detail
