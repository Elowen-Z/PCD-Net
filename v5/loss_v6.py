# v5/loss_v6.py
"""
DenoiserLossV6 — 配合 NoiseAwareDenoiserV6 的简化损失

去掉 V5 中拖累 CC 的设计:
  - 删除 NLL 项 (容易过度平滑)
  - 删除 identity / quality / consistency (V5 中权重很大但效果可疑)
  - 保留: MSE (主) + 频域 MSE + 一阶差分 MSE + 检测 BCE + VQ commit/diversity
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoiserLossV6(nn.Module):
    def __init__(
        self,
        alpha_mse:          float = 1.0,
        alpha_freq:         float = 0.20,
        alpha_grad:         float = 0.20,
        alpha_detect:       float = 1.0,
        alpha_vq_commit:    float = 0.25,
        alpha_vq_diversity: float = 0.30,
        valid_weight:       float = 3.0,
        bg_weight:          float = 0.3,
    ):
        super().__init__()
        self.a_mse    = alpha_mse
        self.a_freq   = alpha_freq
        self.a_grad   = alpha_grad
        self.a_det    = alpha_detect
        self.a_vqc    = alpha_vq_commit
        self.a_vqd    = alpha_vq_diversity
        self.w_valid  = valid_weight
        self.w_bg     = bg_weight

    @staticmethod
    def _weighted_mse(pred, target, weight):
        """weight: [B, T] (与 channels 广播). 返回标量"""
        # pred/target: [B,C,T]; weight: [B,T] -> [B,1,T]
        w = weight.unsqueeze(1)
        diff2 = (pred - target) ** 2
        return (diff2 * w).sum() / (w.sum() * pred.shape[1] + 1e-8)

    @staticmethod
    def _freq_mse(pred, target):
        # rFFT 幅度差 (对最后一维)
        P = torch.fft.rfft(pred,   dim=-1).abs()
        T = torch.fft.rfft(target, dim=-1).abs()
        return F.mse_loss(P, T)

    @staticmethod
    def _grad_mse(pred, target):
        dp = pred[..., 1:] - pred[..., :-1]
        dt = target[..., 1:] - target[..., :-1]
        return F.mse_loss(dp, dt)

    def forward(
        self,
        pred:           torch.Tensor,             # [B,C,T] 模型输出 clean
        target:         torch.Tensor,             # [B,C,T] 干净 ground truth
        valid_mask:     torch.Tensor,             # [B,T]   1=信号区
        has_target:     torch.Tensor,             # [B]     1=Part A
        det_mask:       Optional[torch.Tensor] = None,  # [B,1,T] 模型检测概率
        vq_commit:      Optional[torch.Tensor] = None,
        vq_diversity:   Optional[torch.Tensor] = None,
        # 兼容旧 trainer 传入的参数, 忽略
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = pred.device
        sup_mask = has_target.bool()              # [B]
        detail: Dict[str, float] = {}

        loss = pred.new_zeros(())

        # ---- 1) 加权 MSE (信号区权重高, 背景权重低) ----
        if sup_mask.any():
            p_s = pred[sup_mask]
            t_s = target[sup_mask]
            vm  = valid_mask[sup_mask]            # [b,T]
            w   = vm * self.w_valid + (1.0 - vm) * self.w_bg
            l_mse = self._weighted_mse(p_s, t_s, w)
            loss = loss + self.a_mse * l_mse
            detail["mse"] = float(l_mse.detach())

            # ---- 2) 频域 MSE (仅在信号区, 帮助高频) ----
            if self.a_freq > 0:
                l_freq = self._freq_mse(p_s, t_s)
                loss = loss + self.a_freq * l_freq
                detail["freq"] = float(l_freq.detach())

            # ---- 3) 一阶差分 MSE (波形锐度) ----
            if self.a_grad > 0:
                l_grad = self._grad_mse(p_s, t_s)
                loss = loss + self.a_grad * l_grad
                detail["grad"] = float(l_grad.detach())

        # ---- 4) 检测 BCE (AMP-safe: 在 fp32 下计算) ----
        if self.a_det > 0 and det_mask is not None:
            d = det_mask.squeeze(1).float().clamp(1e-6, 1 - 1e-6)  # [B,T]
            vm = valid_mask.float()
            with torch.cuda.amp.autocast(enabled=False):
                l_det = F.binary_cross_entropy(d, vm, reduction="mean")
            loss = loss + self.a_det * l_det
            detail["detect_bce"] = float(l_det.detach())

        # ---- 5) VQ commit / diversity ----
        if self.a_vqc > 0 and vq_commit is not None:
            loss = loss + self.a_vqc * vq_commit
            detail["vq_commit"] = float(vq_commit.detach())
        if self.a_vqd > 0 and vq_diversity is not None:
            loss = loss + self.a_vqd * vq_diversity
            detail["vq_diversity"] = float(vq_diversity.detach())

        detail["total"] = float(loss.detach())
        return loss, detail
