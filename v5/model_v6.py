# v5/model_v6.py
"""
NoiseAwareDenoiserV6 — 针对 CC 偏低问题重新设计的去噪主干

相对 V5 的关键改动（解决 CC < 0.5 / DPRNN 优势的根因）:

  1. **去掉输出端 mask 乘法**: V5 用 `clean = cur * mask` 会在边缘削幅、
     mask 错位即整段相关性归零；V6 直接输出 `cur`，mask 只作为辅助检测头。
  2. **去掉 NLL / log_var**: V5 的异方差头允许"对难样本提高 σ"逃避惩罚，
     导致峰值被压平、波形相关性下降；V6 改纯 MSE，强制对每个样本承诺幅度。
  3. **加宽 UNet**: base_ch 16 → 32，所有层翻倍 (32/64/128/256/256)。
     约 5M 参数（vs V5 1.3M），匹配 DPRNN 量级能力。
  4. **精炼迭代次数**: n_refine 2 → 3。
  5. **保留 VQ noise prototypes + cross-attn**: 跨域泛化的核心机制不动。

输出与 V5 兼容: forward 仍返回 (clean, quality, z_noise, aux)
  - clean    : [B, C, T]  最终去噪结果（= cur，不乘 mask）
  - quality  : [B, 1]      由 mask 平均值粗略估计
  - z_noise  : [B, d]
  - aux      : {"det_mask":..., "cur_premask":..., "vq_commit":..., ...}
"""

from __future__ import annotations
import os, sys
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT   = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from v4.model_v4 import (
    ConvBlock1d, ConvBlockLarge,
    VQNoisePrototypes,
    NoiseEncoderV4,
    CrossAttnBottleneck,
)


# ============================================================
#  消融用: 拼接 + FFN 条件模块 (替代 Cross-Attention)
# ============================================================
class ConcatConditioning(nn.Module):
    """
    Ablation-2: 去掉 Cross-Attention, 改为简单拼接:
        H~ = FFN([H_s ; z_n])
    其中 z_n 为噪声向量 (对条件 token 取均值得到), 沿时间轴广播后
    与信号特征通道拼接, 经 1x1 卷积融合 + FFN。
    保留与 CrossAttnBottleneck 同构的残差/LayerNorm 结构以保证公平对比。
    """
    def __init__(self, c_dim: int, p_dim: int, ff_mult: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.z_proj = nn.Linear(p_dim, c_dim)
        self.fuse   = nn.Conv1d(c_dim * 2, c_dim, kernel_size=1)
        self.norm1  = nn.LayerNorm(c_dim)
        self.norm2  = nn.LayerNorm(c_dim)
        self.ff = nn.Sequential(
            nn.Linear(c_dim, c_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_dim * ff_mult, c_dim),
        )

    def forward(self, x: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]; protos: [B, M, d] -> z_n: [B, d]
        B, C, T = x.shape
        z = protos.mean(dim=1)                     # [B, d]
        z = self.z_proj(z)                         # [B, C]
        z_b = z.unsqueeze(-1).expand(-1, -1, T)    # [B, C, T]
        fused = self.fuse(torch.cat([x, z_b], dim=1))   # [B, C, T]
        tokens = x.permute(0, 2, 1)                # [B, T, C]
        tokens = self.norm1(tokens + fused.permute(0, 2, 1))
        tokens = self.norm2(tokens + self.ff(tokens))
        return tokens.permute(0, 2, 1)             # [B, C, T]


# ============================================================
#  V6 UNet 主干 (宽度翻倍, 单通道 μ_res 输出, 无 log_var)
# ============================================================
class UNetBackboneV6(nn.Module):
    """
    Encoder + cross-attn bottleneck + decoder
    out_head 仅输出 in_ch 通道 (μ_residual)；不再预测 log_var。
    mask_head 仍预测 [B,1,T] 的逐时刻信号概率（辅助检测/可视化用，
    不参与最终输出乘法）。
    """

    def __init__(self, in_ch: int = 3, p_dim: int = 128,
                 num_heads: int = 4, base_ch: int = 32,
                 use_cross_attn: bool = True,
                 use_quality_head: bool = True):
        super().__init__()
        self.in_ch   = in_ch
        self.base_ch = base_ch
        self.use_cross_attn   = use_cross_attn
        self.use_quality_head = use_quality_head
        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 8

        # ── Encoder ─────────────────────────────────────
        self.enc1 = ConvBlock1d(in_ch, c1, kernel=3, stride=2)
        self.enc2 = ConvBlock1d(c1,    c2, kernel=3, stride=2)
        self.enc3 = ConvBlock1d(c2,    c3, kernel=3, stride=2)
        self.enc4 = ConvBlock1d(c3,    c4, kernel=3, stride=2)
        self.enc5 = ConvBlock1d(c4,    c5, kernel=3, stride=2)

        self.ref1 = ConvBlockLarge(c1, c1, kernel=15)
        self.ref2 = ConvBlockLarge(c2, c2, kernel=15)
        self.ref3 = ConvBlockLarge(c3, c3, kernel=15)
        self.ref4 = ConvBlockLarge(c4, c4, kernel=7)
        self.ref5 = ConvBlockLarge(c5, c5, kernel=5)

        # ── Bottleneck 条件模块 (cross-attn 或 消融: concat+FFN) ────
        self.bn_conv  = ConvBlockLarge(c5, c5, kernel=5)
        if use_cross_attn:
            self.bn_xattn = CrossAttnBottleneck(
                c_dim=c5, p_dim=p_dim, num_heads=num_heads,
            )
        else:
            self.bn_xattn = ConcatConditioning(c_dim=c5, p_dim=p_dim)

        # ── Decoder ─────────────────────────────────────
        self.dec4 = nn.Sequential(
            ConvBlock1d(c5 + c4, c4, kernel=3),
            ConvBlockLarge(c4, c3, kernel=15),
        )
        self.dec3 = nn.Sequential(
            ConvBlock1d(c3 + c3, c3, kernel=3),
            ConvBlockLarge(c3, c2, kernel=15),
        )
        self.dec2 = nn.Sequential(
            ConvBlock1d(c2 + c2, c2, kernel=3),
            ConvBlockLarge(c2, c1, kernel=15),
        )
        self.dec1 = nn.Sequential(
            ConvBlock1d(c1 + c1, c1, kernel=3),
            ConvBlockLarge(c1, c1, kernel=15),
        )
        self.dec0 = ConvBlockLarge(c1, c1, kernel=7)

        # ── 输出头: 只输出 μ_residual (in_ch 通道, 无 log_var) ──
        self.out_head = nn.Conv1d(c1, in_ch, kernel_size=1)
        # ── 质量/检测头: aux 用, 不参与输出乘法 (Ablation-3 可删除) ──
        if use_quality_head:
            self.mask_head = nn.Sequential(
                nn.Conv1d(c1, c1, kernel_size=7, padding=3),
                nn.SiLU(),
                nn.Conv1d(c1, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.mask_head = None

    @staticmethod
    def _align(x, target):
        if x.shape[-1] != target.shape[-1]:
            x = F.interpolate(x, size=target.shape[-1], mode="nearest")
        return x

    @staticmethod
    def _up2(x):
        return x.repeat_interleave(2, dim=-1)

    def forward(self, x: torch.Tensor, protos: torch.Tensor):
        e1 = self.ref1(self.enc1(x))
        e2 = self.ref2(self.enc2(e1))
        e3 = self.ref3(self.enc3(e2))
        e4 = self.ref4(self.enc4(e3))
        e5 = self.ref5(self.enc5(e4))

        bn = self.bn_conv(e5)
        bn = self.bn_xattn(bn, protos)

        d4 = self.dec4(torch.cat([self._align(self._up2(bn), e4), e4], 1))
        d3 = self.dec3(torch.cat([self._align(self._up2(d4), e3), e3], 1))
        d2 = self.dec2(torch.cat([self._align(self._up2(d3), e2), e2], 1))
        d1 = self.dec1(torch.cat([self._align(self._up2(d2), e1), e1], 1))
        d0 = self.dec0(self._align(self._up2(d1), x))

        mu_res = self.out_head(d0)        # [B, in_ch, T]
        mask   = self.mask_head(d0) if self.mask_head is not None else None
        return mu_res, mask


# ============================================================
#  完整模型 V6 (IRR, 不乘 mask, 无 NLL)
# ============================================================
class NoiseAwareDenoiserV6(nn.Module):
    """
    Pipeline:
      1) noise_encoder(z_cond) → h
      2) VQ → (z_noise, probs)  ; prototypes = codebook × probs
      3) IRR: 共享权重迭代 (n_refine+1) 次预测残差
            cur ← cur + μ_res
      4) clean = cur          (不乘 mask, 让输出携带连续幅度)
      5) mask 仅作辅助检测头
    """

    def __init__(
        self,
        in_ch:           int   = 3,
        z_dim:           int   = 128,
        cond_len:        int   = 400,
        num_prototypes:  int   = 16,
        num_heads:       int   = 4,
        n_refine:        int   = 3,
        base_ch:         int   = 32,
        vq_temperature:  float = 0.3,
        use_prototypes:  bool  = True,
        use_cross_attn:  bool  = True,
        use_quality_head: bool = True,
    ):
        super().__init__()
        self.in_ch    = in_ch
        self.z_dim    = z_dim
        self.n_refine = n_refine
        self.use_prototypes   = use_prototypes
        self.use_cross_attn   = use_cross_attn
        self.use_quality_head = use_quality_head

        self.noise_encoder = NoiseEncoderV4(in_ch, dim=z_dim, cond_len=cond_len)
        self.vq = VQNoisePrototypes(
            num_prototypes=num_prototypes, dim=z_dim,
            temperature=vq_temperature,
        )
        self.backbone = UNetBackboneV6(
            in_ch=in_ch, p_dim=z_dim,
            num_heads=num_heads, base_ch=base_ch,
            use_cross_attn=use_cross_attn,
            use_quality_head=use_quality_head,
        )

        self.register_parameter(
            "tta_logits",
            nn.Parameter(torch.zeros(num_prototypes), requires_grad=False),
        )

    def _get_protos(self, probs: torch.Tensor) -> torch.Tensor:
        B = probs.size(0)
        E = self.vq.codebook
        bias = F.softmax(self.tta_logits, dim=-1)
        scale = probs * bias.unsqueeze(0)
        scale = scale / (scale.sum(-1, keepdim=True) + 1e-8)
        protos = E.unsqueeze(0).expand(B, -1, -1)
        protos = protos * scale.unsqueeze(-1)
        return protos

    def forward(
        self,
        x:      torch.Tensor,
        z_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        h = self.noise_encoder(z_cond)

        if self.use_prototypes:
            z_noise, probs, vq_aux = self.vq(h)
            protos = self._get_protos(probs)         # [B, K, d]
            vq_commit    = vq_aux["vq_commit"]
            vq_diversity = vq_aux["vq_diversity"]
            hard_idx     = vq_aux["hard_idx"]
        else:
            # Ablation-1: 去掉原型字典, 只保留连续噪声向量 h
            z_noise = h
            probs   = torch.ones(h.size(0), 1, device=h.device, dtype=h.dtype)
            protos  = h.unsqueeze(1)                  # [B, 1, d] 连续噪声 token
            vq_commit = vq_diversity = None
            hard_idx  = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        cur = x
        mask_last = None
        history = []
        for _ in range(self.n_refine + 1):
            mu_res, mask = self.backbone(cur, protos)
            cur = cur + mu_res
            mask_last = mask
            history.append(cur)

        clean = cur                          # 不乘 mask, 连续幅度输出

        # quality: 用 mask 平均值粗略估计 (0~1); Ablation-3 无质量头时为 0
        if mask_last is not None:
            quality = mask_last.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)  # [B,1]
        else:
            quality = clean.new_zeros(clean.size(0), 1)

        aux = {
            "det_mask":         mask_last,
            "det_mask_used":    mask_last,
            "cur_premask":      cur,         # 兼容老接口, 与 clean 相同
            "prototype_probs":  probs,
            "vq_commit":        vq_commit,
            "vq_diversity":     vq_diversity,
            "hard_idx":         hard_idx,
            "refine_history":   history,
            "log_var":          torch.zeros_like(cur),  # 兼容旧训练循环
        }
        return clean, quality, z_noise, aux

    def forward_legacy(self, x, z_cond):
        clean, q, z, _ = self.forward(x, z_cond)
        return clean, q, z


# ============================================================
#  Smoke test
# ============================================================
if __name__ == "__main__":
    m = NoiseAwareDenoiserV6(
        in_ch=3, z_dim=128, cond_len=400,
        num_prototypes=16, num_heads=4, n_refine=3, base_ch=32,
    )
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"V6 trainable params: {n_params/1e6:.2f} M")
    x = torch.randn(2, 3, 6000)
    z = torch.randn(2, 3, 400)
    clean, q, zn, aux = m(x, z)
    print("clean:", tuple(clean.shape), "quality:", tuple(q.shape),
          "z:", tuple(zn.shape), "det_mask:", tuple(aux["det_mask"].shape))
