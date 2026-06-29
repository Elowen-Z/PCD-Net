# -*- coding: utf-8 -*-
"""
v3/baeslines/restormer1d.py
===========================
Restormer-1D: 把 Restormer (CVPR 2022) 的核心模块 (MDTA + GDFN) 适配到 1D
地震/微震信号去噪。该 1D 变体被 2024–2025 多篇地震/EEG 去噪论文用作
SOTA Transformer baseline (例: SeisRestormer 2024、EEGRestormer 2025)。

Key blocks:
  - MDTA  : Multi-Dconv head Transposed self-Attention (沿通道维度 attn,
            复杂度 O(C^2) 而非 O(T^2), 适合长序列 T=6000)
  - GDFN  : Gated-Dconv Feed-Forward Network (门控+深度卷积 FFN)
  - U-Net : 3 级编码/解码 + bottleneck + skip

接口与 DeepDenoiser / DPRNN 兼容: forward(x) 直接返回 [B,3,T] 去噪输出
(残差学习: output = input + Δ)。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# LayerNorm 1D (BiasFree, 沿通道维)
# ------------------------------------------------------------
class LayerNorm1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # GroupNorm(1, dim) 等价于通道维 LayerNorm
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


# ------------------------------------------------------------
# MDTA: Multi-Dconv Head Transposed Self-Attention (1D)
# ------------------------------------------------------------
class MDTA1d(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv1d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv1d(
            dim * 3, dim * 3, kernel_size=3, padding=1,
            groups=dim * 3, bias=False,
        )
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        qkv = self.qkv_dw(self.qkv(x))                   # [B, 3C, T]
        q, k, v = qkv.chunk(3, dim=1)                    # 3 × [B, C, T]

        H = self.num_heads
        q = q.reshape(B, H, C // H, T)
        k = k.reshape(B, H, C // H, T)
        v = v.reshape(B, H, C // H, T)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature   # [B,H,Ch,Ch]
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(B, C, T)
        return self.proj(out)


# ------------------------------------------------------------
# GDFN: Gated-Dconv Feed-Forward Network (1D)
# ------------------------------------------------------------
class GDFN1d(nn.Module):
    def __init__(self, dim: int, ffn_expansion: float = 2.66):
        super().__init__()
        hidden = int(dim * ffn_expansion)
        self.proj_in = nn.Conv1d(dim, hidden * 2, kernel_size=1, bias=False)
        self.dw = nn.Conv1d(
            hidden * 2, hidden * 2, kernel_size=3, padding=1,
            groups=hidden * 2, bias=False,
        )
        self.proj_out = nn.Conv1d(hidden, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(self.proj_in(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.proj_out(F.gelu(x1) * x2)


# ------------------------------------------------------------
# Transformer block (MDTA + GDFN, residual)
# ------------------------------------------------------------
class TransformerBlock1d(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_expansion: float = 2.66):
        super().__init__()
        self.norm1 = LayerNorm1d(dim)
        self.attn  = MDTA1d(dim, num_heads)
        self.norm2 = LayerNorm1d(dim)
        self.ffn   = GDFN1d(dim, ffn_expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ------------------------------------------------------------
# Down / Up sample (stride 2 conv / pixelshuffle1d 等价)
# ------------------------------------------------------------
class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Conv1d(dim, dim * 2, kernel_size=3, stride=2,
                              padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.ConvTranspose1d(dim, dim // 2, kernel_size=2,
                                       stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ------------------------------------------------------------
# Restormer-1D 主体 (residual learning)
# ------------------------------------------------------------
class Restormer1D(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        dim: int = 32,
        num_blocks: tuple = (2, 2, 2, 2),
        num_refinement_blocks: int = 2,
        heads: tuple = (1, 2, 4, 8),
        ffn_expansion: float = 2.0,
    ):
        super().__init__()

        self.patch_embed = nn.Conv1d(in_ch, dim, kernel_size=3,
                                     padding=1, bias=False)

        # ── Encoder ────────────────────────────────
        self.enc1 = nn.Sequential(*[
            TransformerBlock1d(dim, heads[0], ffn_expansion)
            for _ in range(num_blocks[0])
        ])
        self.down1 = Downsample1d(dim)                               # dim   → 2 dim

        self.enc2 = nn.Sequential(*[
            TransformerBlock1d(dim * 2, heads[1], ffn_expansion)
            for _ in range(num_blocks[1])
        ])
        self.down2 = Downsample1d(dim * 2)                           # 2 dim → 4 dim

        self.enc3 = nn.Sequential(*[
            TransformerBlock1d(dim * 4, heads[2], ffn_expansion)
            for _ in range(num_blocks[2])
        ])
        self.down3 = Downsample1d(dim * 4)                           # 4 dim → 8 dim

        # ── Bottleneck ─────────────────────────────
        self.latent = nn.Sequential(*[
            TransformerBlock1d(dim * 8, heads[3], ffn_expansion)
            for _ in range(num_blocks[3])
        ])

        # ── Decoder (skip via concat + 1x1 reduce) ─
        self.up3   = Upsample1d(dim * 8)                             # 8 dim → 4 dim
        self.red3  = nn.Conv1d(dim * 8, dim * 4, kernel_size=1, bias=False)
        self.dec3  = nn.Sequential(*[
            TransformerBlock1d(dim * 4, heads[2], ffn_expansion)
            for _ in range(num_blocks[2])
        ])

        self.up2   = Upsample1d(dim * 4)
        self.red2  = nn.Conv1d(dim * 4, dim * 2, kernel_size=1, bias=False)
        self.dec2  = nn.Sequential(*[
            TransformerBlock1d(dim * 2, heads[1], ffn_expansion)
            for _ in range(num_blocks[1])
        ])

        self.up1   = Upsample1d(dim * 2)
        # 此处不做 reduce, 让最后阶段保持 2 dim (与原 Restormer 一致)
        self.dec1  = nn.Sequential(*[
            TransformerBlock1d(dim * 2, heads[0], ffn_expansion)
            for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock1d(dim * 2, heads[0], ffn_expansion)
            for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv1d(dim * 2, in_ch, kernel_size=3,
                                padding=1, bias=False)

    # 兼容与 DPRNN/DeepDenoiser 同样的可选 z_cond 参数 (此处忽略)
    def forward(self, x: torch.Tensor, z_cond=None) -> torch.Tensor:
        x_in = x
        T = x.shape[-1]
        # 长度对齐 8 的倍数 (3 次下采样)
        pad = (8 - T % 8) % 8
        if pad:
            x = F.pad(x, (0, pad), mode="replicate")

        f  = self.patch_embed(x)                  # [B, dim, T']
        e1 = self.enc1(f)                         # dim
        e2 = self.enc2(self.down1(e1))            # 2 dim
        e3 = self.enc3(self.down2(e2))            # 4 dim
        lt = self.latent(self.down3(e3))          # 8 dim

        d3 = self.up3(lt)                         # 4 dim
        d3 = self.red3(torch.cat([d3, e3], 1))    # 8 dim → 4 dim
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                         # 2 dim
        d2 = self.red2(torch.cat([d2, e2], 1))    # 4 dim → 2 dim
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                         # dim
        d1 = torch.cat([d1, e1], 1)               # 2 dim
        d1 = self.dec1(d1)
        d1 = self.refinement(d1)

        out = self.output(d1)
        if pad:
            out = out[..., :T]
        return out + x_in                         # residual learning


if __name__ == "__main__":
    m = Restormer1D()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Restormer1D params: {n / 1e6:.2f} M")
    y = m(torch.randn(2, 3, 6000))
    print("output:", y.shape)
