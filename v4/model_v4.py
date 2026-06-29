# v4/model_v4.py
"""
NoiseAwareDenoiserV4 — 顶会向重构 (CS-conference-friendly)

相对 V3 的 5 个 new claim
==========================
(1) VQ Noise Prototypes (VQ-NP)
    用 K 个可学习噪声原型组成 codebook；NoiseEncoder 输出 soft 分配 p∈Δ^K，
    z_noise = Σ p_k · e_k。可解释 + 跨域 transfer 只需更新 codebook。

(2) Cross-Attention Conditioning
    bottleneck 处不再用 FiLM；signal tokens 作 Q，noise prototype tokens 作 K/V。
    表达力强于通道仿射。

(3) Iterative Residual Refinement (IRR)
    decoder 输出 residual；用同一组权重迭代 N_refine 次：
        x_{t+1} = x_t - g_θ(x_t, z)
    与 diffusion / deep equilibrium 关联，提供动态深度。

(4) Heteroscedastic Uncertainty Head
    输出 (μ, logσ²) per timestep；用 Gaussian NLL 训练 → 可做 calibration
    (ECE, reliability diagram)。quality_score 由 σ 聚合得到，物理意义明确。

(5) Test-Time Adaptation (TTA)
    .adapt(x, n_steps) 仅更新 prototype-assignment 的 logits，主干冻结，
    用 self-consistency loss 在线适应 unseen noise。

I/O 兼容
========
forward(x, z_cond) → (clean, quality, z_noise, aux)
    aux 包含: log_var, prototype_probs, vq_loss, refine_history
旧训练循环若只取前三个返回值，可直接 unpacking 前三个 (Python 报错? 见下)。
为了兼容性，提供 forward_legacy(x, z_cond) → (clean, quality, z_noise)。

依赖: torch>=1.10
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  基础模块 (复用 V3 风格)
# ============================================================
class ConvBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding),
            nn.GroupNorm(min(8, out_ch), out_ch),  # GroupNorm 对小 batch 更友好
            nn.SiLU(),  # inplace=True 在 IRR 共享权重循环下会触发 CUDA 非法内存访问
        )

    def forward(self, x):
        return self.net(x)


class ConvBlockLarge(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=1, padding=kernel // 2),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
#  (1) VQ Noise Prototypes
# ============================================================
class VQNoisePrototypes(nn.Module):
    """
    可学习噪声原型 codebook E ∈ R^{K×d}。
    给定噪声特征 h ∈ R^{B×d}：
        logits  = -||h - e_k||² / τ      [B, K]
        p       = softmax(logits)         [B, K]   (soft assignment)
        z_noise = p @ E                   [B, d]   (prototype mixture)
    训练时加 commitment & diversity loss，避免坍缩。
    """

    def __init__(self, num_prototypes: int = 16, dim: int = 128,
                 temperature: float = 0.3, ema_decay: float = 0.99):
        """
        temperature 默认 0.3（原 1.0 使赋值过平，造成坐塌）。
        """
        super().__init__()
        self.K = num_prototypes
        self.d = dim
        self.tau = temperature
        self.ema_decay = ema_decay

        # codebook
        codebook = torch.randn(num_prototypes, dim) * 0.02
        self.codebook = nn.Parameter(codebook)

        # EMA usage tracker (for diversity regularization & TTA)
        self.register_buffer("ema_usage", torch.ones(num_prototypes) / num_prototypes)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        h: [B, d]
        return:
            z_noise:        [B, d]    soft prototype mixture
            probs:          [B, K]    assignment distribution
            aux: {vq_commit, vq_diversity, hard_idx}
        """
        B = h.size(0)
        # dist2 强制 fp32：避免 bf16/fp16 下大范数时溢出
        h32 = h.float()
        cb32 = self.codebook.float()
        # squared distance to each prototype
        # ||h - e||^2 = h·h - 2 h·e + e·e
        h_norm2 = (h32 * h32).sum(-1, keepdim=True)            # [B,1]
        e_norm2 = (cb32 * cb32).sum(-1)                        # [K]
        cross = h32 @ cb32.t()                                 # [B,K]
        dist2 = (h_norm2 - 2 * cross + e_norm2.unsqueeze(0)).clamp(min=0)  # [B,K] >=0

        logits = (-dist2 / max(self.tau, 1e-6)).to(h.dtype)
        probs = F.softmax(logits, dim=-1)                      # [B,K]

        z_noise = (probs.float() @ cb32).to(h.dtype)            # [B,d]
        hard_idx = probs.argmax(dim=-1)                        # [B]

        # commitment loss: encoder 输出靠近其最近原型
        with torch.no_grad():
            nearest = cb32[hard_idx]                           # [B,d] fp32
        vq_commit = F.mse_loss(h32, nearest)

        # ── 双重多样性损失修复坐塌 ─────────────────────
        # (a) per_sample_entropy: 鼓励单样本 probs peaky (低熵)
        # (b) batch_usage_entropy: 鼓励 batch 平均使用均匀 (高熵)
        # 总损失与原接口同名 (vq_diversity)：
        #   vq_diversity = per_sample_H - batch_usage_H + log K
        # 两项均 >=0，为 0 则完美。
        # 强制 fp32 并 clamp，防止 bf16 概率极小负值导致 log(负数) → NaN
        probs_f = probs.float().clamp(min=1e-8)                # [B,K] fp32, >=1e-8
        avg_p   = probs_f.mean(0)                              # [K]
        per_sample_H  = -(probs_f * probs_f.log()).sum(-1).mean()
        batch_usage_H = -(avg_p   * avg_p.log()).sum()
        log_K = float(np.log(self.K))
        vq_diversity = per_sample_H + (log_K - batch_usage_H)

        if self.training:
            with torch.no_grad():
                self.ema_usage.mul_(self.ema_decay).add_(
                    avg_p.detach(), alpha=1 - self.ema_decay
                )

        aux = {
            "vq_commit": vq_commit,
            "vq_diversity": vq_diversity,
            "hard_idx": hard_idx,
        }
        return z_noise, probs, aux


# ============================================================
#  Noise Encoder (输出连续特征 h，再交给 VQ-NP)
# ============================================================
class NoiseEncoderV4(nn.Module):
    def __init__(self, in_ch: int = 3, dim: int = 128, cond_len: int = 400):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBlock1d(in_ch, 32,  kernel=3, stride=2),  # 400→200
            ConvBlock1d(32,    64,  kernel=3, stride=2),  # 200→100
            ConvBlock1d(64,   128,  kernel=3, stride=2),  # 100→50
            ConvBlock1d(128,  128,  kernel=3, stride=2),  # 50→25
            ConvBlockLarge(128, 128, kernel=15),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.proj(self.enc(cond))                       # [B, dim]


# ============================================================
#  (2) Cross-Attention Conditioning Bottleneck
# ============================================================
class CrossAttnBottleneck(nn.Module):
    """
    signal tokens (Q) attend to noise prototype tokens (K, V).
    Input :
        x:     [B, C, T_b]      bottleneck signal feature
        protos:[B, K, d]        per-sample expanded codebook (broadcast)
    Output:
        x':    [B, C, T_b]
    """

    def __init__(self, c_dim: int, p_dim: int, num_heads: int = 4,
                 ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(c_dim, c_dim)
        self.k_proj = nn.Linear(p_dim, c_dim)
        self.v_proj = nn.Linear(p_dim, c_dim)
        self.attn = nn.MultiheadAttention(
            c_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(c_dim)
        self.norm2 = nn.LayerNorm(c_dim)
        self.ff = nn.Sequential(
            nn.Linear(c_dim, c_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_dim * ff_mult, c_dim),
        )

    def forward(self, x: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] → tokens
        B, C, T = x.shape
        tokens = x.permute(0, 2, 1)                            # [B, T, C]

        q = self.q_proj(tokens)
        k = self.k_proj(protos)
        v = self.v_proj(protos)

        attn_out, _ = self.attn(q, k, v, need_weights=False)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ff(tokens))

        return tokens.permute(0, 2, 1)                         # [B, C, T]


# ============================================================
#  Decoder + (4) Heteroscedastic Output Head
# ============================================================
class UNetBackboneV4(nn.Module):
    """
    Encoder + bottleneck (cross-attn) + decoder。
    output_head 输出 2*in_ch 通道：前 in_ch 为 μ_residual，后 in_ch 为 logσ²。
    """

    def __init__(self, in_ch: int = 3, p_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.in_ch = in_ch

        # ── Encoder ─────────────────────────────────────
        self.enc1 = ConvBlock1d(in_ch, 16,  kernel=3, stride=2)
        self.enc2 = ConvBlock1d(16,    32,  kernel=3, stride=2)
        self.enc3 = ConvBlock1d(32,    64,  kernel=3, stride=2)
        self.enc4 = ConvBlock1d(64,   128,  kernel=3, stride=2)
        self.enc5 = ConvBlock1d(128,  128,  kernel=3, stride=2)

        self.ref1 = ConvBlockLarge(16,  16,  kernel=15)
        self.ref2 = ConvBlockLarge(32,  32,  kernel=15)
        self.ref3 = ConvBlockLarge(64,  64,  kernel=15)
        self.ref4 = ConvBlockLarge(128, 128, kernel=7)
        self.ref5 = ConvBlockLarge(128, 128, kernel=5)

        # ── Cross-attn bottleneck (NEW) ─────────────────
        self.bn_conv = ConvBlockLarge(128, 128, kernel=5)
        self.bn_xattn = CrossAttnBottleneck(
            c_dim=128, p_dim=p_dim, num_heads=num_heads,
        )

        # ── Decoder ─────────────────────────────────────
        self.dec4 = nn.Sequential(
            ConvBlock1d(128 + 128, 128, kernel=3),
            ConvBlockLarge(128, 64, kernel=15),
        )
        self.dec3 = nn.Sequential(
            ConvBlock1d(64 + 64, 64, kernel=3),
            ConvBlockLarge(64, 32, kernel=15),
        )
        self.dec2 = nn.Sequential(
            ConvBlock1d(32 + 32, 32, kernel=3),
            ConvBlockLarge(32, 16, kernel=15),
        )
        self.dec1 = nn.Sequential(
            ConvBlock1d(16 + 16, 16, kernel=3),
            ConvBlockLarge(16, 16, kernel=15),
        )
        self.dec0 = ConvBlockLarge(16, 16, kernel=7)

        # ── 输出头：μ_residual + logσ² ───────────────────
        self.out_head = nn.Conv1d(16, 2 * in_ch, kernel_size=1)
        # ── 信号检测头：输出逐时刻信号概率 mask [B,1,T] ─────
        # 从 dec0 的 16ch 特征预测，与 out_head 并行不干扰
        self.mask_head = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(16, 1,  kernel_size=1),
            nn.Sigmoid(),
        )
    @staticmethod
    def _align(x, target):
        if x.shape[-1] != target.shape[-1]:
            x = F.interpolate(x, size=target.shape[-1], mode="nearest")
        return x

    @staticmethod
    def _up2(x):
        return x.repeat_interleave(2, dim=-1)

    def forward(self, x: torch.Tensor, protos: torch.Tensor):
        """
        x      : [B, in_ch, T]
        protos : [B, K, p_dim]
        return : (mu_res [B, in_ch, T], log_var [B, in_ch, T])
        """
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

        out = self.out_head(d0)                                # [B, 2C, T]
        mu_res, log_var = out.chunk(2, dim=1)                  # each [B, C, T]
        # 收紧 log_var 的下界，避免 NLL 退化为“预测极小方差刷分”
        log_var = torch.clamp(log_var, min=-4.0, max=4.0)
        # 信号检测 mask（与 out_head 共享 d0）
        mask = self.mask_head(d0)                              # [B, 1, T]
        return mu_res, log_var, mask


# ============================================================
#  完整模型 (含 IRR 与 TTA)
# ============================================================
class NoiseAwareDenoiserV4(nn.Module):
    """
    Stage 1: NoiseEncoder → h ∈ R^d
    VQ-NP : h → (z_noise, probs, vq_aux)
    Stage 2: UNetBackbone (cross-attn 条件 protos)
    IRR   : 共享权重迭代 N_refine 次预测残差
    Output: clean = x_noisy + Σ μ_residual_t (residual prediction)
            quality 由 σ 聚合
    """

    def __init__(
        self,
        in_ch:           int   = 3,
        z_dim:           int   = 128,
        cond_len:        int   = 400,
        num_prototypes:  int   = 16,
        num_heads:       int   = 4,
        n_refine:        int   = 2,
        vq_temperature:  float = 1.0,
    ):
        super().__init__()
        self.in_ch    = in_ch
        self.z_dim    = z_dim
        self.n_refine = n_refine

        self.noise_encoder = NoiseEncoderV4(in_ch, dim=z_dim, cond_len=cond_len)
        self.vq = VQNoisePrototypes(
            num_prototypes=num_prototypes, dim=z_dim,
            temperature=vq_temperature,
        )
        self.backbone = UNetBackboneV4(
            in_ch=in_ch, p_dim=z_dim, num_heads=num_heads,
        )

        # TTA 用的可学习偏置 logits（默认 0；adapt() 时仅更新此参数）
        self.register_parameter(
            "tta_logits",
            nn.Parameter(torch.zeros(num_prototypes), requires_grad=False),
        )

    # ------------------------------------------------------------
    def _get_protos(self, probs: torch.Tensor) -> torch.Tensor:
        """
        将整个 codebook 作为 K 个 prototype tokens 暴露给 cross-attn。
        额外乘以 (probs * tta_bias) 做样本相关重标定。
        protos: [B, K, d]
        """
        B = probs.size(0)
        E = self.vq.codebook                                  # [K, d]
        # 应用 TTA bias（推理时可被 adapt 修改）
        bias = F.softmax(self.tta_logits, dim=-1)             # [K]
        scale = (probs * bias.unsqueeze(0))                   # [B, K]
        scale = scale / (scale.sum(-1, keepdim=True) + 1e-8)
        # 返回 per-sample re-weighted prototypes（保留全部 K 个 token）
        protos = E.unsqueeze(0).expand(B, -1, -1)             # [B, K, d]
        protos = protos * scale.unsqueeze(-1)                 # broadcast
        return protos

    # ------------------------------------------------------------
    def forward(
        self,
        x:      torch.Tensor,
        z_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        x      : [B, in_ch, T]
        z_cond : [B, in_ch, cond_len]
        return : (clean, quality, z_noise, aux)
        """
        # -- (1) noise embedding via VQ-NP ---------------
        h = self.noise_encoder(z_cond)                        # [B, d]
        z_noise, probs, vq_aux = self.vq(h)                   # [B, d], [B, K]
        protos = self._get_protos(probs)                      # [B, K, d]

        # -- (3) Iterative Residual Refinement -----------
        # IRR: 共享权重迭代 n_refine 次精炼残差；mask 在最后一次提取
        cur = x
        log_var_last = None
        mask_last    = None
        history      = []
        for t in range(self.n_refine + 1):
            mu_res, log_var, mask = self.backbone(cur, protos)
            cur          = cur + mu_res           # 逐步精炼（不加 mask，保持 IRR 稳定性）
            log_var_last = log_var
            mask_last    = mask
            history.append(cur)

        # 最终用 mask 抑制背景：clean = denoised * mask
        # 背景区 mask→0 → clean≈0；信号区 mask→1 → clean≈去噪结果
        # ── soft dilation: max-pool 把 mask 在信号边缘向外扩展若干样本，
        #    避免 sigmoid 在 onset / 弱振幅处 (<1) 把真实信号一起削掉。
        #    mask_dilate_k 由 self.mask_dilate_k 控制 (默认 0=关闭，保持向后兼容)
        if getattr(self, "mask_dilate_k", 0) and self.mask_dilate_k > 1:
            k = int(self.mask_dilate_k)
            mask_used = F.max_pool1d(
                mask_last, kernel_size=k, stride=1, padding=k // 2
            )
            if mask_used.shape[-1] != mask_last.shape[-1]:
                mask_used = mask_used[..., : mask_last.shape[-1]]
        else:
            mask_used = mask_last
        clean = cur * mask_used                                 # [B, C, T]

        # -- (4) Quality from heteroscedastic σ ----------
        # quality ∈ (0,1)：σ 越小 → 质量越高
        sigma = torch.exp(0.5 * log_var_last)                  # [B, C, T]
        sigma_mean = sigma.mean(dim=(1, 2))                    # [B]
        quality = torch.sigmoid(-sigma_mean.unsqueeze(-1))     # [B, 1]

        aux = {
            "log_var": log_var_last,
            "det_mask": mask_last,                             # [B,1,T] 信号概率(原始)
            "det_mask_used": mask_used,                        # [B,1,T] 实际使用的 mask
            "cur_premask": cur,                                # [B, C, T] 未乘 mask 的去噪
            "prototype_probs": probs,
            "vq_commit": vq_aux["vq_commit"],
            "vq_diversity": vq_aux["vq_diversity"],
            "hard_idx": vq_aux["hard_idx"],
            "refine_history": history,
        }
        return clean, quality, z_noise, aux

    # 兼容旧训练循环
    def forward_legacy(self, x, z_cond):
        clean, quality, z_noise, _ = self.forward(x, z_cond)
        return clean, quality, z_noise

    # ------------------------------------------------------------
    @torch.no_grad()
    def encode_noise(self, z_cond):
        h = self.noise_encoder(z_cond)
        z_noise, probs, _ = self.vq(h)
        return z_noise, probs

    # ------------------------------------------------------------
    #  (5) Test-Time Adaptation
    # ------------------------------------------------------------
    def adapt(
        self,
        x:           torch.Tensor,
        z_cond:      torch.Tensor,
        n_steps:     int   = 5,
        lr:          float = 0.05,
        consistency: bool  = True,
    ) -> Dict:
        """
        在线无监督适应：仅更新 self.tta_logits（K 维），主干冻结。
        损失 = self-consistency:
            两次随机扰动输入下 → 去噪结果应一致。
        返回最终 aux dict。
        """
        # 暂存训练状态并冻结
        was_training = self.training
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.tta_logits.requires_grad_(True)

        opt = torch.optim.Adam([self.tta_logits], lr=lr)

        with torch.enable_grad():          # 即使外层有 no_grad，也强制开启梯度
            for step in range(n_steps):
                # 两个随机噪声扰动视图
                v1 = x + 0.01 * torch.randn_like(x)
                v2 = x + 0.01 * torch.randn_like(x)
                c1, _, _, _ = self.forward(v1, z_cond)
                c2, _, _, _ = self.forward(v2, z_cond)

                loss = F.mse_loss(c1, c2)

                opt.zero_grad()
                loss.backward()
                opt.step()

        # 还原状态
        self.tta_logits.requires_grad_(False)
        for p in self.parameters():
            p.requires_grad_(False)
        if was_training:
            self.train()
        else:
            self.eval()

        with torch.no_grad():
            clean, quality, z_noise, aux = self.forward(x, z_cond)
        aux["tta_bias"] = F.softmax(self.tta_logits, dim=-1).detach()
        aux["tta_final_loss"] = loss.detach()
        return clean, quality, z_noise, aux

    # ------------------------------------------------------------
    def reset_tta(self):
        """重置 TTA bias 为均匀分布"""
        with torch.no_grad():
            self.tta_logits.zero_()


# ============================================================
#  推荐损失（供 train_v4.py 使用，非必须 import）
# ============================================================
def gaussian_nll_loss(
    pred_mu: torch.Tensor,
    target:  torch.Tensor,
    log_var: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    NLL = 0.5 * (log_var + (target - pred)^2 * exp(-log_var))
    """
    sq = (target - pred_mu) ** 2
    nll = 0.5 * (log_var + sq * torch.exp(-log_var))
    if valid_mask is not None:
        # valid_mask: [B, T] → broadcast 到 [B, C, T]
        m = valid_mask.unsqueeze(1)
        nll = (nll * m).sum() / (m.sum() * pred_mu.size(1) + 1e-8)
    else:
        nll = nll.mean()
    return nll


# ============================================================
#  Sanity test
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = NoiseAwareDenoiserV4(
        in_ch=3, z_dim=128, cond_len=400,
        num_prototypes=16, num_heads=4, n_refine=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {n_params/1e6:.2f} M")

    for T in [6000, 3000, 8000]:
        x      = torch.randn(2, 3, T, device=device)
        z_cond = torch.randn(2, 3, 400, device=device)
        try:
            clean, quality, z_noise, aux = model(x, z_cond)
            print(f"OK T={T:5d} | clean={tuple(clean.shape)} "
                  f"q={tuple(quality.shape)} z={tuple(z_noise.shape)} "
                  f"logvar={tuple(aux['log_var'].shape)} "
                  f"probs={tuple(aux['prototype_probs'].shape)}")
        except Exception as e:
            print(f"FAIL T={T}: {e}")
            raise

    # NLL loss smoke test
    target = torch.randn_like(clean)
    nll = gaussian_nll_loss(clean, target, aux["log_var"])
    print(f"NLL loss (random) = {nll.item():.4f}")

    # TTA smoke test
    print("\n[TTA] before:", model.tta_logits.detach().cpu().numpy()[:4])
    clean2, q2, z2, aux2 = model.adapt(x, z_cond, n_steps=3, lr=0.1)
    print("[TTA] after :", model.tta_logits.detach().cpu().numpy()[:4])
    print("[TTA] final loss:", aux2["tta_final_loss"].item())

    print("\n[OK] all tests passed.")
