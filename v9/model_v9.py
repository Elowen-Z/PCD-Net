"""PCD-Net V9: cosine soft routing with a routing-biased prototype attention.

The implementation follows the paper equations directly:

    alpha = softmax(cos(E_b(b), P) / tau)
    A = softmax(Q K^T / sqrt(d_h) + lambda_r log(alpha + eps)) V

All K prototypes remain in the attention memory.  No Top-M/Top-1 operation is
used.  The surrounding V7 restoration/refinement pipeline is retained so the
routing change is isolated from unrelated architectural changes.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v5.model_v6 import UNetBackboneV6
from v5.model_v6 import ConcatConditioning
from v7.model_v7 import NoiseAwareDenoiserV7


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("positive routing parameters are required")
    return math.log(math.expm1(value))


class SoftPrototypeRouter(nn.Module):
    """Route a background embedding softly over the complete prototype bank."""

    def __init__(
        self,
        temperature: float = 0.3,
        routing_strength: float = 1.0,
        learnable: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = float(eps)
        self.raw_temperature = nn.Parameter(
            torch.tensor(_inverse_softplus(temperature), dtype=torch.float32),
            requires_grad=learnable,
        )
        self.raw_routing_strength = nn.Parameter(
            torch.tensor(_inverse_softplus(routing_strength), dtype=torch.float32),
            requires_grad=learnable,
        )

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature).clamp_min(self.eps)

    @property
    def routing_strength(self) -> torch.Tensor:
        return F.softplus(self.raw_routing_strength)

    def forward(
        self,
        background: torch.Tensor,
        prototypes: torch.Tensor,
        use_routing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if background.ndim != 2 or prototypes.ndim != 2:
            raise ValueError("background and prototypes must have shapes [B,D] and [K,D]")
        if background.size(1) != prototypes.size(1):
            raise ValueError("background and prototype dimensions must match")

        cosine = F.normalize(background.float(), dim=-1) @ F.normalize(
            prototypes.float(), dim=-1
        ).t()
        logits = cosine / self.temperature.float()
        if use_routing:
            alpha = F.softmax(logits, dim=-1)
        else:
            alpha = torch.full_like(logits, 1.0 / logits.size(-1))
        route_bias = self.routing_strength.float() * torch.log(
            alpha.clamp_min(self.eps)
        )
        return alpha.to(background.dtype), cosine.to(background.dtype), route_bias.to(background.dtype)


class RoutingPriorCrossAttention(nn.Module):
    """Multi-head cross-attention with the soft routing prior added to logits."""

    def __init__(
        self,
        c_dim: int,
        p_dim: int,
        num_heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if c_dim % num_heads != 0:
            raise ValueError("c_dim must be divisible by num_heads")
        self.c_dim = c_dim
        self.p_dim = p_dim
        self.num_heads = num_heads
        self.head_dim = c_dim // num_heads
        self.q_proj = nn.Linear(c_dim, c_dim)
        self.k_proj = nn.Linear(p_dim, c_dim)
        self.v_proj = nn.Linear(p_dim, c_dim)
        self.out_proj = nn.Linear(c_dim, c_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(c_dim)
        self.norm2 = nn.LayerNorm(c_dim)
        self.ff = nn.Sequential(
            nn.Linear(c_dim, c_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_dim * ff_mult, c_dim),
        )
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, routed_memory: torch.Tensor) -> torch.Tensor:
        if routed_memory.size(-1) != self.p_dim + 1:
            raise ValueError("routed memory must contain prototype features plus one route-bias channel")
        batch, _, steps = x.shape
        prototypes = routed_memory[..., : self.p_dim]
        route_bias = routed_memory[..., self.p_dim]
        tokens = x.permute(0, 2, 1)

        q = self.q_proj(tokens).view(batch, steps, self.num_heads, self.head_dim)
        k = self.k_proj(prototypes).view(
            batch, prototypes.size(1), self.num_heads, self.head_dim
        )
        v = self.v_proj(prototypes).view(
            batch, prototypes.size(1), self.num_heads, self.head_dim
        )
        logits = torch.einsum("bthd,bkhd->bhtk", q, k) / math.sqrt(self.head_dim)
        logits = logits + route_bias[:, None, None, :]
        attention = F.softmax(logits, dim=-1)
        attended = torch.einsum("bhtk,bkhd->bthd", attention, v)
        attended = attended.reshape(batch, steps, self.c_dim)
        attended = self.out_proj(attended)
        tokens = self.norm1(tokens + self.dropout(attended))
        tokens = self.norm2(tokens + self.ff(tokens))
        self.last_attention = attention.detach()
        return tokens.permute(0, 2, 1)


class UNetBackboneV9(UNetBackboneV6):
    def __init__(
        self,
        *args,
        p_dim: int = 128,
        num_heads: int = 4,
        prototype_interaction: str = "cross_attention",
        **kwargs,
    ):
        if prototype_interaction not in {"cross_attention", "weighted_concat"}:
            raise ValueError(
                "prototype_interaction must be 'cross_attention' or 'weighted_concat'"
            )
        super().__init__(*args, p_dim=p_dim, num_heads=num_heads, use_cross_attn=False, **kwargs)
        if prototype_interaction == "cross_attention":
            self.bn_xattn = RoutingPriorCrossAttention(
                c_dim=self.base_ch * 8,
                p_dim=p_dim,
                num_heads=num_heads,
            )
            self.use_cross_attn = True
        else:
            self.bn_xattn = ConcatConditioning(
                c_dim=self.base_ch * 8,
                p_dim=p_dim,
            )
            self.use_cross_attn = False
        self.prototype_interaction = prototype_interaction

    def forward(self, x: torch.Tensor, routed_memory: torch.Tensor):
        if self.prototype_interaction == "cross_attention":
            return super().forward(x, routed_memory)
        prototypes = routed_memory[..., : self.bn_xattn.z_proj.in_features]
        route_bias = routed_memory[..., self.bn_xattn.z_proj.in_features]
        weights = F.softmax(route_bias.float(), dim=-1).to(prototypes.dtype)
        weighted_context = torch.sum(prototypes * weights.unsqueeze(-1), dim=1, keepdim=True)
        return super().forward(x, weighted_context)


class NoiseAwareDenoiserV9(NoiseAwareDenoiserV7):
    """V7 restoration pipeline with dense, differentiable prototype routing."""

    def __init__(
        self,
        in_ch: int = 3,
        z_dim: int = 128,
        cond_len: int = 400,
        num_prototypes: int = 16,
        num_heads: int = 4,
        n_refine: int = 0,
        base_ch: int = 32,
        routing_temperature: float = 0.3,
        routing_strength: float = 1.0,
        learnable_routing: bool = True,
        use_routing: bool = True,
        use_prototypes: bool = True,
        use_quality_head: bool = True,
        use_residual_feedback: bool = False,
        adaptive_inference: bool = False,
        stop_threshold: float = 0.95,
        min_refine_steps: int = 1,
        prototype_interaction: str = "cross_attention",
    ):
        super().__init__(
            in_ch=in_ch,
            z_dim=z_dim,
            cond_len=cond_len,
            num_prototypes=num_prototypes,
            top_m=num_prototypes,
            num_heads=num_heads,
            n_refine=n_refine,
            base_ch=base_ch,
            vq_temperature=routing_temperature,
            use_prototypes=use_prototypes,
            use_sparse_selection=False,
            use_cross_attn=True,
            use_quality_head=use_quality_head,
            use_residual_feedback=use_residual_feedback,
            adaptive_inference=adaptive_inference,
            stop_threshold=stop_threshold,
            min_refine_steps=min_refine_steps,
        )
        # These V7 compatibility objects implement Top-M routing and are not
        # part of V9. Removing them also keeps V9 checkpoints unambiguous.
        del self.selector
        del self.tta_logits
        self.use_routing = bool(use_routing)
        self.router = SoftPrototypeRouter(
            temperature=routing_temperature,
            routing_strength=routing_strength,
            learnable=learnable_routing,
        )
        self.backbone = UNetBackboneV9(
            in_ch=in_ch,
            p_dim=z_dim,
            num_heads=num_heads,
            base_ch=base_ch,
            prototype_interaction=prototype_interaction,
            use_quality_head=True,
        )

    def _prototype_tokens(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.use_prototypes:
            alpha = h.new_ones(h.size(0), 1)
            routed_memory = torch.cat([h.unsqueeze(1), h.new_zeros(h.size(0), 1, 1)], dim=-1)
            aux = {
                "vq_commit": None,
                "vq_diversity": None,
                "hard_idx": h.new_zeros(h.size(0), dtype=torch.long),
                "prototype_probs": alpha,
                "routing_alpha": alpha,
                "routing_similarity": alpha,
                "top_indices": h.new_zeros(h.size(0), 1, dtype=torch.long),
                "top_weights": alpha,
                "sparse_probs": alpha,
                "selection_mask": alpha,
                "selected_mass": alpha.squeeze(1),
            }
            return routed_memory, h, aux

        alpha, similarity, route_bias = self.router(
            h, self.vq.codebook, use_routing=self.use_routing
        )
        codebook = self.vq.codebook.unsqueeze(0).expand(h.size(0), -1, -1)
        routed_memory = torch.cat([codebook, route_bias.unsqueeze(-1)], dim=-1)
        z_noise = alpha.float() @ self.vq.codebook.float()
        z_noise = z_noise.to(h.dtype)
        hard_idx = alpha.argmax(dim=-1)
        nearest = self.vq.codebook.float()[hard_idx]
        vq_commit = F.mse_loss(h.float(), nearest.detach())
        mean_usage = alpha.float().mean(dim=0).clamp_min(1e-8)
        per_sample_entropy = -(alpha.float().clamp_min(1e-8) * alpha.float().clamp_min(1e-8).log()).sum(-1).mean()
        batch_entropy = -(mean_usage * mean_usage.log()).sum()
        vq_diversity = per_sample_entropy + math.log(alpha.size(1)) - batch_entropy
        if self.training:
            with torch.no_grad():
                self.vq.ema_usage.mul_(self.vq.ema_decay).add_(mean_usage, alpha=1 - self.vq.ema_decay)

        all_indices = torch.arange(alpha.size(1), device=alpha.device).unsqueeze(0).expand(alpha.size(0), -1)
        aux = {
            "vq_commit": vq_commit,
            "vq_diversity": vq_diversity,
            "hard_idx": hard_idx,
            "prototype_probs": alpha,
            "routing_alpha": alpha,
            "routing_similarity": similarity,
            "top_indices": all_indices,
            "top_weights": alpha,
            "sparse_probs": alpha,
            "selection_mask": torch.ones_like(alpha),
            "selected_mass": torch.ones(alpha.size(0), device=alpha.device, dtype=alpha.dtype),
        }
        return routed_memory, z_noise, aux

    def forward(self, *args, **kwargs):
        clean, quality, z_noise, aux = super().forward(*args, **kwargs)
        alpha = aux["prototype_probs"]
        entropy = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(-1)
        aux["routing_alpha"] = alpha
        aux["routing_entropy"] = entropy
        aux["effective_prototypes"] = torch.exp(entropy)
        aux["routing_temperature"] = self.router.temperature.detach()
        aux["routing_strength"] = self.router.routing_strength.detach()
        aux["prototype_attention"] = getattr(
            self.backbone.bn_xattn, "last_attention", None
        )
        return clean, quality, z_noise, aux

    def load_compatible_state_dict(self, state_dict: dict) -> Tuple[list, list]:
        current = self.state_dict()
        compatible = {
            key: value for key, value in state_dict.items()
            if key in current and current[key].shape == value.shape
        }
        incompatible = self.load_state_dict(compatible, strict=False)
        return incompatible.missing_keys, incompatible.unexpected_keys


if __name__ == "__main__":
    model = NoiseAwareDenoiserV9(z_dim=32, num_prototypes=8, num_heads=4, n_refine=0, base_ch=8)
    x = torch.randn(2, 3, 512)
    b = torch.randn(2, 3, 400)
    y, q, _, diagnostics = model(x, b, adaptive_stop=False)
    print(y.shape, q.shape, diagnostics["routing_alpha"].shape)
