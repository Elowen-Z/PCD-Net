"""
PCD-Net V7.

This version implements the innovations described in the paper:
  1. differentiable Top-M sparse prototype selection;
  2. prototype-guided cross-attention with only M key/value tokens;
  3. residual-feedback prototype re-retrieval;
  4. quality-driven adaptive inference;
  5. an independently learned denoising-quality estimator.

The signal backbone and VQ encoder are reused from V6 so existing V6
checkpoints can be partially loaded with strict=False.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v4.model_v4 import NoiseEncoderV4, VQNoisePrototypes
from v5.model_v6 import UNetBackboneV6


class SparsePrototypeSelector(nn.Module):
    """Select and reweight the Top-M prototype tokens for each sample."""

    def __init__(self, top_m: int = 4, straight_through: bool = True):
        super().__init__()
        if top_m < 1:
            raise ValueError("top_m must be positive")
        self.top_m = int(top_m)
        self.straight_through = bool(straight_through)

    def forward(
        self,
        probs: torch.Tensor,
        codebook: torch.Tensor,
        prior_logits: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if probs.ndim != 2 or codebook.ndim != 2:
            raise ValueError("probs and codebook must have shapes [B,K] and [K,D]")
        if probs.size(1) != codebook.size(0):
            raise ValueError("prototype count mismatch")

        k = probs.size(1)
        m = min(self.top_m, k)
        scores = probs
        if prior_logits is not None:
            prior = F.softmax(prior_logits, dim=-1)
            scores = scores * prior.unsqueeze(0)
            scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        top_values, top_indices = torch.topk(scores, k=m, dim=-1, sorted=True)
        sparse_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        batch_codebook = codebook.unsqueeze(0).expand(probs.size(0), -1, -1)
        gather_idx = top_indices.unsqueeze(-1).expand(-1, -1, codebook.size(1))
        selected = torch.gather(batch_codebook, dim=1, index=gather_idx)

        # Forward uses sparse weights. The straight-through term preserves a
        # useful gradient path to all assignment logits around Top-M changes.
        if self.straight_through:
            dense_selected = torch.gather(scores, 1, top_indices)
            dense_selected = dense_selected / dense_selected.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            sparse_weights = (
                sparse_weights.detach() - dense_selected.detach() + dense_selected
            )

        tokens = selected * sparse_weights.unsqueeze(-1)
        hard_mask = torch.zeros_like(scores).scatter(1, top_indices, 1.0)
        sparse_probs = hard_mask * scores
        sparse_probs = sparse_probs / sparse_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        return tokens, {
            "top_indices": top_indices,
            "top_weights": sparse_weights,
            "sparse_probs": sparse_probs,
            "selection_mask": hard_mask,
            "selected_mass": top_values.sum(dim=-1),
        }


class DenoisingQualityHead(nn.Module):
    """Predict output fidelity from input, reconstruction and residual."""

    def __init__(self, in_ch: int = 3, hidden: int = 32):
        super().__init__()
        feature_ch = in_ch * 3 + 1
        self.features = nn.Sequential(
            nn.Conv1d(feature_ch, hidden, 9, stride=4, padding=4),
            nn.GroupNorm(4, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden * 2, 7, stride=4, padding=3),
            nn.GroupNorm(8, hidden * 2),
            nn.SiLU(),
            nn.Conv1d(hidden * 2, hidden * 2, 5, stride=4, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        det_mask: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = noisy - clean
        if det_mask is None:
            det_mask = clean.new_zeros(clean.size(0), 1, clean.size(-1))
        elif det_mask.size(-1) != clean.size(-1):
            det_mask = F.interpolate(
                det_mask, size=clean.size(-1), mode="linear", align_corners=False
            )
        features = torch.cat([noisy, clean, residual, det_mask], dim=1)
        logits = self.regressor(self.features(features))
        return torch.sigmoid(logits), logits


class ResidualPrototypeUpdater(nn.Module):
    """Fuse background and removed-noise evidence into a new noise state."""

    def __init__(self, dim: int):
        super().__init__()
        joint_dim = dim * 3
        self.gate = nn.Sequential(
            nn.Linear(joint_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.candidate = nn.Sequential(
            nn.Linear(joint_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Tanh(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        background: torch.Tensor,
        previous: torch.Tensor,
        residual_evidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint = torch.cat([background, previous, residual_evidence], dim=-1)
        gate = self.gate(joint)
        candidate = background + self.candidate(joint)
        updated = self.norm((1.0 - gate) * previous + gate * candidate)
        return updated, gate


class NoiseAwareDenoiserV7(nn.Module):
    """Paper-aligned PCD-Net with sparse prototype memory."""

    def __init__(
        self,
        in_ch: int = 3,
        z_dim: int = 128,
        cond_len: int = 400,
        num_prototypes: int = 16,
        top_m: int = 4,
        num_heads: int = 4,
        n_refine: int = 3,
        base_ch: int = 32,
        vq_temperature: float = 0.3,
        use_prototypes: bool = True,
        use_sparse_selection: bool = True,
        use_cross_attn: bool = True,
        use_quality_head: bool = True,
        use_residual_feedback: bool = True,
        adaptive_inference: bool = True,
        stop_threshold: float = 0.95,
        min_refine_steps: int = 1,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.z_dim = z_dim
        self.n_refine = int(n_refine)
        self.use_prototypes = bool(use_prototypes)
        self.use_sparse_selection = bool(use_sparse_selection)
        self.use_cross_attn = bool(use_cross_attn)
        self.use_quality_head = bool(use_quality_head)
        self.use_residual_feedback = bool(use_residual_feedback)
        self.adaptive_inference = bool(adaptive_inference)
        self.stop_threshold = float(stop_threshold)
        self.min_refine_steps = int(min_refine_steps)
        self.cond_len = int(cond_len)
        if not 0.0 < self.stop_threshold < 1.0:
            raise ValueError("stop_threshold must be in (0, 1)")
        if self.min_refine_steps < 1:
            raise ValueError("min_refine_steps must be positive")

        self.noise_encoder = NoiseEncoderV4(in_ch, dim=z_dim, cond_len=cond_len)
        self.feedback_updater = ResidualPrototypeUpdater(z_dim)
        self.vq = VQNoisePrototypes(
            num_prototypes=num_prototypes,
            dim=z_dim,
            temperature=vq_temperature,
        )
        self.selector = SparsePrototypeSelector(top_m=top_m)
        self.backbone = UNetBackboneV6(
            in_ch=in_ch,
            p_dim=z_dim,
            num_heads=num_heads,
            base_ch=base_ch,
            use_cross_attn=use_cross_attn,
            use_quality_head=True,
        )
        self.quality_head = (
            DenoisingQualityHead(in_ch=in_ch) if use_quality_head else None
        )
        self.tta_logits = nn.Parameter(
            torch.zeros(num_prototypes), requires_grad=False
        )

    def _encode_residual_feedback(self, removed: torch.Tensor) -> torch.Tensor:
        if removed.size(-1) != self.cond_len:
            removed = F.interpolate(
                removed,
                size=self.cond_len,
                mode="linear",
                align_corners=False,
            )
        return self.noise_encoder(removed)

    def _prototype_tokens(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.use_prototypes:
            probs = h.new_ones(h.size(0), 1)
            aux = {
                "vq_commit": None,
                "vq_diversity": None,
                "hard_idx": h.new_zeros(h.size(0), dtype=torch.long),
                "prototype_probs": probs,
                "top_indices": h.new_zeros(h.size(0), 1, dtype=torch.long),
                "top_weights": probs,
                "sparse_probs": probs,
                "selection_mask": probs,
                "selected_mass": probs.squeeze(1),
            }
            return h.unsqueeze(1), h, aux

        z_noise, probs, vq_aux = self.vq(h)
        if self.use_sparse_selection:
            tokens, selection_aux = self.selector(
                probs, self.vq.codebook, self.tta_logits
            )
        else:
            bias = F.softmax(self.tta_logits, dim=-1)
            weights = probs * bias.unsqueeze(0)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            tokens = self.vq.codebook.unsqueeze(0) * weights.unsqueeze(-1)
            selection_aux = {
                "top_indices": torch.arange(
                    probs.size(1), device=probs.device
                ).unsqueeze(0).expand(probs.size(0), -1),
                "top_weights": weights,
                "sparse_probs": weights,
                "selection_mask": torch.ones_like(probs),
                "selected_mass": torch.ones(
                    probs.size(0), device=probs.device, dtype=probs.dtype
                ),
            }

        aux = dict(vq_aux)
        aux.update(selection_aux)
        aux["prototype_probs"] = probs
        return tokens, z_noise, aux

    @staticmethod
    def _mean_optional(values):
        values = [value for value in values if value is not None]
        return torch.stack(values).mean() if values else None

    def _forward_unrolled(
        self,
        x: torch.Tensor,
        z_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        h_background = self.noise_encoder(z_cond)
        h_state = h_background
        cur = x
        mask_last = None
        refine_history = []
        quality_history = []
        quality_logits_history = []
        prototype_history = []
        sparse_history = []
        top_indices_history = []
        top_weights_history = []
        selected_mass_history = []
        feedback_gate_history = []
        vq_commit_history = []
        vq_diversity_history = []
        proto_aux = None
        z_noise = h_state

        for step in range(self.n_refine + 1):
            protos, z_noise, proto_aux = self._prototype_tokens(h_state)
            mu_res, mask_last = self.backbone(cur, protos)
            cur = cur + mu_res
            refine_history.append(cur)

            if self.quality_head is not None:
                quality, quality_logits = self.quality_head(x, cur, mask_last)
            else:
                quality = cur.new_zeros(cur.size(0), 1)
                quality_logits = cur.new_zeros(cur.size(0), 1)
            quality_history.append(quality)
            quality_logits_history.append(quality_logits)
            prototype_history.append(proto_aux["prototype_probs"])
            sparse_history.append(proto_aux["sparse_probs"])
            top_indices_history.append(proto_aux["top_indices"])
            top_weights_history.append(proto_aux["top_weights"])
            selected_mass_history.append(proto_aux["selected_mass"])
            vq_commit_history.append(proto_aux.get("vq_commit"))
            vq_diversity_history.append(proto_aux.get("vq_diversity"))

            if self.use_residual_feedback and step < self.n_refine:
                residual_evidence = self._encode_residual_feedback(x - cur)
                h_state, feedback_gate = self.feedback_updater(
                    h_background, h_state, residual_evidence
                )
                feedback_gate_history.append(feedback_gate)

        clean = cur
        quality = quality_history[-1]
        quality_logits = quality_logits_history[-1]
        effective_steps = torch.full(
            (x.size(0),),
            self.n_refine + 1,
            device=x.device,
            dtype=torch.long,
        )

        aux = {
            "det_mask": mask_last,
            "det_mask_used": mask_last,
            "cur_premask": clean,
            "prototype_probs": proto_aux["prototype_probs"],
            "sparse_probs": proto_aux["sparse_probs"],
            "top_indices": proto_aux["top_indices"],
            "top_weights": proto_aux["top_weights"],
            "selection_mask": proto_aux["selection_mask"],
            "selected_mass": proto_aux["selected_mass"],
            "vq_commit": self._mean_optional(vq_commit_history),
            "vq_diversity": self._mean_optional(vq_diversity_history),
            "hard_idx": proto_aux.get("hard_idx"),
            "refine_history": refine_history,
            "quality_history": quality_history,
            "quality_logits_history": quality_logits_history,
            "prototype_history": prototype_history,
            "sparse_history": sparse_history,
            "top_indices_history": top_indices_history,
            "top_weights_history": top_weights_history,
            "selected_mass_history": selected_mass_history,
            "feedback_gate_history": feedback_gate_history,
            "quality_logits": quality_logits,
            "noise_embedding": h_background,
            "final_noise_state": h_state,
            "prototype_token_count": protos.size(1),
            "effective_steps": effective_steps,
            "stopped_early": torch.zeros_like(effective_steps, dtype=torch.bool),
            "log_var": torch.zeros_like(clean),
        }
        return clean, quality, z_noise, aux

    @torch.no_grad()
    def _forward_adaptive(
        self,
        x: torch.Tensor,
        z_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        batch = x.size(0)
        h_background = self.noise_encoder(z_cond)
        h_state = h_background.clone()
        cur = x.clone()
        quality = x.new_zeros(batch, 1)
        quality_logits = x.new_zeros(batch, 1)
        effective_steps = torch.zeros(batch, device=x.device, dtype=torch.long)
        active = torch.arange(batch, device=x.device)
        mask_full = x.new_zeros(batch, 1, x.size(-1))
        z_noise_full = h_background.clone()

        prototype_probs = None
        sparse_probs = None
        top_indices = None
        top_weights = None
        selection_mask = None
        selected_mass = None
        hard_idx = None
        refine_history = []
        quality_history = []
        feedback_gate_history = []
        vq_commit_history = []
        vq_diversity_history = []
        prototype_token_count = 1

        for step in range(self.n_refine + 1):
            if active.numel() == 0:
                break

            protos, z_active, proto_aux = self._prototype_tokens(h_state[active])
            prototype_token_count = protos.size(1)
            mu_res, mask_active = self.backbone(cur[active], protos)
            candidate = cur[active] + mu_res
            q_active, logits_active = self.quality_head(
                x[active], candidate, mask_active
            )

            cur[active] = candidate
            quality[active] = q_active
            quality_logits[active] = logits_active
            mask_full[active] = mask_active
            z_noise_full[active] = z_active
            effective_steps[active] += 1

            if prototype_probs is None:
                k = proto_aux["prototype_probs"].size(1)
                m = proto_aux["top_indices"].size(1)
                prototype_probs = x.new_zeros(batch, k)
                sparse_probs = x.new_zeros(batch, k)
                top_indices = torch.zeros(
                    batch, m, device=x.device, dtype=torch.long
                )
                top_weights = x.new_zeros(batch, m)
                selection_mask = x.new_zeros(batch, k)
                selected_mass = x.new_zeros(batch)
                hard_idx = torch.zeros(batch, device=x.device, dtype=torch.long)

            prototype_probs[active] = proto_aux["prototype_probs"]
            sparse_probs[active] = proto_aux["sparse_probs"]
            top_indices[active] = proto_aux["top_indices"]
            top_weights[active] = proto_aux["top_weights"]
            selection_mask[active] = proto_aux["selection_mask"]
            selected_mass[active] = proto_aux["selected_mass"]
            if proto_aux.get("hard_idx") is not None:
                hard_idx[active] = proto_aux["hard_idx"]
            vq_commit_history.append(proto_aux.get("vq_commit"))
            vq_diversity_history.append(proto_aux.get("vq_diversity"))
            refine_history.append(cur.clone())
            quality_history.append(quality.clone())

            can_stop = step + 1 >= self.min_refine_steps
            stop_local = (
                q_active.squeeze(-1) >= self.stop_threshold
                if can_stop
                else torch.zeros(
                    active.numel(), device=x.device, dtype=torch.bool
                )
            )
            continue_local = ~stop_local
            if step == self.n_refine or not continue_local.any():
                active = active[continue_local]
                continue

            next_active = active[continue_local]
            if self.use_residual_feedback:
                residual_evidence = self._encode_residual_feedback(
                    x[next_active] - cur[next_active]
                )
                updated, gate = self.feedback_updater(
                    h_background[next_active],
                    h_state[next_active],
                    residual_evidence,
                )
                h_state[next_active] = updated
                gate_full = x.new_zeros(batch, self.z_dim)
                gate_full[next_active] = gate
                feedback_gate_history.append(gate_full)
            active = next_active

        stopped_early = effective_steps < (self.n_refine + 1)
        aux = {
            "det_mask": mask_full,
            "det_mask_used": mask_full,
            "cur_premask": cur,
            "prototype_probs": prototype_probs,
            "sparse_probs": sparse_probs,
            "top_indices": top_indices,
            "top_weights": top_weights,
            "selection_mask": selection_mask,
            "selected_mass": selected_mass,
            "vq_commit": self._mean_optional(vq_commit_history),
            "vq_diversity": self._mean_optional(vq_diversity_history),
            "hard_idx": hard_idx,
            "refine_history": refine_history,
            "quality_history": quality_history,
            "quality_logits_history": [],
            "prototype_history": [],
            "sparse_history": [],
            "top_indices_history": [],
            "top_weights_history": [],
            "selected_mass_history": [],
            "feedback_gate_history": feedback_gate_history,
            "quality_logits": quality_logits,
            "noise_embedding": h_background,
            "final_noise_state": h_state,
            "prototype_token_count": prototype_token_count,
            "effective_steps": effective_steps,
            "stopped_early": stopped_early,
            "log_var": torch.zeros_like(cur),
        }
        return cur, quality, z_noise_full, aux

    def forward(
        self,
        x: torch.Tensor,
        z_cond: torch.Tensor,
        adaptive_stop: bool | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        if adaptive_stop is None:
            adaptive_stop = (
                self.adaptive_inference
                and not self.training
                and self.quality_head is not None
            )
        if adaptive_stop:
            return self._forward_adaptive(x, z_cond)
        return self._forward_unrolled(x, z_cond)

    def load_v6_state_dict(self, state_dict: dict) -> Tuple[list, list]:
        """Load compatible V6 weights while leaving V7-only modules fresh."""
        incompatible = self.load_state_dict(state_dict, strict=False)
        return incompatible.missing_keys, incompatible.unexpected_keys


if __name__ == "__main__":
    model = NoiseAwareDenoiserV7(
        num_prototypes=16, top_m=4, n_refine=1, base_ch=16
    )
    x = torch.randn(2, 3, 1024)
    z = torch.randn(2, 3, 400)
    y, q, zn, aux = model(x, z)
    print(y.shape, q.shape, zn.shape, aux["top_indices"].shape)
