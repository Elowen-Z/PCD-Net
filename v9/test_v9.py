"""Fast CPU checks for dense routing, attention, and adaptation scope."""

import math

import torch

from v9.model_v9 import NoiseAwareDenoiserV9, RoutingPriorCrossAttention, SoftPrototypeRouter
from v9.transfer_v9 import configure_adaptation


def test_soft_router_uses_all_prototypes():
    torch.manual_seed(1)
    router = SoftPrototypeRouter(temperature=0.5, routing_strength=1.2)
    background = torch.randn(3, 8, requires_grad=True)
    prototypes = torch.randn(5, 8, requires_grad=True)
    alpha, cosine, bias = router(background, prototypes)
    assert alpha.shape == (3, 5)
    assert cosine.shape == bias.shape == alpha.shape
    assert torch.all(alpha > 0)
    assert torch.allclose(alpha.sum(-1), torch.ones(3), atol=1e-6)
    (alpha[:, 0].sum() + bias[:, 1].sum()).backward()
    assert background.grad is not None and prototypes.grad is not None


def test_uniform_routing_ablation():
    router = SoftPrototypeRouter()
    alpha, _, bias = router(torch.randn(2, 4), torch.randn(6, 4), use_routing=False)
    assert torch.allclose(alpha, torch.full_like(alpha, 1.0 / 6.0))
    assert torch.allclose(bias.std(-1), torch.zeros(2), atol=1e-6)


def test_routing_prior_changes_attention():
    torch.manual_seed(2)
    module = RoutingPriorCrossAttention(c_dim=8, p_dim=4, num_heads=2, dropout=0.0).eval()
    waveform = torch.randn(1, 8, 7)
    prototypes = torch.randn(1, 3, 4)
    neutral = torch.cat([prototypes, torch.zeros(1, 3, 1)], dim=-1)
    preferred = torch.cat([prototypes, torch.tensor([[[0.0], [0.0], [8.0]]])], dim=-1)
    module(waveform, neutral)
    neutral_attention = module.last_attention.clone()
    module(waveform, preferred)
    preferred_attention = module.last_attention
    assert preferred_attention[..., 2].mean() > neutral_attention[..., 2].mean()


def test_model_and_environment_scope():
    torch.manual_seed(3)
    model = NoiseAwareDenoiserV9(
        z_dim=16, num_prototypes=6, num_heads=4,
        base_ch=4, use_quality_head=True,
    )
    noisy = torch.randn(2, 3, 256)
    background = torch.randn(2, 3, 400)
    output, quality, _, aux = model(noisy, background, adaptive_stop=False)
    assert output.shape == noisy.shape and quality.shape == (2, 1)
    assert aux["routing_alpha"].shape == (2, 6)
    assert aux["prototype_token_count"] == 6
    assert len(aux["refine_history"]) == 1
    assert aux["effective_steps"].tolist() == [1, 1]
    assert model.n_refine == 0
    assert not model.use_residual_feedback
    assert not model.adaptive_inference
    assert torch.allclose(aux["routing_alpha"].sum(-1), torch.ones(2), atol=1e-5)
    assert torch.isfinite(aux["routing_entropy"]).all()
    assert torch.all(aux["effective_prototypes"] >= 1.0)
    assert torch.all(aux["effective_prototypes"] <= 6.0 + 1e-5)

    configure_adaptation(model, "environment")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert any(name.startswith("noise_encoder.") for name in trainable)
    assert "vq.codebook" in trainable
    assert any(name.startswith("router.") for name in trainable)
    assert not any(name.startswith("backbone.enc") for name in trainable)
    assert not any(name.startswith("backbone.dec") for name in trainable)
    assert not any(name.startswith("quality_head.") for name in trainable)

    configure_adaptation(model, "decoder_only")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    decoder_prefixes = tuple(
        f"backbone.{name}." for name in ("dec4", "dec3", "dec2", "dec1", "dec0")
    ) + ("backbone.out_head.",)
    assert trainable
    assert all(name.startswith(decoder_prefixes) for name in trainable)
    assert any(name.startswith("backbone.dec4.") for name in trainable)
    assert any(name.startswith("backbone.dec0.") for name in trainable)
    assert any(name.startswith("backbone.out_head.") for name in trainable)

    configure_adaptation(model, "environment_decoder")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    env_decoder_prefixes = (
        "noise_encoder.", "vq.", "router.",
        "backbone.dec4.", "backbone.dec3.", "backbone.dec2.",
        "backbone.dec1.", "backbone.dec0.", "backbone.out_head.",
    )
    assert trainable
    assert all(name.startswith(env_decoder_prefixes) for name in trainable)
    assert any(name.startswith("noise_encoder.") for name in trainable)
    assert "vq.codebook" in trainable
    assert any(name.startswith("router.") for name in trainable)
    assert any(name.startswith("backbone.dec4.") for name in trainable)
    assert any(name.startswith("backbone.out_head.") for name in trainable)
    assert not any(name.startswith("backbone.enc") for name in trainable)
    assert not any(name.startswith("backbone.bn_xattn.") for name in trainable)


if __name__ == "__main__":
    test_soft_router_uses_all_prototypes()
    test_uniform_routing_ablation()
    test_routing_prior_changes_attention()
    test_model_and_environment_scope()
    print("V9 smoke tests passed.")
