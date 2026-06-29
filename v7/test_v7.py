"""Fast CPU smoke tests for V7 model and loss."""

import torch

from v7.loss_v7 import DenoiserLossV7
from v7.model_v7 import NoiseAwareDenoiserV7, SparsePrototypeSelector
from v7.transfer_staged_v7 import (
    DomainDiscriminator,
    configure_stage,
    gradient_reverse,
    task_loss,
)


def test_sparse_selector():
    selector = SparsePrototypeSelector(top_m=2)
    probs = torch.tensor([[0.1, 0.6, 0.2, 0.1]], requires_grad=True)
    codebook = torch.randn(4, 8)
    tokens, aux = selector(probs, codebook)
    assert tokens.shape == (1, 2, 8)
    assert aux["top_indices"].tolist() == [[1, 2]]
    assert torch.allclose(aux["sparse_probs"].sum(-1), torch.ones(1))
    tokens.sum().backward()
    assert probs.grad is not None


def test_model_and_loss():
    torch.manual_seed(7)
    model = NoiseAwareDenoiserV7(
        z_dim=32,
        num_prototypes=8,
        top_m=3,
        num_heads=4,
        n_refine=1,
        base_ch=8,
    )
    noisy = torch.randn(2, 3, 1024)
    target = torch.randn_like(noisy)
    z_cond = torch.randn(2, 3, 400)
    valid_mask = torch.ones(2, 1024)
    has_target = torch.ones(2, dtype=torch.bool)

    pred, quality, _, aux = model(noisy, z_cond)
    assert pred.shape == noisy.shape
    assert quality.shape == (2, 1)
    assert aux["top_indices"].shape == (2, 3)
    assert aux["prototype_token_count"] == 3
    assert len(aux["refine_history"]) == 2
    assert len(aux["quality_logits_history"]) == 2
    assert len(aux["feedback_gate_history"]) == 1

    criterion = DenoiserLossV7()
    loss, detail = criterion(
        pred,
        target,
        valid_mask,
        has_target,
        det_mask=aux["det_mask"],
        quality=quality,
        quality_logits=aux["quality_logits"],
        prototype_probs=aux["prototype_probs"],
        sparse_probs=aux["sparse_probs"],
        vq_commit=aux["vq_commit"],
        vq_diversity=aux["vq_diversity"],
        refine_history=aux["refine_history"],
        quality_logits_history=aux["quality_logits_history"],
    )
    assert torch.isfinite(loss)
    assert "sparse_entropy" in detail
    assert "quality" in detail
    assert "intermediate" in detail
    loss.backward()


def test_quality_driven_early_stop():
    torch.manual_seed(11)
    model = NoiseAwareDenoiserV7(
        z_dim=32,
        num_prototypes=8,
        top_m=2,
        num_heads=4,
        n_refine=3,
        base_ch=8,
        stop_threshold=0.01,
        min_refine_steps=1,
    ).eval()
    noisy = torch.randn(3, 3, 512)
    z_cond = torch.randn(3, 3, 400)
    pred, quality, _, aux = model(noisy, z_cond)
    assert pred.shape == noisy.shape
    assert quality.shape == (3, 1)
    assert aux["effective_steps"].tolist() == [1, 1, 1]
    assert aux["stopped_early"].all()
    assert len(aux["refine_history"]) == 1


def test_transfer_alignment_step():
    torch.manual_seed(17)
    model = NoiseAwareDenoiserV7(
        z_dim=32,
        num_prototypes=8,
        top_m=2,
        num_heads=4,
        n_refine=0,
        base_ch=8,
    )
    discriminator = DomainDiscriminator(channels=64, hidden=32)
    trainable = configure_stage(model, stage=2)
    criterion = DenoiserLossV7(alpha_intermediate=0.0)
    feature_store = {}
    hook = model.backbone.bn_xattn.register_forward_hook(
        lambda _module, _inputs, output: feature_store.update(
            bottleneck=output
        )
    )

    def make_batch():
        return {
            "x": torch.randn(2, 3, 512),
            "y_clean": torch.randn(2, 3, 512),
            "z_cond": torch.randn(2, 3, 400),
            "valid_mask": torch.ones(2, 512),
            "has_target": torch.ones(2, dtype=torch.bool),
        }

    try:
        target = make_batch()
        source = make_batch()
        target_output = model(
            target["x"], target["z_cond"], adaptive_stop=False
        )
        target_feature = feature_store["bottleneck"]
        target_task, _ = task_loss(criterion, target_output, target)

        source_output = model(
            source["x"], source["z_cond"], adaptive_stop=False
        )
        source_feature = feature_store["bottleneck"]
        source_task, _ = task_loss(criterion, source_output, source)

        target_logits = discriminator(gradient_reverse(target_feature, 0.5))
        source_logits = discriminator(gradient_reverse(source_feature, 0.5))
        domain = 0.5 * (
            torch.nn.functional.binary_cross_entropy_with_logits(
                target_logits, torch.ones_like(target_logits)
            )
            + torch.nn.functional.binary_cross_entropy_with_logits(
                source_logits, torch.zeros_like(source_logits)
            )
        )
        loss = target_task + 0.2 * source_task + 0.2 * domain
        assert torch.isfinite(loss)
        loss.backward()
        assert any(parameter.grad is not None for parameter in trainable)
        assert any(
            parameter.grad is not None
            for parameter in discriminator.parameters()
        )
    finally:
        hook.remove()


if __name__ == "__main__":
    test_sparse_selector()
    test_model_and_loss()
    test_quality_driven_early_stop()
    test_transfer_alignment_step()
    print("V7 smoke tests passed.")
