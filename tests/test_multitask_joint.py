"""Tests for MultiTaskJointWrapper — Bayesian joint inference over both heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from toxfam.model.architectures import MultiTaskMLP, MultiTaskMultiInputMLP


def _make_model(input_dim=64, hidden_dims=None, num_family_classes=5):
    """Create a small MultiTaskMLP for testing."""
    if hidden_dims is None:
        hidden_dims = [32]
    return MultiTaskMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_family_classes=num_family_classes,
        num_binary_classes=2,
    )


def _make_dual_input_model(embed_dim=64, tax_dim=10, hidden_dims=None, num_family_classes=5):
    """Create a small MultiTaskMultiInputMLP for testing."""
    if hidden_dims is None:
        hidden_dims = [32]
    return MultiTaskMultiInputMLP(
        embed_dim=embed_dim,
        tax_dim=tax_dim,
        hidden_dims=hidden_dims,
        num_family_classes=num_family_classes,
        num_binary_classes=2,
    )


class TestJointProbabilityMath:
    """Verify that MultiTaskJointWrapper produces valid joint probabilities."""

    def test_joint_probs_sum_to_one(self):
        """P_joint sums to 1 for every sample in the batch."""
        from toxfam.training.strategies import MultiTaskJointWrapper

        model = _make_model()
        wrapper = MultiTaskJointWrapper(model, nontoxin_indices=[2])
        wrapper.eval()

        x = torch.randn(8, 64)
        log_probs = wrapper(x)
        probs = torch.exp(log_probs)

        sums = probs.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(8), atol=1e-5, rtol=1e-5)

    def test_nontoxin_prob_equals_binary_head(self):
        """P_joint(nontoxin) == P_bin(nontoxin) from the binary head."""
        from toxfam.training.strategies import MultiTaskJointWrapper

        model = _make_model()
        wrapper = MultiTaskJointWrapper(model, nontoxin_indices=[2])
        wrapper.eval()

        x = torch.randn(8, 64)
        with torch.no_grad():
            log_probs = wrapper(x)
            probs = torch.exp(log_probs)

            # Get binary head probabilities directly
            fam_logits, bin_logits = model(x)
            bin_probs = F.softmax(bin_logits, dim=1)

        # P_joint for nontoxin class (index 2) should equal P_bin(nontoxin) (index 0)
        torch.testing.assert_close(
            probs[:, 2], bin_probs[:, 0], atol=1e-5, rtol=1e-5
        )

    def test_toxic_probs_sum_equals_binary_head(self):
        """Sum of P_joint over toxic families == P_bin(toxic)."""
        from toxfam.training.strategies import MultiTaskJointWrapper

        model = _make_model()
        wrapper = MultiTaskJointWrapper(model, nontoxin_indices=[2])
        wrapper.eval()

        x = torch.randn(8, 64)
        with torch.no_grad():
            log_probs = wrapper(x)
            probs = torch.exp(log_probs)

            _, bin_logits = model(x)
            bin_probs = F.softmax(bin_logits, dim=1)

        # Toxic family indices are [0, 1, 3, 4] (all except 2)
        toxic_sum = probs[:, [0, 1, 3, 4]].sum(dim=1)
        torch.testing.assert_close(
            toxic_sum, bin_probs[:, 1], atol=1e-5, rtol=1e-5
        )

    def test_toxic_family_ranking_preserved(self):
        """Within toxic families, relative ordering matches the family head."""
        from toxfam.training.strategies import MultiTaskJointWrapper

        model = _make_model()
        wrapper = MultiTaskJointWrapper(model, nontoxin_indices=[2])
        wrapper.eval()

        x = torch.randn(8, 64)
        with torch.no_grad():
            log_probs = wrapper(x)
            probs = torch.exp(log_probs)

            fam_logits, _ = model(x)
            fam_probs = F.softmax(fam_logits, dim=1)

        # Toxic indices: [0, 1, 3, 4]
        toxic_idx = [0, 1, 3, 4]
        joint_toxic = probs[:, toxic_idx]
        fam_toxic = fam_probs[:, toxic_idx]

        # The ranking (argsort) within toxic families should be the same
        # because P_joint(fam_i) = P_bin(toxic) * P_fam(fam_i) / sum_toxic
        # The P_bin(toxic) / sum_toxic factor is constant across toxic families
        # for a given sample, so ranking is preserved.
        joint_ranks = joint_toxic.argsort(dim=1, descending=True)
        fam_ranks = fam_toxic.argsort(dim=1, descending=True)
        assert torch.equal(joint_ranks, fam_ranks)


class TestNumericalStability:
    """Verify wrapper handles extreme inputs without NaN or Inf."""

    def test_extreme_logits_no_nan(self):
        """Model outputs extreme logits — wrapper must not produce NaN/Inf."""
        from toxfam.training.strategies import MultiTaskJointWrapper

        model = _make_model(num_family_classes=5)

        class _ExtremeModel(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.family_head = base.family_head

            def forward(self, x):
                fam, bin_ = self.base(x)
                return fam * 100, bin_ * 100

        extreme = _ExtremeModel(model)
        wrapper = MultiTaskJointWrapper(extreme, nontoxin_indices=[2])

        x = torch.randn(8, 64)
        wrapper.eval()
        with torch.no_grad():
            log_probs = wrapper(x)

        assert not torch.isnan(log_probs).any(), "NaN detected in output"
        assert not torch.isinf(log_probs).any(), "Inf detected in output"

        probs = F.softmax(log_probs, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-4)

    def test_single_sample_batch(self):
        """Wrapper works with batch size 1."""
        from toxfam.training.strategies import MultiTaskJointWrapper

        model = _make_model(num_family_classes=5)
        wrapper = MultiTaskJointWrapper(model, nontoxin_indices=[2])

        x = torch.randn(1, 64)
        wrapper.eval()
        with torch.no_grad():
            log_probs = wrapper(x)
            probs = F.softmax(log_probs, dim=1)

        assert probs.shape == (1, 5)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


class TestDownstreamCompatibility:
    """Verify wrapper output works with CrossEntropyLoss and ModelWithTemperature."""

    def test_cross_entropy_loss_compatible(self):
        """CE loss on wrapper output equals -log(joint_prob[true_class])."""
        from toxfam.training.strategies import MultiTaskJointWrapper

        model = _make_model(num_family_classes=5)
        wrapper = MultiTaskJointWrapper(model, nontoxin_indices=[2])

        x = torch.randn(4, 64)
        labels = torch.tensor([0, 1, 2, 3])

        wrapper.eval()
        with torch.no_grad():
            log_probs = wrapper(x)
            joint_probs = F.softmax(log_probs, dim=1)

            ce_loss = F.cross_entropy(log_probs, labels)
            manual_nll = -torch.log(
                joint_probs[range(4), labels].clamp(min=1e-8)
            ).mean()

        assert torch.allclose(ce_loss, manual_nll, atol=1e-4), (
            f"CE loss ({ce_loss:.6f}) != manual NLL ({manual_nll:.6f})"
        )

    def test_temperature_scaling_integration(self):
        """ModelWithTemperature wrapping the joint wrapper produces valid probs."""
        from toxfam.model.calibration import ModelWithTemperature
        from toxfam.training.strategies import MultiTaskJointWrapper

        model = _make_model(num_family_classes=5)
        wrapper = MultiTaskJointWrapper(model, nontoxin_indices=[2])

        device = torch.device("cpu")
        scaled = ModelWithTemperature(wrapper, device)
        scaled.temperature.data.fill_(2.0)

        x = torch.randn(8, 64)
        scaled.eval()
        with torch.no_grad():
            out = scaled(x)
            probs = F.softmax(out, dim=1)

        assert probs.shape == (8, 5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)

    def test_temperature_gradient_flows(self):
        """Temperature parameter receives gradients through the joint wrapper."""
        from toxfam.model.calibration import ModelWithTemperature
        from toxfam.training.strategies import MultiTaskJointWrapper

        model = _make_model(num_family_classes=5)
        wrapper = MultiTaskJointWrapper(model, nontoxin_indices=[2])

        device = torch.device("cpu")
        scaled = ModelWithTemperature(wrapper, device)

        x = torch.randn(4, 64)
        labels = torch.tensor([0, 1, 2, 3])

        out = scaled(x)
        loss = F.cross_entropy(out, labels)
        loss.backward()

        assert scaled.temperature.grad is not None, (
            "Temperature should receive gradients"
        )
        assert scaled.temperature.grad.abs().item() > 0, (
            "Temperature gradient should be non-zero"
        )


class TestDualInputJointWrapper:
    """Verify joint wrapper works with MultiTaskMultiInputMLP (emb + tax)."""

    def test_joint_probs_sum_to_one(self):
        from toxfam.training.strategies import MultiTaskJointWrapper

        model = _make_dual_input_model()
        wrapper = MultiTaskJointWrapper(model, nontoxin_indices=[2])
        wrapper.eval()

        emb = torch.randn(8, 64)
        tax = torch.randn(8, 10)
        with torch.no_grad():
            log_probs = wrapper(emb, tax)
            probs = F.softmax(log_probs, dim=1)

        sums = probs.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(8), atol=1e-5, rtol=1e-5)

    def test_binary_head_controls_boundary(self):
        """P_joint(nontoxin) == P_bin(nontoxin) for dual-input model."""
        from toxfam.training.strategies import MultiTaskJointWrapper

        model = _make_dual_input_model()
        nontoxin_idx = 2
        wrapper = MultiTaskJointWrapper(model, nontoxin_indices=[nontoxin_idx])
        wrapper.eval()

        emb = torch.randn(8, 64)
        tax = torch.randn(8, 10)
        with torch.no_grad():
            _, bin_logits = model(emb, tax)
            p_bin_nontoxin = F.softmax(bin_logits, dim=1)[:, 0]

            log_probs = wrapper(emb, tax)
            joint_probs = F.softmax(log_probs, dim=1)
            p_joint_nontoxin = joint_probs[:, nontoxin_idx]

        torch.testing.assert_close(
            p_joint_nontoxin, p_bin_nontoxin, atol=1e-5, rtol=1e-5
        )

    def test_temperature_scaling_dual_input(self):
        from toxfam.model.calibration import ModelWithTemperature
        from toxfam.training.strategies import MultiTaskJointWrapper

        model = _make_dual_input_model()
        wrapper = MultiTaskJointWrapper(model, nontoxin_indices=[2])

        device = torch.device("cpu")
        scaled = ModelWithTemperature(wrapper, device)
        scaled.temperature.data.fill_(2.0)
        scaled.eval()

        emb = torch.randn(4, 64)
        tax = torch.randn(4, 10)
        with torch.no_grad():
            out = scaled(emb, tax)
            probs = F.softmax(out, dim=1)

        assert probs.shape == (4, 5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
