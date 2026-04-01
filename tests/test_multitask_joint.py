"""Tests for MultiTaskJointWrapper — Bayesian joint inference over both heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from toxfam.model.architectures import MultiTaskMLP


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
