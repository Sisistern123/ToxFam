"""Tests for FocalLoss."""

from __future__ import annotations

import torch
import pytest

from toxfam.training.trainer import FocalLoss


@pytest.fixture
def sample_logits():
    """2 samples, 3 classes."""
    return torch.tensor([[2.0, 0.5, -1.0], [0.1, 2.5, 0.3]])


@pytest.fixture
def sample_targets():
    return torch.tensor([0, 1])


class TestFocalLoss:
    def test_output_shape(self, sample_logits, sample_targets):
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(sample_logits, sample_targets)
        assert loss.ndim == 0  # scalar

    def test_gamma_zero_equals_cross_entropy(self, sample_logits, sample_targets):
        """With gamma=0, focal loss should equal cross-entropy."""
        focal = FocalLoss(gamma=0.0)
        ce = torch.nn.CrossEntropyLoss()
        fl_val = focal(sample_logits, sample_targets)
        ce_val = ce(sample_logits, sample_targets)
        torch.testing.assert_close(fl_val, ce_val, atol=1e-5, rtol=1e-5)

    def test_higher_gamma_lower_easy_loss(self, sample_logits, sample_targets):
        """Higher gamma should produce lower loss for well-classified samples."""
        loss_g0 = FocalLoss(gamma=0.0)(sample_logits, sample_targets)
        loss_g2 = FocalLoss(gamma=2.0)(sample_logits, sample_targets)
        # gamma=2 should down-weight easy examples, giving lower loss
        assert loss_g2 < loss_g0

    def test_with_weights(self, sample_logits, sample_targets):
        weights = torch.tensor([1.0, 2.0, 0.5])
        loss_fn = FocalLoss(weight=weights, gamma=2.0)
        loss = loss_fn(sample_logits, sample_targets)
        assert loss.item() > 0

    def test_reduction_none(self, sample_logits, sample_targets):
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        loss = loss_fn(sample_logits, sample_targets)
        assert loss.shape == (2,)

    def test_reduction_sum(self, sample_logits, sample_targets):
        loss_fn = FocalLoss(gamma=2.0, reduction="sum")
        loss = loss_fn(sample_logits, sample_targets)
        assert loss.ndim == 0
