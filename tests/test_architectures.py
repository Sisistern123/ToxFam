"""Tests for all model architectures: shapes, gradients, transfer learning."""

import torch
import pytest

from toxfam.model.architectures import (
    HierarchicalMLP,
    ModularMLP,
    MultiInputMLP,
    MultiTaskMLP,
)


class TestModularMLP:
    def test_forward_shape(self):
        model = ModularMLP(input_dim=1024, hidden_dims=[256, 128], num_classes=38)
        x = torch.randn(8, 1024)
        out = model(x)
        assert out.shape == (8, 38)

    def test_single_hidden_dim(self):
        model = ModularMLP(input_dim=512, hidden_dims=[64], num_classes=10)
        x = torch.randn(4, 512)
        assert model(x).shape == (4, 10)

    def test_int_hidden_dims(self):
        model = ModularMLP(input_dim=256, hidden_dims=128, num_classes=5)
        x = torch.randn(2, 256)
        assert model(x).shape == (2, 5)

    def test_projector_backbone_separation(self):
        model = ModularMLP(input_dim=1024, hidden_dims=[256, 128], num_classes=38)
        assert hasattr(model, "projector")
        assert hasattr(model, "backbone")
        # Projector output should feed into backbone
        x = torch.randn(4, 1024)
        proj_out = model.projector(x)
        assert proj_out.shape == (4, 256)


class TestMultiInputMLP:
    def test_forward_shape(self):
        model = MultiInputMLP(
            embed_dim=1024, tax_dim=50, hidden_dims=[256], num_classes=38
        )
        emb = torch.randn(8, 1024)
        tax = torch.randn(8, 50)
        out = model(emb, tax)
        assert out.shape == (8, 38)

    def test_custom_tax_hidden(self):
        model = MultiInputMLP(
            embed_dim=1024, tax_dim=56, hidden_dims=[128],
            num_classes=20, tax_hidden_dim=16,
        )
        out = model(torch.randn(4, 1024), torch.randn(4, 56))
        assert out.shape == (4, 20)


class TestHierarchicalMLP:
    @pytest.fixture
    def projector_state(self):
        """Get a projector state dict from a trained ModularMLP."""
        base = ModularMLP(input_dim=1024, hidden_dims=[256], num_classes=38)
        return base.projector.state_dict()

    def test_forward_shape(self, projector_state):
        model = HierarchicalMLP(
            projector_state=projector_state,
            projector_out_dim=256,
            hidden_dim=64,
            num_binary_classes=2,
            freeze_backbone=True,
        )
        x = torch.randn(8, 1024)
        out = model(x)
        assert out.shape == (8, 2)

    def test_frozen_backbone_no_grad(self, projector_state):
        model = HierarchicalMLP(
            projector_state=projector_state,
            projector_out_dim=256,
            hidden_dim=64,
            num_binary_classes=2,
            freeze_backbone=True,
        )
        for param in model.backbone.parameters():
            assert not param.requires_grad
        # Head should still have gradients
        for param in model.head.parameters():
            assert param.requires_grad

    def test_unfrozen_backbone_has_grad(self, projector_state):
        model = HierarchicalMLP(
            projector_state=projector_state,
            projector_out_dim=256,
            hidden_dim=64,
            num_binary_classes=2,
            freeze_backbone=False,
        )
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_projector_weights_loaded(self, projector_state):
        """Verify the backbone actually loads the projector weights."""
        model = HierarchicalMLP(
            projector_state=projector_state,
            projector_out_dim=256,
            hidden_dim=64,
        )
        # The linear layer weight should match
        assert torch.allclose(
            model.backbone[0].weight, projector_state["0.weight"]
        )

    def test_backward_only_updates_head_when_frozen(self, projector_state):
        model = HierarchicalMLP(
            projector_state=projector_state,
            projector_out_dim=256,
            hidden_dim=64,
            freeze_backbone=True,
        )
        backbone_weight_before = model.backbone[0].weight.clone()

        x = torch.randn(4, 1024)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Backbone should not have changed
        assert torch.equal(model.backbone[0].weight, backbone_weight_before)
        # Head should have gradients
        assert model.head[0].weight.grad is not None


class TestMultiTaskMLP:
    def test_forward_shapes(self):
        model = MultiTaskMLP(
            input_dim=1024,
            hidden_dims=[256, 128],
            num_family_classes=38,
            num_binary_classes=2,
        )
        x = torch.randn(8, 1024)
        fam_out, bin_out = model(x)
        assert fam_out.shape == (8, 38)
        assert bin_out.shape == (8, 2)

    def test_single_hidden_dim(self):
        model = MultiTaskMLP(
            input_dim=512,
            hidden_dims=[64],
            num_family_classes=10,
            num_binary_classes=2,
        )
        x = torch.randn(4, 512)
        fam_out, bin_out = model(x)
        assert fam_out.shape == (4, 10)
        assert bin_out.shape == (4, 2)

    def test_shared_backbone_is_shared(self):
        """Both heads should share the same backbone parameters."""
        model = MultiTaskMLP(
            input_dim=1024, hidden_dims=[256], num_family_classes=38,
        )
        # Forward pass and backward — shared params get gradients from both
        x = torch.randn(4, 1024)
        fam_out, bin_out = model(x)
        loss = fam_out.sum() + bin_out.sum()
        loss.backward()
        # Shared backbone should have gradients
        assert model.shared[0].weight.grad is not None

    def test_int_hidden_dims_coerced(self):
        model = MultiTaskMLP(
            input_dim=256, hidden_dims=64, num_family_classes=5,
        )
        x = torch.randn(2, 256)
        fam, bn = model(x)
        assert fam.shape == (2, 5)
        assert bn.shape == (2, 2)
