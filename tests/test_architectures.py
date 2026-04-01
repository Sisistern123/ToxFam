"""Tests for all model architectures: shapes, gradients, transfer learning."""

import torch

from toxfam.model.architectures import (
    ModularMLP,
    MultiInputMLP,
    MultiTaskMLP,
    MultiTaskMultiInputMLP,
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


class TestMultiTaskMultiInputMLP:
    def test_forward_shapes(self):
        model = MultiTaskMultiInputMLP(
            embed_dim=1024, tax_dim=50, hidden_dims=[256, 128],
            num_family_classes=38, num_binary_classes=2,
        )
        emb = torch.randn(8, 1024)
        tax = torch.randn(8, 50)
        fam_out, bin_out = model(emb, tax)
        assert fam_out.shape == (8, 38)
        assert bin_out.shape == (8, 2)

    def test_taxonomy_branch_exists(self):
        model = MultiTaskMultiInputMLP(
            embed_dim=512, tax_dim=50, hidden_dims=[64],
            num_family_classes=10,
        )
        assert hasattr(model, "tax_net")
        assert hasattr(model, "shared")
        assert hasattr(model, "family_head")
        assert hasattr(model, "binary_head")

    def test_shared_backbone_gets_gradients(self):
        model = MultiTaskMultiInputMLP(
            embed_dim=512, tax_dim=50, hidden_dims=[128],
            num_family_classes=10,
        )
        emb = torch.randn(4, 512)
        tax = torch.randn(4, 50)
        fam_out, bin_out = model(emb, tax)
        loss = fam_out.sum() + bin_out.sum()
        loss.backward()
        assert model.shared[0].weight.grad is not None
        assert model.tax_net[0].weight.grad is not None

    def test_custom_tax_hidden(self):
        model = MultiTaskMultiInputMLP(
            embed_dim=256, tax_dim=50, hidden_dims=[64],
            num_family_classes=5, tax_hidden_dim=16,
        )
        emb = torch.randn(2, 256)
        tax = torch.randn(2, 50)
        fam, bn = model(emb, tax)
        assert fam.shape == (2, 5)
        assert bn.shape == (2, 2)
