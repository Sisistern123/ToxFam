"""Tests for model architectures: shapes, gradients, transfer learning."""

import torch

from toxfam.model.architectures import (
    ModularMLP,
    MultiInputMLP,
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
            embed_dim=1024,
            tax_dim=56,
            hidden_dims=[128],
            num_classes=20,
            tax_hidden_dim=16,
        )
        out = model(torch.randn(4, 1024), torch.randn(4, 56))
        assert out.shape == (4, 20)
