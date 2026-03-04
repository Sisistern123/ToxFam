"""Tests for toxfam.model.architectures."""

from __future__ import annotations

import torch

from toxfam.model.architectures import ModularMLP, MultiInputMLP


class TestModularMLP:
    def test_forward_shape(self):
        model = ModularMLP(input_dim=1024, hidden_dims=[256, 128], num_classes=10)
        x = torch.randn(4, 1024)
        out = model(x)
        assert out.shape == (4, 10)

    def test_single_hidden_dim(self):
        model = ModularMLP(input_dim=64, hidden_dims=[32], num_classes=5)
        x = torch.randn(2, 64)
        out = model(x)
        assert out.shape == (2, 5)

    def test_int_hidden_dims(self):
        model = ModularMLP(input_dim=64, hidden_dims=32, num_classes=5)
        x = torch.randn(2, 64)
        out = model(x)
        assert out.shape == (2, 5)

    def test_dropout_attribute(self):
        model = ModularMLP(
            input_dim=64, hidden_dims=[32], num_classes=5, dropout=0.5
        )
        assert model.dropout_rate == 0.5


class TestMultiInputMLP:
    def test_forward_shape(self):
        model = MultiInputMLP(
            embed_dim=1024, tax_dim=56, hidden_dims=[256, 128], num_classes=10
        )
        emb = torch.randn(4, 1024)
        tax = torch.randn(4, 56)
        out = model(emb, tax)
        assert out.shape == (4, 10)

    def test_single_hidden_dim(self):
        model = MultiInputMLP(
            embed_dim=64, tax_dim=8, hidden_dims=[32], num_classes=3
        )
        emb = torch.randn(2, 64)
        tax = torch.randn(2, 8)
        out = model(emb, tax)
        assert out.shape == (2, 3)

    def test_custom_tax_hidden_dim(self):
        model = MultiInputMLP(
            embed_dim=64,
            tax_dim=16,
            hidden_dims=[32],
            num_classes=5,
            tax_hidden_dim=4,
        )
        emb = torch.randn(1, 64)
        tax = torch.randn(1, 16)
        out = model(emb, tax)
        assert out.shape == (1, 5)
