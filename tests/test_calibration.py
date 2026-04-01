"""Tests for toxfam.model.calibration."""

from __future__ import annotations

import torch

from toxfam.model.architectures import ModularMLP
from toxfam.model.calibration import ModelWithTemperature


def test_forward_applies_temperature():
    base = ModularMLP(input_dim=64, hidden_dims=[32], num_classes=5)
    device = torch.device("cpu")
    scaled = ModelWithTemperature(base, device)

    x = torch.randn(2, 64)
    out = scaled(x)
    assert out.shape == (2, 5)


def test_temperature_parameter_exists():
    base = ModularMLP(input_dim=64, hidden_dims=[32], num_classes=5)
    device = torch.device("cpu")
    scaled = ModelWithTemperature(base, device)

    assert hasattr(scaled, "temperature")
    assert isinstance(scaled.temperature, torch.nn.Parameter)


def test_temperature_scaling_effect():
    base = ModularMLP(input_dim=64, hidden_dims=[32], num_classes=5)
    device = torch.device("cpu")
    scaled = ModelWithTemperature(base, device)

    x = torch.randn(2, 64)

    # Get base logits
    base.eval()
    with torch.no_grad():
        base_logits = base(x)

    # With temperature > 1, logits should be smaller in magnitude
    scaled.temperature.data.fill_(2.0)
    scaled.eval()
    with torch.no_grad():
        scaled_logits = scaled(x)

    assert torch.allclose(scaled_logits, base_logits / 2.0, atol=1e-5)
