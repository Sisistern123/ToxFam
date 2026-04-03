"""Tests for toxfam.model.calibration."""

from __future__ import annotations

import pytest
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


def test_set_temperature_with_data():
    base = ModularMLP(input_dim=16, hidden_dims=[8], num_classes=3)
    device = torch.device("cpu")
    scaled = ModelWithTemperature(base, device)

    # Verify initial temperature
    assert scaled.temperature.item() == pytest.approx(1.5)

    # Create a small DataLoader with random data
    xs = torch.randn(10, 16)
    ys = torch.randint(0, 3, (10,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xs, ys), batch_size=5
    )

    scaled.set_temperature(loader)

    # Temperature should have changed from its initial value
    assert scaled.temperature.item() != pytest.approx(1.5)


def test_set_temperature_empty_loader_raises():
    base = ModularMLP(input_dim=16, hidden_dims=[8], num_classes=3)
    device = torch.device("cpu")
    scaled = ModelWithTemperature(base, device)

    empty_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.empty(0, 16), torch.empty(0, dtype=torch.long)
        ),
        batch_size=1,
    )

    with pytest.raises(ValueError, match="empty"):
        scaled.set_temperature(empty_loader)


def test_ece_loss_returns_scalar():
    from toxfam.model.calibration import _ECELoss

    ece = _ECELoss(n_bins=10)
    logits = torch.randn(20, 3)
    labels = torch.randint(0, 3, (20,))
    result = ece(logits, labels)

    assert result.ndim <= 1 and result.numel() == 1  # scalar-like tensor
