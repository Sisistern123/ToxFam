"""Tests for toxfam.device."""

import torch

from toxfam.device import get_device


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_is_known_type():
    device = get_device()
    assert device.type in ("cpu", "cuda", "mps")
