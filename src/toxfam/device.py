"""Canonical device detection for ToxFam."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
