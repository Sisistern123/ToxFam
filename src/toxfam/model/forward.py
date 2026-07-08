"""Input-shape-agnostic forward pass shared by training, calibration, and eval.

Lives in the model layer (a torch-only leaf) so that ``model.calibration`` and
``evaluation.binary`` no longer import from ``training.trainer`` — keeping the
dependency direction strictly training/evaluation -> model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def forward_model(
    model: nn.Module,
    features: torch.Tensor | tuple[torch.Tensor, ...],
    device: torch.device | str,
) -> torch.Tensor:
    """Handle single-input (Tensor) or multi-input ((emb, tax)) forwarding."""
    if isinstance(features, (tuple, list)):
        features = [f.to(device) for f in features]
        return model(*features)
    else:
        return model(features.to(device))
