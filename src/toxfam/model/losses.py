"""Custom loss functions for ToxFam."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma=0 this reduces to weighted cross-entropy.

    Args:
        weight: Per-class weights tensor (same as CrossEntropyLoss).
        gamma: Focusing parameter. Higher values down-weight easy examples more.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("weight", weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Raw logits of shape (N, C).
            targets: Class indices of shape (N,).
        """
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight, reduction="none"
        )
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
