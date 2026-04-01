"""Model configuration for deterministic architecture reconstruction.

Saved as ``model_config.json`` alongside model checkpoints during training.
At inference time, the config is loaded to reconstruct the exact architecture
without fragile state_dict key parsing.

Follows the HuggingFace pattern: config.json + model weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch.nn as nn
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Minimal config to reconstruct a trained model for inference."""

    architecture: Literal["ModularMLP", "MultiInputMLP"]
    embedding_dim: int
    hidden_dims: list[int]
    num_classes: int
    dropout: float = 0.3
    # MultiInputMLP-specific
    tax_dim: int | None = None
    tax_hidden_dim: int = 8

    def save(self, path: str | Path) -> None:
        """Write config to JSON file."""
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        """Load config from JSON file."""
        return cls.model_validate_json(Path(path).read_text())

    def build_model(self) -> nn.Module:
        """Construct the nn.Module from this config."""
        from toxfam.model.architectures import ModularMLP, MultiInputMLP

        if self.architecture == "MultiInputMLP":
            return MultiInputMLP(
                embed_dim=self.embedding_dim,
                tax_dim=self.tax_dim,
                hidden_dims=self.hidden_dims,
                num_classes=self.num_classes,
                dropout=self.dropout,
                tax_hidden_dim=self.tax_hidden_dim,
            )
        return ModularMLP(
            input_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            num_classes=self.num_classes,
            dropout=self.dropout,
        )
