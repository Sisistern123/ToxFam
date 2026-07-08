from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class TrainConfig(BaseModel):
    """Pydantic model for all training hyperparameters and paths."""

    input_csv: Path
    h5_paths: list[Path] = Field(default_factory=list)
    h5_paths_glob: str | None = None
    h5_path: str | None = None
    tax_h5_path: Path | None = None
    output_dir: Path

    training_strategy: Literal["standard", "combined", "binary"]

    # Architecture
    embedding_dim: int = 1024
    tax_dim: int = 50
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 256])
    dropout: float = 0.3

    # Training
    batch_size: int = 64
    num_epochs: int = 200
    learning_rate: float = 0.0001
    early_stopping_patience: int = 10
    early_stopping_metric: Literal["loss", "mcc"] = "mcc"
    max_grad_norm: float | None = 1.0
    seed: int | None = 42

    # Optimizer
    optimizer: Literal["adam", "adamw"] = "adamw"
    weight_decay: float = 1e-2

    # LR Scheduler
    lr_scheduler: Literal["none", "cosine"] = "cosine"
    warmup_epochs: int = 5

    # Loss
    use_focal_loss: bool = False
    focal_loss_gamma: float = 2.0
    label_smoothing: float = 0.0

    # wandb
    wandb_project: str = "toxfam"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None

    model_config = {"extra": "ignore"}  # Pydantic's model config, not ML model config

    @field_validator("dropout")
    @classmethod
    def _check_dropout(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {v}")
        return v

    @field_validator("learning_rate")
    @classmethod
    def _check_lr(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"learning_rate must be > 0, got {v}")
        return v

    @field_validator("num_epochs")
    @classmethod
    def _check_epochs(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"num_epochs must be > 0, got {v}")
        return v

    @field_validator("batch_size")
    @classmethod
    def _check_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"batch_size must be > 0, got {v}")
        return v

    @field_validator("early_stopping_patience")
    @classmethod
    def _check_patience(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"early_stopping_patience must be > 0, got {v}")
        return v

    @field_validator("label_smoothing")
    @classmethod
    def _check_label_smoothing(cls, v: float) -> float:
        if not 0 <= v < 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {v}")
        return v

    @field_validator("weight_decay")
    @classmethod
    def _check_weight_decay(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"weight_decay must be >= 0, got {v}")
        return v

    @model_validator(mode="after")
    def _check_focal_gamma(self) -> TrainConfig:
        if self.use_focal_loss and self.focal_loss_gamma <= 0:
            raise ValueError(
                f"focal_loss_gamma must be > 0 when use_focal_loss is True, "
                f"got {self.focal_loss_gamma}"
            )
        return self

    @model_validator(mode="after")
    def resolve_h5_paths(self) -> TrainConfig:
        """Back-compat: resolve h5_paths_glob or h5_path into h5_paths list."""
        if not self.h5_paths:
            if self.h5_paths_glob:
                from glob import glob as globfn

                self.h5_paths = sorted(Path(p) for p in globfn(self.h5_paths_glob))
            elif self.h5_path:
                self.h5_paths = [Path(self.h5_path)]
        if not self.h5_paths:
            raise ValueError("No HDF5 embedding files found — check config.")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
