from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class TrainConfig(BaseModel):
    input_csv: Path
    h5_paths: list[Path] = Field(default_factory=list)
    h5_paths_glob: str | None = None
    h5_path: str | None = None
    tax_h5_path: Path | None = None
    output_dir: Path

    training_strategy: Literal[
        "standard", "combined", "hierarchical", "binary", "multitask"
    ]

    embedding_dim: int = 1024
    tax_dim: int = 56
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 256])
    dropout: float = 0.3
    batch_size: int = 64
    num_epochs: int = 200
    learning_rate: float = 0.0001
    early_stopping_patience: int = 10

    # CPP feature fields
    cpp_h5_path: Path | None = None
    cpp_dim: int = 100

    # HBI feature fields
    hbi_h5_path: Path | None = None
    hbi_dim: int = 4

    # Handcrafted feature fields (Atchley factors + cysteine patterns)
    handcrafted_h5_path: Path | None = None
    handcrafted_dim: int = 15

    # Additional scalar features
    include_length: bool = False
    include_venom_indicator: bool = False

    # Hierarchical strategy fields
    stage1_model_path: Path | None = None
    stage2_freeze_backbone: bool = True
    stage2_learning_rate: float | None = None
    stage2_hidden_dim: int = 64
    family_min_count: int = 10

    # Identity-aware split threshold
    split_seq_id: float = 0.3

    # Loss function
    loss_function: Literal["cross_entropy", "focal"] = "cross_entropy"
    focal_gamma: float = 2.0

    # Multi-task weights
    multitask_family_weight: float = 1.0
    multitask_binary_weight: float = 1.0

    # k-Fold cross-validation
    n_folds: int = 1

    model_config = {"extra": "ignore"}

    @property
    def effective_embedding_dim(self) -> int:
        """Input dim = embedding_dim + optional feature dimensions."""
        dim = self.embedding_dim
        if self.cpp_h5_path:
            dim += self.cpp_dim
        if self.hbi_h5_path:
            dim += self.hbi_dim
        if self.handcrafted_h5_path:
            dim += self.handcrafted_dim
        if self.include_length:
            dim += 1
        if self.include_venom_indicator:
            dim += 1
        return dim

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
