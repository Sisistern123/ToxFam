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

    training_strategy: Literal["standard", "combined"]

    embedding_dim: int = 1024
    tax_dim: int = 56
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 256])
    dropout: float = 0.3
    batch_size: int = 64
    num_epochs: int = 200
    learning_rate: float = 0.0001
    early_stopping_patience: int = 10

    model_config = {"extra": "ignore"}

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
