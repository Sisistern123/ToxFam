"""Tests for k-fold cross-validation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from toxfam.config import TrainConfig


@pytest.fixture
def kfold_csv(tmp_path) -> Path:
    """Create a minimal training CSV for k-fold testing."""
    rng = np.random.default_rng(42)
    n = 30
    df = pd.DataFrame({
        "identifier": [f"P{i:03d}" for i in range(n)],
        "Sequence": [f"MKTA{'A' * (i % 10)}" for i in range(n)],
        "Protein families": (
            ["famA"] * 10 + ["famB"] * 10 + ["nontox"] * 10
        ),
        "Split": (
            ["train"] * 20 + ["val"] * 5 + ["test"] * 5
        ),
    })
    csv_path = tmp_path / "training_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def kfold_h5(tmp_path) -> Path:
    """Create embeddings H5 for k-fold testing."""
    import h5py

    h5_path = tmp_path / "embeddings.h5"
    rng = np.random.default_rng(42)
    with h5py.File(h5_path, "w") as f:
        for i in range(30):
            f.create_dataset(f"P{i:03d}", data=rng.standard_normal(1024).astype(np.float32))
    return h5_path


class TestAggregateMetrics:
    def test_aggregate_empty(self):
        from toxfam.training.cross_validation import _aggregate_fold_metrics

        assert _aggregate_fold_metrics([]) == {}

    def test_aggregate_single_fold(self):
        from toxfam.training.cross_validation import _aggregate_fold_metrics

        result = _aggregate_fold_metrics([{"roc_auc": 0.9, "pr_auc": 0.8}])
        assert abs(result["roc_auc_mean"] - 0.9) < 1e-6
        assert abs(result["roc_auc_std"] - 0.0) < 1e-6

    def test_aggregate_multiple_folds(self):
        from toxfam.training.cross_validation import _aggregate_fold_metrics

        metrics = [
            {"roc_auc": 0.8, "f1": 0.7},
            {"roc_auc": 0.9, "f1": 0.8},
            {"roc_auc": 0.85, "f1": 0.75},
        ]
        result = _aggregate_fold_metrics(metrics)
        assert abs(result["roc_auc_mean"] - 0.85) < 1e-6
        assert result["roc_auc_std"] > 0
        assert abs(result["f1_mean"] - 0.75) < 1e-6
