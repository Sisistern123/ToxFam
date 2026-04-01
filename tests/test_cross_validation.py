"""Tests for k-fold cross-validation metric aggregation."""

import pytest

from toxfam.training.cross_validation import _aggregate_fold_metrics


class TestAggregateFoldMetrics:
    def test_single_fold(self):
        folds = [{"accuracy": 0.9, "mcc": 0.8}]
        result = _aggregate_fold_metrics(folds)
        assert result["accuracy_mean"] == pytest.approx(0.9)
        assert result["accuracy_std"] == pytest.approx(0.0)
        assert result["mcc_mean"] == pytest.approx(0.8)
        assert result["mcc_std"] == pytest.approx(0.0)

    def test_multiple_folds(self):
        folds = [
            {"accuracy": 0.9, "mcc": 0.8},
            {"accuracy": 0.8, "mcc": 0.7},
            {"accuracy": 0.85, "mcc": 0.75},
        ]
        result = _aggregate_fold_metrics(folds)
        assert result["accuracy_mean"] == pytest.approx(0.85, abs=0.01)
        assert result["mcc_mean"] == pytest.approx(0.75, abs=0.01)
        assert result["accuracy_std"] > 0
        assert result["mcc_std"] > 0

    def test_empty_folds(self):
        result = _aggregate_fold_metrics([])
        assert result == {}

    def test_ignores_non_numeric(self):
        folds = [{"accuracy": 0.9, "name": "fold_0"}]
        result = _aggregate_fold_metrics(folds)
        assert "accuracy_mean" in result
        assert "name_mean" not in result
