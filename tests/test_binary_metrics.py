"""Tests for score-based binary metrics and threshold optimization."""

import numpy as np
import pytest

from toxfam.evaluation.metrics import (
    NONTOXIN_LABELS,
    calculate_binary_metrics_with_scores,
    find_optimal_threshold,
    to_binary_class,
)


class TestToBinaryClass:
    def test_nontox_label(self):
        assert to_binary_class("nontox") == "nontoxin"

    def test_nontoxic_label(self):
        assert to_binary_class("nontoxic") == "nontoxin"

    def test_toxin_family(self):
        assert to_binary_class("Phospholipase A2") == "toxin"

    def test_other_is_toxin(self):
        assert to_binary_class("other") == "toxin"


class TestNontoxinLabels:
    def test_all_variants(self):
        assert "nontox" in NONTOXIN_LABELS
        assert "nontoxic" in NONTOXIN_LABELS
        assert "nontoxin" in NONTOXIN_LABELS

    def test_nontoxin_maps_correctly(self):
        assert to_binary_class("nontoxin") == "nontoxin"


class TestCalculateBinaryMetricsWithScores:
    def test_perfect_scores(self):
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.1, 0.2])
        result = calculate_binary_metrics_with_scores(y_true, y_scores)
        assert result["roc_auc"] == pytest.approx(1.0)
        assert result["pr_auc"] == pytest.approx(1.0)

    def test_returns_expected_keys(self):
        y_true = np.array([1, 0, 1, 0])
        y_scores = np.array([0.8, 0.3, 0.7, 0.4])
        result = calculate_binary_metrics_with_scores(y_true, y_scores)
        assert "roc_auc" in result
        assert "pr_auc" in result
        assert "f1" in result
        assert "mcc" in result
        assert "threshold" in result

    def test_threshold_default_05(self):
        y_true = np.array([1, 0])
        y_scores = np.array([0.6, 0.4])
        result = calculate_binary_metrics_with_scores(y_true, y_scores)
        assert result["threshold"] == 0.5


class TestFindOptimalThreshold:
    def test_youden_method(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        thresh = find_optimal_threshold(y_true, y_scores, method="youden")
        assert 0.3 < thresh < 0.8

    def test_f1_method(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        thresh = find_optimal_threshold(y_true, y_scores, method="f1")
        assert 0.0 < thresh < 1.0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            find_optimal_threshold(np.array([1]), np.array([0.5]), method="invalid")
