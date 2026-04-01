"""Tests for toxfam.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from toxfam.evaluation.metrics import (
    NONTOXIN_LABELS,
    calculate_binary_metrics,
    calculate_binary_metrics_with_scores,
    calculate_multiclass_metrics,
    find_optimal_threshold,
    to_binary_class,
)


# ---------- to_binary_class ----------


def test_to_binary_class_nontox():
    assert to_binary_class("nontox") == "nontoxin"


def test_to_binary_class_nontoxic():
    """Regression: 'nontoxic' label (used by binary/hierarchical) must map to nontoxin."""
    assert to_binary_class("nontoxic") == "nontoxin"


def test_to_binary_class_toxin_family():
    assert to_binary_class("Conotoxin family") == "toxin"


def test_to_binary_class_other():
    assert to_binary_class("other") == "toxin"


def test_nontoxin_labels_constant():
    assert "nontox" in NONTOXIN_LABELS
    assert "nontoxic" in NONTOXIN_LABELS


# ---------- calculate_binary_metrics ----------


def test_binary_metrics_perfect():
    df = pd.DataFrame(
        {"truth": ["nontox", "famA", "famB"], "pred": ["nontox", "famA", "famB"]}
    )
    m = calculate_binary_metrics(df, "truth", "pred")
    assert m["acc"] == 1.0
    assert m["mcc"] == 1.0
    assert m["n_samples"] == 3
    assert "report" in m


def test_binary_metrics_imperfect():
    df = pd.DataFrame(
        {"truth": ["nontox", "famA", "famB"], "pred": ["famA", "famA", "famB"]}
    )
    m = calculate_binary_metrics(df, "truth", "pred")
    # One wrong: nontox predicted as toxin
    assert 0.0 < m["acc"] < 1.0


# ---------- calculate_multiclass_metrics ----------


def test_multiclass_metrics_perfect():
    df = pd.DataFrame({"truth": ["A", "B", "C"], "pred": ["A", "B", "C"]})
    m = calculate_multiclass_metrics(df, "truth", "pred")
    assert m["acc"] == 1.0
    assert m["mcc"] == 1.0
    assert len(m["class_list"]) == 3
    assert "report" in m
    assert "y_true_encoded" in m
    assert "y_pred_encoded" in m


def test_multiclass_metrics_with_shared_class_list():
    df = pd.DataFrame({"truth": ["A", "B"], "pred": ["A", "B"]})
    m = calculate_multiclass_metrics(
        df, "truth", "pred", shared_class_list=["A", "B", "C"]
    )
    assert len(m["class_list"]) == 3


def test_multiclass_metrics_imperfect():
    df = pd.DataFrame(
        {"truth": ["A", "A", "B", "B"], "pred": ["A", "B", "B", "A"]}
    )
    m = calculate_multiclass_metrics(df, "truth", "pred")
    assert m["acc"] == 0.5
    assert m["n_samples"] == 4


# ---------- calculate_binary_metrics_with_scores ----------


def test_binary_metrics_with_scores_returns_thresholds():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.3, 0.7, 0.9])
    m = calculate_binary_metrics_with_scores(y_true, y_scores)
    assert "roc_thresholds" in m
    assert "pr_thresholds" in m
    assert len(m["roc_thresholds"]) > 0
    assert len(m["pr_thresholds"]) > 0


def test_binary_metrics_with_scores_custom_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.3, 0.7, 0.9])
    m = calculate_binary_metrics_with_scores(y_true, y_scores, threshold=0.6)
    assert m["threshold"] == 0.6
    assert m["accuracy"] == 1.0  # all correctly classified at 0.6


# ---------- find_optimal_threshold ----------


def test_find_optimal_threshold_youden():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    result = find_optimal_threshold(y_true, y_scores, method="youden")
    assert "optimal_threshold" in result
    assert result["method"] == "youden"
    assert 0.0 < result["optimal_threshold"] < 1.0
    assert "youden_j" in result


def test_find_optimal_threshold_f1():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    result = find_optimal_threshold(y_true, y_scores, method="f1")
    assert result["method"] == "f1"
    assert "best_f1" in result
    assert result["best_f1"] > 0


def test_find_optimal_threshold_target_precision():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    result = find_optimal_threshold(
        y_true, y_scores, method="target_precision", target_precision=0.9,
    )
    assert result["method"] == "target_precision"
    assert "achieved_precision" in result


def test_find_optimal_threshold_invalid_method():
    y_true = np.array([0, 1])
    y_scores = np.array([0.3, 0.7])
    with pytest.raises(ValueError, match="Unknown method"):
        find_optimal_threshold(y_true, y_scores, method="invalid")
