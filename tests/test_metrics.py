"""Tests for toxfam.evaluation.metrics — both MetricsResult and binary score APIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from toxfam.evaluation.metrics import (
    NONTOXIN_LABELS,
    MetricsResult,
    calculate_binary_metrics,
    calculate_binary_metrics_with_scores,
    calculate_metrics,
    find_optimal_threshold,
    to_binary_class,
)


# ---------- to_binary_class ----------


def test_to_binary_class_nontox():
    assert to_binary_class("nontox") == "nontoxin"


def test_to_binary_class_nontoxic():
    assert to_binary_class("nontoxic") == "nontoxin"


def test_to_binary_class_nontoxin():
    assert to_binary_class("nontoxin") == "nontoxin"


def test_to_binary_class_toxin_family():
    assert to_binary_class("Conotoxin family") == "toxin"


def test_to_binary_class_other():
    assert to_binary_class("other") == "toxin"


def test_nontoxin_labels_constant():
    assert "nontox" in NONTOXIN_LABELS
    assert "nontoxic" in NONTOXIN_LABELS
    assert "nontoxin" in NONTOXIN_LABELS


# ---------- calculate_metrics (MetricsResult) ----------


def test_calculate_metrics_perfect():
    y_true = pd.Series(["A", "B", "C", "A", "B"])
    y_pred = pd.Series(["A", "B", "C", "A", "B"])
    m = calculate_metrics(y_true, y_pred)
    assert isinstance(m, MetricsResult)
    assert m.accuracy == 1.0
    assert m.mcc == 1.0
    assert m.n_samples == 5


def test_calculate_metrics_with_class_list():
    y_true = pd.Series(["A", "B"])
    y_pred = pd.Series(["A", "B"])
    m = calculate_metrics(y_true, y_pred, class_list=["A", "B", "C"])
    assert len(m.class_list) == 3


def test_calculate_metrics_to_json_dict():
    y_true = pd.Series(["A", "B"])
    y_pred = pd.Series(["A", "B"])
    m = calculate_metrics(y_true, y_pred)
    d = m.to_json_dict()
    assert "numeric_metrics" in d
    assert "classification_report" in d


def test_calculate_metrics_to_summary_dict():
    y_true = pd.Series(["A", "B"])
    y_pred = pd.Series(["A", "B"])
    m = calculate_metrics(y_true, y_pred)
    d = m.to_summary_dict("test_method")
    assert d["Method"] == "test_method"
    assert "Accuracy" in d


# ---------- calculate_binary_metrics ----------


def test_binary_metrics_returns_metrics_result():
    y_true = pd.Series(["nontox", "famA", "famB"])
    y_pred = pd.Series(["nontox", "famA", "famB"])
    m = calculate_binary_metrics(y_true, y_pred)
    assert isinstance(m, MetricsResult)
    assert m.accuracy == 1.0


# ---------- calculate_binary_metrics_with_scores ----------


def test_binary_metrics_with_scores_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = calculate_binary_metrics_with_scores(y_true, y_scores)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert "fpr" in m
    assert "tpr" in m
    assert "roc_thresholds" in m
    assert "pr_thresholds" in m


def test_binary_metrics_with_scores_custom_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.3, 0.7, 0.9])
    m = calculate_binary_metrics_with_scores(y_true, y_scores, threshold=0.6)
    assert m["threshold"] == 0.6
    assert m["accuracy"] == 1.0


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
