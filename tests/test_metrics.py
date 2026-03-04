"""Tests for toxfam.evaluation.metrics."""

from __future__ import annotations

import pandas as pd

from toxfam.evaluation.metrics import (
    NONTOXIN_LABELS,
    calculate_binary_metrics,
    calculate_multiclass_metrics,
    to_binary_class,
)


# ---------- to_binary_class ----------


def test_to_binary_class_nontox():
    assert to_binary_class("nontox") == "nontoxin"


def test_to_binary_class_toxin_family():
    assert to_binary_class("Conotoxin family") == "toxin"


def test_to_binary_class_other():
    assert to_binary_class("other") == "toxin"


def test_nontoxin_labels_constant():
    assert "nontox" in NONTOXIN_LABELS


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
