"""Tests for calibration-quality metrics and the binary Platt calibrator.

These cover recommendation #2 (richer calibration metrics beyond top-1 ECE:
classwise-ECE/SCE, adaptive ECE, Brier, NLL) and recommendation #1 (a separate
Platt calibrator for the derived binary P(toxic) score).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import log_loss, roc_auc_score

from sklearn.metrics import brier_score_loss

from toxfam.evaluation.metrics import (
    PlattCalibrator,
    adaptive_ece,
    binary_calibration_analysis,
    binary_calibration_report,
    brier_score,
    classwise_ece,
    classwise_ece_per_class,
    classwise_ece_weighted,
    multiclass_calibration_report,
    nll_score,
    top_label_ece,
)


# ---------------------------------------------------------------------------
# top-1 confidence ECE
# ---------------------------------------------------------------------------


def test_top_label_ece_perfectly_calibrated_is_zero():
    # 100 samples, all predicted class 1 with confidence 0.9; exactly 90 correct.
    probs = np.tile([0.1, 0.9], (100, 1))
    labels = np.array([1] * 90 + [0] * 10)
    assert top_label_ece(probs, labels, n_bins=15) == pytest.approx(0.0, abs=1e-9)


def test_top_label_ece_fully_miscalibrated():
    # Confidence 0.9 on every sample but every prediction is wrong -> ECE = 0.9.
    probs = np.tile([0.1, 0.9], (100, 1))
    labels = np.zeros(100, dtype=int)  # true class 0, always predicts class 1
    assert top_label_ece(probs, labels, n_bins=15) == pytest.approx(0.9, abs=1e-9)


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


def test_brier_score_matches_hand_computation():
    probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    labels = np.array([0, 1])
    # sample 0: (0.8-1)^2 + (0.2-0)^2 = 0.08 ; sample 1: (0.3-0)^2 + (0.7-1)^2 = 0.18
    assert brier_score(probs, labels) == pytest.approx((0.08 + 0.18) / 2)


def test_brier_score_zero_when_perfect():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert brier_score(probs, labels) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# NLL / log loss
# ---------------------------------------------------------------------------


def test_nll_matches_sklearn_log_loss():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(50, 4))
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 4, size=50)
    assert nll_score(probs, labels) == pytest.approx(
        log_loss(labels, probs, labels=[0, 1, 2, 3])
    )


# ---------------------------------------------------------------------------
# classwise-ECE (SCE) and adaptive ECE
# ---------------------------------------------------------------------------


def test_classwise_ece_zero_when_perfectly_calibrated():
    # Every sample prob [0.6, 0.4]; class frequencies match exactly.
    probs = np.tile([0.6, 0.4], (100, 1))
    labels = np.array([0] * 60 + [1] * 40)
    assert classwise_ece(probs, labels, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_classwise_ece_detects_per_class_miscalibration():
    # Predicted prob for class 0 is 0.6 everywhere, but class 0 never occurs.
    probs = np.tile([0.6, 0.4], (100, 1))
    labels = np.ones(100, dtype=int)  # all class 1
    # class 0 contributes |0.6 - 0| = 0.6 ; class 1 contributes |0.4 - 1| = 0.6
    # SCE = mean of the two per-class ECEs = 0.6
    assert classwise_ece(probs, labels, n_bins=10) == pytest.approx(0.6, abs=1e-9)


def test_classwise_ece_is_nonnegative():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(200, 5))
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 5, size=200)
    assert classwise_ece(probs, labels) >= 0.0


def test_adaptive_ece_confident_but_wrong():
    # Equal-mass bins are degenerate on tied confidences, but "confident and
    # always wrong" gives acc=0 in every bin regardless of the split -> 0.9.
    probs = np.tile([0.1, 0.9], (100, 1))
    labels = np.zeros(100, dtype=int)
    assert adaptive_ece(probs, labels, n_bins=10) == pytest.approx(0.9, abs=1e-9)


def test_adaptive_ece_bounded():
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(200, 3))
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 3, size=200)
    assert 0.0 <= adaptive_ece(probs, labels) <= 1.0


# ---------------------------------------------------------------------------
# report dict builders
# ---------------------------------------------------------------------------


def test_multiclass_calibration_report_has_expected_keys():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(80, 3))
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 3, size=80)
    report = multiclass_calibration_report(probs, labels)
    assert set(report) >= {"ece", "adaptive_ece", "classwise_ece", "brier", "nll"}
    assert all(isinstance(v, float) for v in report.values())


def test_binary_calibration_report_has_expected_keys():
    rng = np.random.default_rng(3)
    p = rng.uniform(size=200)
    y = (rng.uniform(size=200) < p).astype(int)
    report = binary_calibration_report(p, y)
    assert set(report) >= {"ece", "brier", "nll"}


# ---------------------------------------------------------------------------
# Platt calibrator (recommendation #1)
# ---------------------------------------------------------------------------


def _overconfident_binary(rng, n):
    """Scores whose true accuracy is ~p but pushed toward 0/1 (overconfident)."""
    true_p = rng.uniform(0.05, 0.95, size=n)
    y = (rng.uniform(size=n) < true_p).astype(int)
    # sharpen: push probabilities away from 0.5 -> overconfident scores
    logit = np.log(true_p / (1 - true_p))
    sharp = 1 / (1 + np.exp(-1.8 * logit))
    return sharp, y


def test_platt_preserves_ranking_and_auc():
    rng = np.random.default_rng(10)
    scores, y = _overconfident_binary(rng, 500)
    cal = PlattCalibrator().fit(scores, y)
    out = cal.transform(scores)
    # Platt is a monotonic map -> ROC-AUC (rank-based) is unchanged.
    assert roc_auc_score(y, out) == pytest.approx(roc_auc_score(y, scores), abs=1e-9)
    # And it is order-preserving element-wise.
    order_in = np.argsort(scores)
    assert np.all(np.diff(out[order_in]) >= -1e-9)


def test_platt_reduces_ece_on_overconfident_data():
    rng = np.random.default_rng(11)
    fit_scores, fit_y = _overconfident_binary(rng, 2000)
    test_scores, test_y = _overconfident_binary(rng, 2000)
    cal = PlattCalibrator().fit(fit_scores, fit_y)
    raw_ece = binary_calibration_report(test_scores, test_y)["ece"]
    cal_ece = binary_calibration_report(cal.transform(test_scores), test_y)["ece"]
    assert cal_ece < raw_ece


def test_platt_serialization_roundtrip():
    rng = np.random.default_rng(12)
    scores, y = _overconfident_binary(rng, 400)
    cal = PlattCalibrator().fit(scores, y)
    restored = PlattCalibrator.from_dict(cal.to_dict())
    np.testing.assert_allclose(restored.transform(scores), cal.transform(scores))


def test_platt_transform_clips_to_unit_interval():
    rng = np.random.default_rng(13)
    scores, y = _overconfident_binary(rng, 300)
    cal = PlattCalibrator().fit(scores, y)
    out = cal.transform(np.array([0.0, 1.0, 0.5]))
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


# ---------------------------------------------------------------------------
# binary_calibration_analysis (fit on val, measure on test) — the core of #1
# ---------------------------------------------------------------------------


def test_binary_calibration_analysis_has_expected_structure():
    rng = np.random.default_rng(20)
    val_p, val_y = _overconfident_binary(rng, 500)
    test_p, test_y = _overconfident_binary(rng, 500)
    out = binary_calibration_analysis(val_p, val_y, test_p, test_y)
    assert set(out) >= {
        "platt", "test_raw", "test_calibrated", "delta",
        "roc_auc_raw", "roc_auc_calibrated",
    }
    assert set(out["platt"]) >= {"a", "b"}
    assert set(out["test_raw"]) >= {"ece", "brier", "nll"}


def test_binary_calibration_analysis_preserves_auc():
    rng = np.random.default_rng(21)
    val_p, val_y = _overconfident_binary(rng, 800)
    test_p, test_y = _overconfident_binary(rng, 800)
    out = binary_calibration_analysis(val_p, val_y, test_p, test_y)
    # Platt is monotonic -> discrimination (AUC) is unchanged by calibration.
    assert out["roc_auc_calibrated"] == pytest.approx(out["roc_auc_raw"], abs=1e-9)


def test_binary_calibration_analysis_improves_ece_on_overconfident():
    rng = np.random.default_rng(22)
    val_p, val_y = _overconfident_binary(rng, 2000)
    test_p, test_y = _overconfident_binary(rng, 2000)
    out = binary_calibration_analysis(val_p, val_y, test_p, test_y)
    # Calibrated ECE is lower than raw -> negative delta.
    assert out["delta"]["ece"] < 0
    assert out["test_calibrated"]["ece"] < out["test_raw"]["ece"]


def test_binary_calibration_analysis_bootstrap_ci_brackets_point_delta():
    rng = np.random.default_rng(23)
    val_p, val_y = _overconfident_binary(rng, 1500)
    test_p, test_y = _overconfident_binary(rng, 1500)
    out = binary_calibration_analysis(
        val_p, val_y, test_p, test_y, n_boot=300, seed=0
    )
    assert set(out["delta_ci95"]) >= {"ece", "brier", "nll"}
    for k in ("ece", "brier", "nll"):
        lo, hi = out["delta_ci95"][k]
        assert lo <= hi


def test_binary_calibration_analysis_no_ci_by_default():
    rng = np.random.default_rng(24)
    val_p, val_y = _overconfident_binary(rng, 400)
    test_p, test_y = _overconfident_binary(rng, 400)
    assert "delta_ci95" not in binary_calibration_analysis(val_p, val_y, test_p, test_y)


# ---------------------------------------------------------------------------
# Fixes surfaced by adversarial review
# ---------------------------------------------------------------------------


def test_binary_report_brier_is_standard_binary_brier():
    # Must equal sklearn's single-event binary Brier, not 2x it.
    rng = np.random.default_rng(25)
    p = rng.uniform(size=300)
    y = (rng.uniform(size=300) < p).astype(int)
    assert binary_calibration_report(p, y)["brier"] == pytest.approx(
        brier_score_loss(y, p)
    )


def test_platt_fit_single_class_falls_back_to_identity():
    cal = PlattCalibrator().fit(np.array([0.2, 0.4, 0.7, 0.9]), np.array([1, 1, 1, 1]))
    assert cal.a == 1.0 and cal.b == 0.0
    x = np.array([0.2, 0.5, 0.8])
    np.testing.assert_allclose(cal.transform(x), x, atol=1e-6)


def test_platt_fit_single_sample_is_identity():
    cal = PlattCalibrator().fit(np.array([0.7]), np.array([1]))
    assert cal.a == 1.0 and cal.b == 0.0


# ---------------------------------------------------------------------------
# Support-weighted / per-class classwise-ECE (de-dilute the many-class average)
# ---------------------------------------------------------------------------


def _many_class_one_populated_miscalibrated():
    # 10 classes, 200 samples: class 0 populated (190) and miscalibrated for
    # its own column (p=0.5 vs true freq 0.95); classes 1..9 are near-empty.
    n, k = 200, 10
    probs = np.full((n, k), 0.5 / (k - 1))
    probs[:, 0] = 0.5
    labels = np.zeros(n, dtype=int)
    labels[190:] = np.arange(1, 11)[:10]  # 10 rare samples spread over classes 1..9+
    labels[190:] = (np.arange(10) % 9) + 1
    return probs, labels


def test_classwise_ece_weighted_zero_when_calibrated():
    probs = np.tile([0.6, 0.4], (100, 1))
    labels = np.array([0] * 60 + [1] * 40)
    assert classwise_ece_weighted(probs, labels, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_classwise_ece_weighted_exceeds_unweighted_under_dilution():
    probs, labels = _many_class_one_populated_miscalibrated()
    unweighted = classwise_ece(probs, labels, n_bins=10)
    weighted = classwise_ece_weighted(probs, labels, n_bins=10)
    # Support weighting recovers the populated-class miscalibration the plain
    # mean dilutes across ~9 near-empty classes.
    assert weighted > 2 * unweighted
    assert 0.0 <= weighted <= 1.0


def test_classwise_ece_per_class_supports_sum_to_n():
    probs, labels = _many_class_one_populated_miscalibrated()
    out = classwise_ece_per_class(probs, labels, n_bins=10)
    assert len(out) == probs.shape[1]
    assert sum(e["support"] for e in out) == len(labels)
    assert all(0.0 <= e["ece"] <= 1.0 for e in out)
    # The populated class 0 carries the largest per-class ECE here.
    assert max(out, key=lambda e: e["ece"])["class_index"] == 0
