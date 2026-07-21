"""Evaluation metrics for protein family classification.

Provides:
- MetricsResult dataclass with multiclass metrics (accuracy, MCC, micro-MCC)
- Score-based binary metrics (ROC-AUC, PR-AUC, F1, MCC from probability scores)
- Threshold optimization (Youden's J, F1, target precision)

Predictions not in the class list (including "no hit") are mapped to an
out-of-vocabulary index and counted as wrong predictions, following the
ProtTucker convention (Heinzinger et al. 2022).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

console = Console()

NONTOXIN_LABELS: set[str] = {"nontox", "nontoxic", "nontoxin"}


# ---------------------------------------------------------------------------
# Multiclass metrics (MetricsResult dataclass)
# ---------------------------------------------------------------------------


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""

    accuracy: float
    mcc: float
    micro_mcc: float
    macro_mcc: float
    std_error: float
    n_samples: int
    class_list: list[str] = field(repr=False)
    classification_report: dict[str, Any] = field(repr=False)
    y_true_encoded: np.ndarray = field(repr=False)
    y_pred_encoded: np.ndarray = field(repr=False)

    def to_summary_dict(self, method_name: str) -> dict[str, Any]:
        """Return a dict suitable for one row of a summary DataFrame."""
        return {
            "Method": method_name,
            "Accuracy": self.accuracy,
            "MCC": self.mcc,
            "Micro_MCC": self.micro_mcc,
            "Macro_MCC": self.macro_mcc,
            "Std_Error": self.std_error,
            "Sample_Size": self.n_samples,
        }

    def to_json_dict(self) -> dict[str, Any]:
        """Return a dict suitable for JSON serialization."""
        return {
            "numeric_metrics": {
                "Test_Accuracy": self.accuracy,
                "Test_MCC": self.mcc,
                "Test_Micro_MCC": self.micro_mcc,
                "Test_Macro_MCC": self.macro_mcc,
                "Test_Std_Error": self.std_error,
            },
            "classification_report": self.classification_report,
        }


def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    class_list: list[str] | None = None,
) -> MetricsResult:
    """Compute multiclass classification metrics.

    Parameters
    ----------
    y_true : Series of string labels (ground truth).
    y_pred : Series of string labels (predictions).
    class_list : Ordered class names. Defaults to ``sorted(y_true.unique())``.
        Predictions not in this list (e.g. "no hit") are mapped to an
        out-of-vocabulary index and counted as wrong.
    """
    if class_list is None:
        class_list = sorted(y_true.unique().tolist())

    n_classes = len(class_list)
    cls2idx = {name: i for i, name in enumerate(class_list)}
    oov_idx = n_classes  # guaranteed wrong for any valid true label

    y_true_enc = y_true.map(lambda x: cls2idx.get(x, oov_idx)).to_numpy(dtype=int)
    y_pred_enc = y_pred.map(lambda x: cls2idx.get(x, oov_idx)).to_numpy(dtype=int)

    n_oov_true = int((y_true_enc == oov_idx).sum())
    if n_oov_true > 0:
        console.print(
            f"   [yellow]WARNING: {n_oov_true} ground-truth labels not in class_list[/]"
        )

    n_samples = len(y_true_enc)
    acc = accuracy_score(y_true_enc, y_pred_enc)
    mcc = matthews_corrcoef(y_true_enc, y_pred_enc)

    # Micro-MCC via binarization
    all_labels = list(range(n_classes)) + [oov_idx]
    y_true_bin = label_binarize(y_true_enc, classes=all_labels)
    y_pred_bin = label_binarize(y_pred_enc, classes=all_labels)

    if len(all_labels) == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        y_pred_bin = np.hstack((1 - y_pred_bin, y_pred_bin))

    micro_mcc = matthews_corrcoef(y_true_bin.ravel(), y_pred_bin.ravel())

    # Macro-MCC: unweighted mean of per-class one-vs-rest MCC over the real
    # classes (the trailing oov column is excluded, but oov predictions still
    # count against their true class as misses). Matches the training-path
    # definition in trainer.py, so model runs and eval runs are comparable.
    # Unlike micro-MCC, this weights all families equally, so rare families
    # dominate — the axis on which homology (one hit suffices) tends to lead.
    per_class_mcc = [
        matthews_corrcoef(y_true_bin[:, c], y_pred_bin[:, c]) for c in range(n_classes)
    ]
    macro_mcc = float(np.mean(per_class_mcc)) if per_class_mcc else float("nan")

    std_error = (
        np.sqrt((acc * (1 - acc)) / n_samples) if n_samples > 0 else float("nan")
    )

    report = classification_report(
        y_true_enc,
        y_pred_enc,
        labels=list(range(n_classes)),
        target_names=class_list,
        output_dict=True,
        zero_division=0,
    )

    return MetricsResult(
        accuracy=acc,
        mcc=mcc,
        micro_mcc=micro_mcc,
        macro_mcc=macro_mcc,
        std_error=std_error,
        n_samples=n_samples,
        class_list=class_list,
        classification_report=report,
        y_true_encoded=y_true_enc,
        y_pred_encoded=y_pred_enc,
    )


def is_nontoxin(label: object) -> bool:
    """True if a family label denotes a non-toxin class (case-insensitive)."""
    return str(label).lower() in NONTOXIN_LABELS


def nontoxin_indices(labels: Iterable[object]) -> list[int]:
    """Positions of the non-toxin classes in an ordered label sequence.

    Single source for the P(toxic) = 1 - sum(P(nontoxin classes)) column, shared
    by eval (`compute_p_toxic`) and prediction (`run_topk_inference`).
    """
    return [i for i, label in enumerate(labels) if is_nontoxin(label)]


def to_binary_class(label: str) -> str:
    """Map a protein family label to binary toxin/nontoxin."""
    return "nontoxin" if is_nontoxin(label) else "toxin"


def calculate_binary_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> MetricsResult:
    """Compute binary toxin/nontoxin metrics."""
    return calculate_metrics(
        y_true.apply(to_binary_class),
        y_pred.apply(to_binary_class),
    )


def print_metrics_table(results: dict[str, MetricsResult]) -> None:
    """Print a rich comparison table of metrics from multiple methods."""
    table = Table(show_header=True, header_style="bold", padding=(0, 1))
    table.add_column("Method", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("MCC", justify="right")
    table.add_column("Micro-MCC", justify="right")
    table.add_column("Std Error", justify="right")
    table.add_column("Samples", justify="right")

    for name, m in results.items():
        table.add_row(
            name,
            f"{m.accuracy:.4f}",
            f"{m.mcc:.4f}",
            f"{m.micro_mcc:.4f}",
            f"{m.std_error:.4f}",
            str(m.n_samples),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Score-based binary metrics (dict returns for flexibility)
# ---------------------------------------------------------------------------


def calculate_binary_metrics_with_scores(
    y_true_binary: np.ndarray,
    y_scores_toxic: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Binary metrics using toxic class probability scores.

    Args:
        y_true_binary: Binary labels (1=toxic, 0=nontoxic).
        y_scores_toxic: Probability of being toxic for each sample.
        threshold: Classification threshold.

    Returns:
        Dict with roc_auc, pr_auc, f1, mcc, accuracy, threshold,
        fpr, tpr, precision_curve, recall_curve, roc_thresholds, pr_thresholds.
    """
    y_pred = (y_scores_toxic >= threshold).astype(int)

    fpr, tpr, roc_thresholds = roc_curve(y_true_binary, y_scores_toxic)
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(
        y_true_binary, y_scores_toxic
    )

    return {
        "roc_auc": float(roc_auc_score(y_true_binary, y_scores_toxic)),
        "pr_auc": float(average_precision_score(y_true_binary, y_scores_toxic)),
        "f1": float(f1_score(y_true_binary, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_binary, y_pred)),
        "accuracy": float(accuracy_score(y_true_binary, y_pred)),
        "threshold": threshold,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": prec_curve.tolist(),
        "recall_curve": rec_curve.tolist(),
        "roc_thresholds": roc_thresholds.tolist(),
        "pr_thresholds": pr_thresholds.tolist(),
    }


# ---------------------------------------------------------------------------
# Calibration-quality metrics
#
# Global temperature scaling (ModelWithTemperature) is judged in-repo only by
# top-1 ECE. These add the richer notions the calibration literature uses to
# decide whether a single scalar is enough: classwise-ECE / Static Calibration
# Error (per-class one-vs-rest reliability), adaptive (equal-mass) ECE, Brier,
# and NLL. See Kull et al. NeurIPS 2019 (classwise-ECE) and Nixon et al.
# CVPRW 2019 (binning sensitivity / SCE).
# ---------------------------------------------------------------------------


def _onehot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    oh = np.zeros((len(labels), n_classes), dtype=float)
    oh[np.arange(len(labels)), labels] = 1.0
    return oh


def _reliability_ece(scores: np.ndarray, positives: np.ndarray, n_bins: int) -> float:
    """Binned reliability gap E|conf - freq| for one score vs its outcome.

    ``scores`` is a per-sample probability, ``positives`` the matching 0/1
    outcome. Equal-width bins on [0, 1], lower-exclusive / upper-inclusive to
    match the torch ``_ECELoss`` used during calibration; the first bin also
    captures an exact-zero score so a never-predicted class still contributes.
    """
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positives, dtype=float)
    n = len(scores)
    if n == 0:
        return 0.0
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (scores > lo) & (scores <= hi)
        if lo == 0.0:
            in_bin = in_bin | (scores == 0.0)
        prop = in_bin.mean()
        if prop > 0:
            ece += abs(positives[in_bin].mean() - scores[in_bin].mean()) * prop
    return float(ece)


def _top_conf_acc(
    probs: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample top-1 confidence (max prob) and 0/1 correctness of the argmax."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    return probs.max(axis=1), (probs.argmax(axis=1) == labels).astype(float)


def top_label_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-1 confidence ECE (Guo et al.'s notion): calibration of max prob."""
    confidences, accuracies = _top_conf_acc(probs, labels)
    return _reliability_ece(confidences, accuracies, n_bins)


def adaptive_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Adaptive ECE (ACE): equal-mass confidence bins instead of equal-width.

    Removes ECE's sensitivity to empty/underpopulated high-confidence bins by
    putting an equal number of samples in each bin (Nixon et al. 2019).
    """
    confidences, accuracies = _top_conf_acc(probs, labels)
    n = len(confidences)
    if n == 0:
        return 0.0
    order = np.argsort(confidences, kind="stable")
    conf_sorted = confidences[order]
    acc_sorted = accuracies[order]
    ece = 0.0
    for idx in np.array_split(np.arange(n), min(n_bins, n)):
        if len(idx) == 0:
            continue
        ece += abs(acc_sorted[idx].mean() - conf_sorted[idx].mean()) * (len(idx) / n)
    return float(ece)


def classwise_ece_per_class(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> list[dict[str, Any]]:
    """Per-class one-vs-rest ECE with its test support, one entry per class.

    The honest way to read classwise reliability on an imbalanced many-class
    problem: the aggregate is dominated by near-empty classes (each ~0), so the
    per-class breakdown (with support) is needed to see whether a *populated*
    class is miscalibrated.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    n_classes = probs.shape[1]
    out = []
    for c in range(n_classes):
        yc = (labels == c).astype(float)
        out.append(
            {
                "class_index": c,
                "support": int(yc.sum()),
                "ece": _reliability_ece(probs[:, c], yc, n_bins),
            }
        )
    return out


def classwise_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Classwise-ECE / Static Calibration Error: mean per-class one-vs-rest ECE.

    For each class ``c`` it bins ``probs[:, c]`` and compares the mean predicted
    probability to the empirical frequency of that class in the bin, then
    averages over classes. Unlike top-1 ECE this exposes per-class over/under-
    confidence a single global temperature cannot fix (Kull et al. 2019).

    NOTE: this is the *unweighted* mean over classes. On an imbalanced
    many-class problem it is diluted toward 0 by near-empty classes — read it
    alongside :func:`classwise_ece_weighted` and :func:`classwise_ece_per_class`.
    """
    per_class = classwise_ece_per_class(probs, labels, n_bins)
    return float(np.mean([e["ece"] for e in per_class])) if per_class else 0.0


def classwise_ece_weighted(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Support-weighted classwise-ECE: per-class ECE weighted by class frequency.

    De-dilutes the plain mean: a miscalibrated populated class dominates instead
    of being averaged away against many empty classes. This is the classwise
    number that actually bears on "would a per-class recalibrator help here?".
    """
    per_class = classwise_ece_per_class(probs, labels, n_bins)
    n = len(np.asarray(labels))
    if n == 0 or not per_class:
        return 0.0
    return float(sum(e["ece"] * e["support"] for e in per_class) / n)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean sum_c (p_c - onehot_c)^2, range [0, 2]."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    oh = _onehot(labels, probs.shape[1])
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def nll_score(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Negative log-likelihood (multiclass cross-entropy), matches log_loss."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    p_true = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    return float(-np.mean(np.log(p_true)))


def multiclass_calibration_report(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict[str, float]:
    """All calibration metrics for a multiclass prob matrix + integer labels."""
    # Reduce the per-class reliability pass two ways (unweighted + support-weighted)
    # from a single computation, rather than looping every class × bin twice.
    per_class = classwise_ece_per_class(probs, labels, n_bins)
    eces = [e["ece"] for e in per_class]
    n = len(np.asarray(labels))
    return {
        "ece": top_label_ece(probs, labels, n_bins),
        "adaptive_ece": adaptive_ece(probs, labels, n_bins),
        "classwise_ece": float(np.mean(eces)) if eces else 0.0,
        "classwise_ece_weighted": (
            float(sum(e["ece"] * e["support"] for e in per_class) / n)
            if n and per_class
            else 0.0
        ),
        "brier": brier_score(probs, labels),
        "nll": nll_score(probs, labels),
    }


def binary_calibration_report(
    p_toxic: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> dict[str, float]:
    """Calibration metrics for a scalar P(toxic) score against 0/1 labels.

    ``ece`` here is positive-class reliability (bin P(toxic), compare to the
    empirical toxic frequency) — the exact quantity a separate binary
    calibrator (:class:`PlattCalibrator`) targets. ``brier`` is the standard
    single-event binary Brier ``mean((p - y)^2)`` (== sklearn brier_score_loss),
    not the two-column multiclass sum.
    """
    p = np.asarray(p_toxic, dtype=float)
    y = np.asarray(y_true, dtype=float)
    return {
        "ece": _reliability_ece(p, (y == 1).astype(float), n_bins),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": nll_score(np.column_stack([1.0 - p, p]), y.astype(int)),
    }


class PlattCalibrator:
    """Platt scaling for a single binary score: sigmoid(a * logit(s) + b).

    Recalibrates the derived P(toxic) = 1 - sum(P(nontoxin)) score with its own
    two parameters instead of inheriting the 38-class temperature. The map is
    monotonic (a > 0 in practice), so it is rank-preserving: it changes ECE /
    Brier / NLL and the fixed-0.5 operating point, but not ROC-AUC / PR-AUC.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, eps: float = 1e-6) -> None:
        self.a = a
        self.b = b
        self.eps = eps

    def _logit(self, s: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=float), self.eps, 1.0 - self.eps)
        return np.log(s / (1.0 - s))

    def fit(self, scores: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        from sklearn.linear_model import LogisticRegression

        x = self._logit(scores).reshape(-1, 1)
        y = np.asarray(y).astype(int)
        # A single observed class (or <2 samples) has no calibration signal and
        # would crash LogisticRegression — fall back to the identity map.
        if len(y) < 2 or np.unique(y).size < 2:
            self.a, self.b = 1.0, 0.0
            return self
        # Unregularized MLE — the canonical Platt fit for a single feature.
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        # Well-separated scores make the L-BFGS line search touch large weights,
        # which trips spurious BLAS "divide by zero / overflow in matmul" FP
        # flags; the fit still converges. Silence just those FP flags.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            lr.fit(x, y)
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        z = self.a * self._logit(scores) + self.b
        return np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0)

    def to_dict(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "eps": self.eps}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> PlattCalibrator:
        return cls(a=d["a"], b=d["b"], eps=d.get("eps", 1e-6))


def binary_calibration_analysis(
    val_p_toxic: np.ndarray,
    val_y: np.ndarray,
    test_p_toxic: np.ndarray,
    test_y: np.ndarray,
    *,
    n_bins: int = 15,
    n_boot: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a Platt calibrator for P(toxic) on val, measure its effect on test.

    Answers "does the derived binary score deserve its own calibrator?" — it
    fits Platt on the validation P(toxic)/label pairs and reports the change in
    calibration (ECE/Brier/NLL) on the held-out test set, plus a rank-preserving
    sanity check that ROC-AUC is unchanged. Pure: numpy in, JSON-able dict out.

    With ``n_boot > 0`` it adds ``delta_ci95``: percentile bootstrap 95% CIs for
    each metric's change, resampling the test set with the (val-fixed) calibrator
    held constant. Binned ECE is a positively-biased small-sample estimator, so
    the point delta alone overstates certainty — the CI is what makes the effect
    size interpretable.
    """
    cal = PlattCalibrator().fit(val_p_toxic, val_y)
    test_p_toxic = np.asarray(test_p_toxic, dtype=float)
    test_y = np.asarray(test_y)
    test_cal = cal.transform(test_p_toxic)

    raw = binary_calibration_report(test_p_toxic, test_y, n_bins)
    calibrated = binary_calibration_report(test_cal, test_y, n_bins)
    delta = {k: calibrated[k] - raw[k] for k in raw}

    result = {
        "platt": cal.to_dict(),
        "n_bins": n_bins,
        "test_raw": raw,
        "test_calibrated": calibrated,
        "delta": delta,
        "roc_auc_raw": float(roc_auc_score(test_y, test_p_toxic)),
        "roc_auc_calibrated": float(roc_auc_score(test_y, test_cal)),
    }

    if n_boot > 0:
        rng = np.random.default_rng(seed)
        n = len(test_y)
        boot = {k: [] for k in delta}
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            r = binary_calibration_report(test_p_toxic[idx], test_y[idx], n_bins)
            c = binary_calibration_report(test_cal[idx], test_y[idx], n_bins)
            for k in delta:
                boot[k].append(c[k] - r[k])
        result["delta_ci95"] = {
            k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in boot.items()
        }
        result["n_boot"] = n_boot

    return result


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    method: str = "youden",
    target_precision: float = 0.9,
) -> dict:
    """Find optimal classification threshold on validation data.

    Methods:
    - youden: maximize TPR - FPR (Youden's J statistic)
    - f1: maximize F1 score
    - target_precision: find threshold achieving target precision with max recall

    Returns dict with optimal_threshold, method, and metrics at that threshold.
    """
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal = float(thresholds[best_idx])
        detail = {"youden_j": float(j_scores[best_idx])}
    elif method == "f1":
        thresholds = np.linspace(0.01, 0.99, 200)
        f1_scores = []
        for t in thresholds:
            preds = (y_scores >= t).astype(int)
            f1_scores.append(f1_score(y_true, preds, zero_division=0))
        best_idx = int(np.argmax(f1_scores))
        optimal = float(thresholds[best_idx])
        detail = {"best_f1": float(f1_scores[best_idx])}
    elif method == "target_precision":
        prec, rec, thresholds = precision_recall_curve(y_true, y_scores)
        valid = prec[:-1] >= target_precision
        if valid.any():
            recall_filtered = np.where(valid, rec[:-1], -1)
            best_idx = int(np.argmax(recall_filtered))
            optimal = float(thresholds[best_idx])
            detail = {
                "achieved_precision": float(prec[best_idx]),
                "achieved_recall": float(rec[best_idx]),
            }
        else:
            best_idx = int(np.argmax(prec[:-1]))
            optimal = float(thresholds[best_idx])
            detail = {
                "achieved_precision": float(prec[best_idx]),
                "achieved_recall": float(rec[best_idx]),
                "warning": f"No threshold achieved target precision {target_precision}",
            }
    else:
        raise ValueError(f"Unknown method: {method}")

    y_pred = (y_scores >= optimal).astype(int)
    return {
        "optimal_threshold": optimal,
        "method": method,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        **detail,
    }
