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
