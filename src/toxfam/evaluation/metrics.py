"""Evaluation metrics for protein family classification.

Provides unified computation of accuracy, MCC, micro-MCC, standard error,
and sklearn classification reports. Predictions not in the class list
(including "no hit") are mapped to an out-of-vocabulary index and counted
as wrong predictions, following the ProtTucker convention (Heinzinger et al.
2022).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

console = Console()

NONTOXIN_LABELS: set[str] = {"nontox", "nontoxic"}


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
    """Compute classification metrics.

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

    y_true_int = y_true_enc
    y_pred_int = y_pred_enc
    n_samples = len(y_true_int)

    acc = accuracy_score(y_true_int, y_pred_int)
    mcc = matthews_corrcoef(y_true_int, y_pred_int)

    # Micro-MCC via binarization
    all_labels = list(range(n_classes)) + [oov_idx]
    y_true_bin = label_binarize(y_true_int, classes=all_labels)
    y_pred_bin = label_binarize(y_pred_int, classes=all_labels)

    if len(all_labels) == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        y_pred_bin = np.hstack((1 - y_pred_bin, y_pred_bin))

    micro_mcc = matthews_corrcoef(y_true_bin.ravel(), y_pred_bin.ravel())

    std_error = (
        np.sqrt((acc * (1 - acc)) / n_samples) if n_samples > 0 else float("nan")
    )

    report = classification_report(
        y_true_int,
        y_pred_int,
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
        y_true_encoded=y_true_int,
        y_pred_encoded=y_pred_int,
    )


def to_binary_class(label: str) -> str:
    """Map a protein family label to binary toxin/nontoxin."""
    if str(label).lower() in NONTOXIN_LABELS:
        return "nontoxin"
    return "toxin"


def calculate_binary_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> MetricsResult:
    """Compute binary toxin/nontoxin metrics."""
    return calculate_metrics(
        y_true.apply(to_binary_class),
        y_pred.apply(to_binary_class),
    )


def calculate_binary_metrics_with_scores(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute binary metrics using probability scores.

    Parameters
    ----------
    y_true : array of {0, 1} — 1 = toxic, 0 = nontoxin.
    y_scores : array of floats — probability of being toxic.
    threshold : decision threshold for binary classification.

    Returns
    -------
    dict with keys: roc_auc, pr_auc, f1, mcc, accuracy, threshold,
    fpr, tpr, precision_curve, recall_curve.
    """
    y_pred = (y_scores >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_scores)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = float(np.trapezoid(precision_vals[::-1], recall_vals[::-1]))
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": threshold,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_vals,
        "recall_curve": recall_vals,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    method: str = "youden",
) -> float:
    """Find optimal classification threshold.

    Parameters
    ----------
    method : "youden" (maximizes TPR - FPR), "f1" (maximizes F1 score).
    """
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])
    elif method == "f1":
        thresholds = np.linspace(0.01, 0.99, 200)
        best_f1, best_t = 0.0, 0.5
        for t in thresholds:
            y_pred = (y_scores >= t).astype(int)
            f = f1_score(y_true, y_pred, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = t
        return float(best_t)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'youden' or 'f1'.")


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
