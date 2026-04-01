"""Shared evaluation metrics for ToxFam."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
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

NONTOXIN_LABELS = {"nontox", "nontoxic", "nontoxin"}


def to_binary_class(label: str) -> str:
    """Map a family label to 'toxin' or 'nontoxin'."""
    if str(label).lower() in NONTOXIN_LABELS:
        return "nontoxin"
    return "toxin"


def calculate_binary_metrics(
    df: pd.DataFrame,
    truth_col: str,
    pred_col: str,
) -> Dict[str, Any]:
    """Binary toxin/nontoxin metrics (accuracy, MCC, classification report)."""
    y_true = df[truth_col].apply(to_binary_class).to_numpy()
    y_pred = df[pred_col].apply(to_binary_class).to_numpy()

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    n_samples = len(y_true)
    std_error = np.sqrt((acc * (1 - acc)) / n_samples)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["nontoxin", "toxin"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "acc": acc,
        "mcc": mcc,
        "std_error": std_error,
        "n_samples": n_samples,
        "report": report,
    }


def calculate_multiclass_metrics(
    df: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    *,
    shared_class_list: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Multiclass metrics with optional shared class list.

    Returns acc, mcc, micro_mcc, std_error, n_samples, report, class_list,
    cls2idx, y_true_encoded, y_pred_encoded.
    """
    if shared_class_list is not None:
        class_list = shared_class_list
    else:
        class_list = sorted(
            list(set(df[truth_col].unique()) | set(df[pred_col].unique()))
        )

    cls2idx = {cls_name: i for i, cls_name in enumerate(class_list)}

    y_true = df[truth_col].map(cls2idx).to_numpy()
    y_pred = df[pred_col].map(cls2idx).to_numpy()

    n_samples = len(y_true)
    n_classes = len(class_list)

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_pred_bin = label_binarize(y_pred, classes=range(n_classes))

    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        y_pred_bin = np.hstack((1 - y_pred_bin, y_pred_bin))

    micro_mcc = matthews_corrcoef(y_true_bin.ravel(), y_pred_bin.ravel())

    std_error = (
        np.sqrt((acc * (1 - acc)) / n_samples) if n_samples > 0 else float("nan")
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=range(n_classes),
        target_names=class_list,
        output_dict=True,
        zero_division=0,
    )

    return {
        "acc": acc,
        "mcc": mcc,
        "micro_mcc": micro_mcc,
        "std_error": std_error,
        "n_samples": n_samples,
        "report": report,
        "class_list": class_list,
        "cls2idx": cls2idx,
        "y_true_encoded": y_true,
        "y_pred_encoded": y_pred,
    }


def calculate_binary_metrics_with_scores(
    y_true_binary: np.ndarray,
    y_scores_toxic: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Binary metrics using toxic class probability scores.

    Args:
        y_true_binary: Binary labels (1=toxic, 0=nontoxic).
        y_scores_toxic: Probability of being toxic for each sample.
        threshold: Classification threshold.

    Returns:
        Dict with roc_auc, pr_auc, f1, mcc, precision, recall, fpr, tpr,
        precision_curve, recall_curve.
    """
    y_pred = (y_scores_toxic >= threshold).astype(int)

    fpr, tpr, roc_thresholds = roc_curve(y_true_binary, y_scores_toxic)
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(
        y_true_binary, y_scores_toxic
    )

    result: Dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_true_binary, y_scores_toxic)),
        "pr_auc": float(average_precision_score(y_true_binary, y_scores_toxic)),
        "f1": float(f1_score(y_true_binary, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_binary, y_pred)),
        "accuracy": float(accuracy_score(y_true_binary, y_pred)),
        "threshold": threshold,
        # Curve data for plotting
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": prec_curve.tolist(),
        "recall_curve": rec_curve.tolist(),
        # Thresholds for downstream optimization
        "roc_thresholds": roc_thresholds.tolist(),
        "pr_thresholds": pr_thresholds.tolist(),
    }
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
        # precision_recall_curve returns n+1 prec/rec values but n thresholds
        valid = prec[:-1] >= target_precision
        if valid.any():
            # Among those meeting precision target, maximize recall
            recall_filtered = np.where(valid, rec[:-1], -1)
            best_idx = int(np.argmax(recall_filtered))
            optimal = float(thresholds[best_idx])
            detail = {
                "achieved_precision": float(prec[best_idx]),
                "achieved_recall": float(rec[best_idx]),
            }
        else:
            # No threshold meets target — use highest precision
            best_idx = int(np.argmax(prec[:-1]))
            optimal = float(thresholds[best_idx])
            detail = {
                "achieved_precision": float(prec[best_idx]),
                "achieved_recall": float(rec[best_idx]),
                "warning": f"No threshold achieved target precision {target_precision}",
            }
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute metrics at optimal threshold
    y_pred = (y_scores >= optimal).astype(int)
    return {
        "optimal_threshold": optimal,
        "method": method,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        **detail,
    }
