"""Shared evaluation metrics for ToxFam."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from sklearn.preprocessing import label_binarize

NONTOXIN_LABELS = {"nontox"}


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
