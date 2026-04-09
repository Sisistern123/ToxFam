"""Score-based binary (toxic/nontoxin) evaluation.

Provides:
- P(toxic) extraction from multiclass or binary models
- Threshold optimization on validation set (Youden's J)
- Full binary evaluation pipeline with ROC/PR curves

Used by both the training orchestrator (auto-runs after training) and
the ``toxfam eval binary`` CLI command (post-hoc on saved models).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rich.console import Console
from torch.utils.data import DataLoader

from toxfam.config import TrainConfig
from toxfam.data.dataset import ToxDataset
from toxfam.device import get_device
from toxfam.training.strategies import DataSelector
from toxfam.training.trainer import forward_model
from toxfam.visualization.analysis import plot_binary_pr, plot_binary_roc

console = Console()


def build_eval_loader(
    dataset_df: pd.DataFrame,
    config: TrainConfig,
    label_encoder,
    label_col: str = "Protein families",
) -> tuple[ToxDataset, DataSelector]:
    """Build a ToxDataset + DataLoader + DataSelector for evaluation.

    Returns (dataset, selector) — caller must call dataset.close() when done.
    """
    ds = ToxDataset(
        dataset_df,
        [str(p) for p in config.h5_paths],
        label_encoder=label_encoder,
        is_train=False,
        label_col=label_col,
        tax_h5_path=str(config.tax_h5_path) if config.tax_h5_path else None,
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)
    selector = DataSelector(
        loader, "both" if config.training_strategy == "combined" else "emb_only",
    )
    return ds, selector


def compute_binary_labels(
    df: pd.DataFrame, label_col: str = "Protein families"
) -> np.ndarray:
    """Convert family labels to binary: 1 = toxic, 0 = nontoxin."""
    from toxfam.evaluation.metrics import to_binary_class

    return (df[label_col].apply(to_binary_class) == "toxin").astype(int).values


def compute_p_toxic(
    model,
    dataset_df: pd.DataFrame,
    config: TrainConfig,
    label_encoder,
    label_col: str = "Protein families",
) -> np.ndarray:
    """Compute P(toxic) for each sample by summing toxic-class probabilities."""
    from toxfam.evaluation.metrics import NONTOXIN_LABELS

    ds, selector = build_eval_loader(dataset_df, config, label_encoder, label_col)

    device = get_device()
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for features, _ in selector:
            outputs = forward_model(model, features, device)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    ds.close()

    # Sum probabilities of all nontoxin classes
    nontox_indices = [
        i for i, cls in enumerate(label_encoder.classes_)
        if cls.lower() in NONTOXIN_LABELS
    ]
    p_nontox = all_probs[:, nontox_indices].sum(axis=1)
    return 1.0 - p_nontox


def run_binary_evaluation(
    model,
    label_encoder,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TrainConfig,
    output_dir: Path,
    label_col: str = "Protein families",
) -> dict:
    """Full binary evaluation: threshold optimization on val, evaluate on test.

    Writes binary_metrics.json, binary_roc.png, and binary_pr.png to output_dir.
    Returns the binary results dict.
    """
    from toxfam.evaluation.metrics import (
        calculate_binary_metrics_with_scores,
        find_optimal_threshold,
    )

    console.print("\n[bold]Running Binary Metrics Pipeline...[/bold]")

    # Val set — threshold optimization
    val_y_true = compute_binary_labels(val_df, label_col)
    val_p_toxic = compute_p_toxic(model, val_df, config, label_encoder, label_col)

    thresh_result = find_optimal_threshold(val_y_true, val_p_toxic, method="youden")
    opt_threshold = thresh_result["optimal_threshold"]
    console.print(f"  Optimized threshold (Youden's J): {opt_threshold:.4f}")

    # Test set — evaluate at both thresholds
    test_y_true = compute_binary_labels(test_df, label_col)
    test_p_toxic = compute_p_toxic(model, test_df, config, label_encoder, label_col)

    test_default = calculate_binary_metrics_with_scores(
        test_y_true, test_p_toxic, threshold=0.5
    )
    console.print(
        f"  Test (t=0.5): ROC-AUC={test_default['roc_auc']:.4f}, "
        f"PR-AUC={test_default['pr_auc']:.4f}, MCC={test_default['mcc']:.4f}"
    )

    test_opt = calculate_binary_metrics_with_scores(
        test_y_true, test_p_toxic, threshold=opt_threshold
    )
    console.print(
        f"  Test (t={opt_threshold:.3f}): ROC-AUC={test_opt['roc_auc']:.4f}, "
        f"PR-AUC={test_opt['pr_auc']:.4f}, MCC={test_opt['mcc']:.4f}"
    )

    # Save metrics
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    _curve_keys = {
        "fpr", "tpr", "precision_curve", "recall_curve",
        "roc_thresholds", "pr_thresholds",
    }
    binary_results = {
        "optimized_threshold": opt_threshold,
        "test_default": {k: v for k, v in test_default.items() if k not in _curve_keys},
        "test_optimized": {k: v for k, v in test_opt.items() if k not in _curve_keys},
    }
    (metrics_dir / "binary_metrics.json").write_text(
        json.dumps(binary_results, indent=4)
    )

    # Plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_binary_roc(
        test_default["fpr"], test_default["tpr"], test_default["roc_auc"],
        plots_dir / "binary_roc.png",
    )
    plot_binary_pr(
        test_default["precision_curve"], test_default["recall_curve"],
        test_default["pr_auc"], plots_dir / "binary_pr.png",
    )

    return binary_results
