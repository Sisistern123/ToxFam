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
from toxfam.model.forward import forward_model
from toxfam.training.strategies import DataSelector
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
    from toxfam.evaluation.metrics import nontoxin_indices

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

    # P(toxic) = 1 - sum over the nontoxin-class probabilities.
    p_nontox = all_probs[:, nontoxin_indices(label_encoder.classes_)].sum(axis=1)
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
        PlattCalibrator,
        binary_calibration_analysis,
        calculate_binary_metrics_with_scores,
        find_optimal_threshold,
    )

    console.print("\n[bold]Running Binary Metrics Pipeline...[/bold]")

    # Raw P(toxic) — inherits the 38-class temperature (systematically over-toxic).
    val_y_true = compute_binary_labels(val_df, label_col)
    val_p_raw = compute_p_toxic(model, val_df, config, label_encoder, label_col)
    test_y_true = compute_binary_labels(test_df, label_col)
    test_p_raw = compute_p_toxic(model, test_df, config, label_encoder, label_col)

    # Recommendation #1 (DEPLOYED): fit a dedicated Platt calibrator for P(toxic)
    # on val and score on the calibrated probability. The raw-vs-calibrated
    # diagnostic (ECE/Brier/NLL + bootstrap CIs) is kept in binary_calibration.json;
    # the calibrator itself is persisted to models/binary_calibrator.json so predict
    # and eval apply it. Platt is monotonic, so ROC-AUC/PR-AUC are unchanged.
    binary_cal = binary_calibration_analysis(
        val_p_raw, val_y_true, test_p_raw, test_y_true, n_boot=1000, seed=42
    )
    calibrator = PlattCalibrator.from_dict(binary_cal["platt"])
    val_p_toxic = calibrator.transform(val_p_raw)
    test_p_toxic = calibrator.transform(test_p_raw)
    console.print(
        f"  Binary P(toxic) Platt-calibrated — ECE {binary_cal['test_raw']['ece']:.4f} "
        f"→ {binary_cal['test_calibrated']['ece']:.4f}, "
        f"Brier {binary_cal['test_raw']['brier']:.4f} "
        f"→ {binary_cal['test_calibrated']['brier']:.4f} "
        f"(ROC-AUC unchanged {binary_cal['roc_auc_raw']:.4f})"
    )

    # Threshold optimized in CALIBRATED score space, to match the deployed P(toxic).
    thresh_result = find_optimal_threshold(val_y_true, val_p_toxic, method="youden")
    opt_threshold = thresh_result["optimal_threshold"]
    console.print(
        f"  Deployed threshold (Youden's J, calibrated space): {opt_threshold:.4f}"
    )

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

    # Persist the deployed calibrator next to the checkpoint so it ships with the
    # model and predict/eval load it (see model.inference._load_binary_calibrator).
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "binary_calibrator.json").write_text(
        json.dumps(
            {**calibrator.to_dict(), "threshold": opt_threshold,
             "threshold_space": "platt"},
            indent=4,
        )
    )

    # Save metrics — binary_metrics.json now reports the DEPLOYED (calibrated) score.
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    _curve_keys = {
        "fpr", "tpr", "precision_curve", "recall_curve",
        "roc_thresholds", "pr_thresholds",
    }
    binary_results = {
        "optimized_threshold": opt_threshold,
        "score_space": "platt_calibrated",
        "test_default": {k: v for k, v in test_default.items() if k not in _curve_keys},
        "test_optimized": {k: v for k, v in test_opt.items() if k not in _curve_keys},
    }
    (metrics_dir / "binary_metrics.json").write_text(
        json.dumps(binary_results, indent=4)
    )
    (metrics_dir / "binary_calibration.json").write_text(
        json.dumps(binary_cal, indent=4)
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
