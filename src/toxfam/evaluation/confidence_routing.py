"""Confidence routing: use HBI prediction when confident, NN otherwise.

When a strong homology hit exists (low e-value), HBI's best-hit transfer is
often more reliable than the NN. For sequences with no or weak hits, the NN
provides better predictions. This module combines both approaches post-hoc.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from toxfam.evaluation.metrics import calculate_binary_metrics_with_scores


def evaluate_confidence_routing(
    nn_p_toxic: np.ndarray,
    hbi_h5_path: Path,
    identifiers: list[str],
    y_true: np.ndarray,
    *,
    evalue_threshold: float = 1e-10,
    output_dir: Path | None = None,
) -> dict:
    """Combine HBI and NN predictions using confidence routing.

    Strategy:
    - If HBI best_hit neg_log_evalue (feature idx 3) is above threshold → use
      HBI best_hit_is_toxic (feature idx 1) as prediction.
    - Otherwise → use NN p_toxic.

    Parameters
    ----------
    nn_p_toxic : array of shape (N,)
        NN-predicted probability of being toxic.
    hbi_h5_path : Path
        H5 file with HBI features (4-dim per sequence).
    identifiers : list[str]
        Sequence IDs matching y_true / nn_p_toxic order.
    y_true : array of shape (N,)
        Binary ground truth (1=toxic, 0=nontoxic).
    evalue_threshold : float
        neg_log_evalue threshold; above → trust HBI.
    output_dir : optional Path
        Save metrics JSON here.

    Returns dict with metrics + routing stats.
    """
    n = len(identifiers)

    # Load HBI features
    hbi_features = np.zeros((n, 4), dtype=np.float32)
    with h5py.File(str(hbi_h5_path), "r") as f:
        for i, sid in enumerate(identifiers):
            if sid in f:
                hbi_features[i] = f[sid][:]

    # Feature indices: 0=fident, 1=is_toxic, 2=top5_frac_toxic, 3=neg_log_evalue
    hbi_has_hit = hbi_features[:, 0] > 0  # has any hit
    hbi_confident = hbi_features[:, 3] >= evalue_threshold  # confident hit
    use_hbi = hbi_has_hit & hbi_confident

    # Build combined predictions
    combined_p_toxic = nn_p_toxic.copy()
    combined_p_toxic[use_hbi] = hbi_features[use_hbi, 1]  # binary: 0 or 1

    metrics = calculate_binary_metrics_with_scores(y_true, combined_p_toxic)

    # Routing stats
    n_hbi = int(use_hbi.sum())
    n_nn = n - n_hbi
    metrics["routing_n_hbi"] = n_hbi
    metrics["routing_n_nn"] = n_nn
    metrics["routing_pct_hbi"] = round(100 * n_hbi / n, 1) if n > 0 else 0

    print(
        f"Confidence routing: {n_hbi} HBI ({metrics['routing_pct_hbi']}%), "
        f"{n_nn} NN — MCC={metrics['mcc']:.4f}, "
        f"ROC-AUC={metrics['roc_auc']:.4f}"
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        serializable = {
            k: v for k, v in metrics.items()
            if k not in ("fpr", "tpr", "precision_curve", "recall_curve",
                         "roc_thresholds", "pr_thresholds")
        }
        (output_dir / "confidence_routing_metrics.json").write_text(
            json.dumps(serializable, indent=4)
        )

    return metrics
