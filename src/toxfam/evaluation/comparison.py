"""Comprehensive method comparison: collect predictions, per-family eval, figures.

Loads results from all trained models, external benchmarks, and HBI baselines,
then runs per-family evaluation and generates all publication figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from toxfam._paths import get_project_root
from toxfam.evaluation.metrics import (
    calculate_binary_metrics_with_scores,
    to_binary_class,
)


def _load_ptoxic_from_predictions_csv(pred_csv: Path) -> tuple[list[str], np.ndarray]:
    """Load identifiers and p_toxic from a predictions CSV.

    The CSV has columns: identifier, actual_label, predicted_label, confidence.
    `confidence` is P(predicted_class), not P(toxic). We convert.
    """
    df = pd.read_csv(pred_csv)
    p_toxic = np.where(
        df["predicted_label"] == "toxic",
        df["confidence"].values,
        1.0 - df["confidence"].values,
    )
    return df["identifier"].tolist(), p_toxic.astype(np.float64)


def _compute_length_baseline(test_df: pd.DataFrame) -> np.ndarray:
    """Compute length-based p_toxic: shorter sequences are more likely toxic.

    Uses a sigmoid function centered at 73aa (optimal threshold from analysis).
    """
    lengths = test_df["Sequence"].str.len().values
    # Sigmoid: P(toxic) = 1 / (1 + exp(k * (length - threshold)))
    # k=0.05 gives a smooth transition
    p_toxic = 1.0 / (1.0 + np.exp(0.05 * (lengths - 73)))
    return p_toxic


def _load_hbi_ptoxic(
    hbi_h5_path: Path,
    identifiers: list[str],
) -> np.ndarray:
    """Load HBI best-hit predictions as p_toxic.

    HBI features: [fident, is_toxic, top5_frac_toxic, neg_log_evalue]
    For best-hit transfer: p_toxic = best_hit_is_toxic (binary 0/1).
    For sequences without hits: p_toxic = 0 (predict nontoxic).
    """
    p_toxic = np.zeros(len(identifiers), dtype=np.float64)
    with h5py.File(str(hbi_h5_path), "r") as f:
        for i, sid in enumerate(identifiers):
            if sid in f:
                feats = f[sid][:]
                p_toxic[i] = feats[1]  # best_hit_is_toxic
    return p_toxic


def collect_all_predictions(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    model_base: Path,
    hbi_h5_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Collect p_toxic predictions from all available methods.

    Returns:
        predictions: {method_name: p_toxic_array}
        metrics: {method_name: {roc_auc, pr_auc, f1, mcc, ...}}
    """
    predictions = {}
    metrics = {}
    test_ids = test_df["identifier"].tolist()

    # --- 1. Length baseline ---
    p_length = _compute_length_baseline(test_df)
    predictions["Length baseline"] = p_length
    metrics["Length baseline"] = calculate_binary_metrics_with_scores(y_true, p_length)

    # --- 2. NN binary (vanilla, no HBI) ---
    vanilla_pred = model_base / "binary_run" / "predictions" / "test_calibrated_predictions.csv"
    if vanilla_pred.exists():
        ids, p = _load_ptoxic_from_predictions_csv(vanilla_pred)
        # Align with test_df order
        id_to_p = dict(zip(ids, p))
        p_aligned = np.array([id_to_p.get(sid, 0.5) for sid in test_ids])
        predictions["NN binary"] = p_aligned
        metrics["NN binary"] = calculate_binary_metrics_with_scores(y_true, p_aligned)

    # --- 3. NN augmented + counterparts ---
    aug_pred = model_base / "binary_augmented_counterparts_run" / "predictions" / "test_calibrated_predictions.csv"
    if aug_pred.exists():
        ids, p = _load_ptoxic_from_predictions_csv(aug_pred)
        id_to_p = dict(zip(ids, p))
        p_aligned = np.array([id_to_p.get(sid, 0.5) for sid in test_ids])
        predictions["NN augmented+CP"] = p_aligned
        metrics["NN augmented+CP"] = calculate_binary_metrics_with_scores(y_true, p_aligned)

    # --- 3b. NN binary + CPP features ---
    cpp_pred = model_base / "binary_cpp_run" / "predictions" / "test_calibrated_predictions.csv"
    if cpp_pred.exists():
        ids, p = _load_ptoxic_from_predictions_csv(cpp_pred)
        id_to_p = dict(zip(ids, p))
        p_aligned = np.array([id_to_p.get(sid, 0.5) for sid in test_ids])
        predictions["NN binary+CPP"] = p_aligned
        metrics["NN binary+CPP"] = calculate_binary_metrics_with_scores(y_true, p_aligned)

    # --- 4. HBI best-hit transfer ---
    if hbi_h5_path and hbi_h5_path.exists():
        p_hbi = _load_hbi_ptoxic(hbi_h5_path, test_ids)
        predictions["HBI best-hit"] = p_hbi
        metrics["HBI best-hit"] = calculate_binary_metrics_with_scores(y_true, p_hbi)

    # --- 5. ToxinPred2 ---
    tp2_dir = model_base / "external_benchmarks"
    tp2_metrics_path = tp2_dir / "toxinpred2_model1_metrics.json"
    if tp2_metrics_path.exists():
        with open(tp2_metrics_path) as f:
            tp2_m = json.load(f)
        metrics["ToxinPred2"] = tp2_m

    # --- 5b. ToxinPred3 ---
    tp3_metrics_path = tp2_dir / "toxinpred3_model1_metrics.json"
    if tp3_metrics_path.exists():
        with open(tp3_metrics_path) as f:
            tp3_m = json.load(f)
        metrics["ToxinPred3"] = tp3_m

    # --- 5c. TOXIFY (reimplemented) ---
    toxify_metrics_path = tp2_dir / "toxify_metrics.json"
    if toxify_metrics_path.exists():
        with open(toxify_metrics_path) as f:
            toxify_m = json.load(f)
        metrics["TOXIFY (reimpl.)"] = toxify_m

    # --- 6. Confidence routing (vanilla NN + HBI) ---
    if "NN binary" in predictions and hbi_h5_path and hbi_h5_path.exists():
        from toxfam.evaluation.confidence_routing import evaluate_confidence_routing

        routing_result = evaluate_confidence_routing(
            nn_p_toxic=predictions["NN binary"],
            hbi_h5_path=hbi_h5_path,
            identifiers=test_ids,
            y_true=y_true,
            evalue_threshold=0.1,
        )
        # Build combined p_toxic for curves
        hbi_feats = np.zeros((len(test_ids), 4), dtype=np.float32)
        with h5py.File(str(hbi_h5_path), "r") as f:
            for i, sid in enumerate(test_ids):
                if sid in f:
                    hbi_feats[i] = f[sid][:]
        use_hbi = (hbi_feats[:, 0] > 0) & (hbi_feats[:, 3] >= 0.1)
        combined = predictions["NN binary"].copy()
        combined[use_hbi] = hbi_feats[use_hbi, 1]
        predictions["Confidence routing"] = combined
        metrics["Confidence routing"] = routing_result

    return predictions, metrics


def run_full_comparison(
    training_csv: Path | None = None,
    hbi_h5_path: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Run the full method comparison pipeline.

    1. Load test data and ground truth
    2. Collect predictions from all methods
    3. Run per-family evaluation
    4. Generate all publication figures
    5. Save everything
    """
    root = get_project_root()
    model_base = root / "model" / "model_output"

    if training_csv is None:
        training_csv = root / "data" / "processed" / "training_data.csv"
    if hbi_h5_path is None:
        # Prefer counterpart-expanded HBI if available
        cp_hbi = root / "data" / "intermediate" / "hbi" / "hbi_features_with_counterparts.h5"
        orig_hbi = root / "data" / "intermediate" / "hbi" / "hbi_features.h5"
        hbi_h5_path = cp_hbi if cp_hbi.exists() else (orig_hbi if orig_hbi.exists() else None)
    if output_dir is None:
        output_dir = model_base / "comparison"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    df = pd.read_csv(training_csv)
    test_df = df[df["Split"] == "test"].copy().reset_index(drop=True)
    test_df["is_toxic"] = test_df["Protein families"].apply(
        lambda x: 0 if to_binary_class(x) == "nontoxin" else 1
    )
    y_true = test_df["is_toxic"].to_numpy()

    print(f"Test set: {len(test_df)} sequences ({y_true.sum()} toxic, {len(y_true) - y_true.sum()} nontox)")

    # Collect all predictions
    print("\n=== Collecting predictions from all methods ===")
    predictions, all_metrics = collect_all_predictions(
        test_df, y_true, model_base, hbi_h5_path
    )

    # Print summary table
    print("\n" + "=" * 80)
    print("METHOD COMPARISON SUMMARY (Test Set)")
    print("=" * 80)
    print(f"{'Method':<25} {'ROC-AUC':>9} {'PR-AUC':>9} {'F1':>9} {'MCC':>9} {'Acc':>9}")
    print("-" * 80)
    for name, m in all_metrics.items():
        roc = m.get("roc_auc", "—")
        pr = m.get("pr_auc", "—")
        f1 = m.get("f1", "—")
        mcc = m.get("mcc", "—")
        acc = m.get("accuracy", "—")
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
        print(f"  {name:<23} {fmt(roc):>9} {fmt(pr):>9} {fmt(f1):>9} {fmt(mcc):>9} {fmt(acc):>9}")
    print("=" * 80)

    # Save summary metrics
    serializable_metrics = {}
    for name, m in all_metrics.items():
        serializable_metrics[name] = {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in m.items()
            if k not in ("fpr", "tpr", "precision_curve", "recall_curve",
                         "roc_thresholds", "pr_thresholds")
        }
    (output_dir / "all_metrics.json").write_text(json.dumps(serializable_metrics, indent=2))

    # --- Per-family evaluation ---
    print("\n=== Per-Family Evaluation ===")
    # Only include methods with per-sample predictions (not ToxinPred2 which only has summary metrics)
    from toxfam.evaluation.per_family_eval import evaluate_per_family

    per_family_df = evaluate_per_family(
        test_df=test_df,
        y_true=y_true,
        predictions=predictions,
        output_dir=output_dir,
    )
    print(f"  Per-family metrics: {len(per_family_df)} families evaluated")

    # --- Publication figures ---
    print("\n=== Generating Publication Figures ===")
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    from toxfam.visualization.publication import (
        generate_all_figures,
        plot_error_venn,
    )

    generate_all_figures(
        output_dir=fig_dir,
        training_csv=training_csv,
        metrics=serializable_metrics,
        y_true=y_true,
        predictions=predictions,
        per_family_df=per_family_df,
    )

    # Additional Venn diagram: NN augmented vs HBI
    if "NN augmented+CP" in predictions and "HBI best-hit" in predictions:
        plot_error_venn(
            y_true, predictions["NN augmented+CP"], predictions["HBI best-hit"],
            "NN augmented+CP", "HBI best-hit",
            fig_dir / "fig9b_error_venn_aug_vs_hbi.png",
        )

    print(f"\nAll results saved to: {output_dir}")
