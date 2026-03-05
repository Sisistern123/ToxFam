"""k-Fold cross-validation at the cluster level."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

if TYPE_CHECKING:
    from toxfam.config import TrainConfig


def run_kfold_training(
    config: TrainConfig,
    *,
    n_folds: int = 5,
    cluster_tsv: Path | None = None,
) -> dict:
    """Run k-fold cross-validation at the cluster level.

    1. Load training_data.csv and cluster assignments
    2. Hold out the existing test set (keep Split=="test" fixed)
    3. Use MultilabelStratifiedKFold on train+val clusters to create k folds
    4. For each fold: re-assign Split, run run_training(config), collect metrics
    5. Aggregate: mean +/- std for all metrics across folds

    Returns dict with per-fold and aggregated metrics.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    from toxfam._paths import intermediate_dir
    from toxfam.training.orchestrator import run_training

    out_root = Path(config.output_dir)
    kfold_dir = out_root / "kfold"
    kfold_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(config.input_csv)

    # Identify test set (fixed across folds)
    test_df = df[df["Split"] == "test"].copy()
    trainval_df = df[df["Split"].isin(["train", "val"])].copy()

    # Load cluster assignments
    if cluster_tsv is None:
        cluster_tsv = intermediate_dir() / "identity_splits" / "global_cluster_cluster.tsv"

    if cluster_tsv.exists():
        rep_to_cluster: dict[str, int] = {}
        cluster_map: dict[str, int] = {}
        cluster_id = 0
        with open(cluster_tsv) as f:
            for line in f:
                rep, member = line.strip().split("\t")
                if rep not in cluster_map:
                    cluster_map[rep] = cluster_id
                    cluster_id += 1
                rep_to_cluster[member] = cluster_map[rep]

        trainval_df["_cluster_id"] = trainval_df["identifier"].map(rep_to_cluster)
        max_cid = trainval_df["_cluster_id"].max() if trainval_df["_cluster_id"].notna().any() else -1
        missing = trainval_df["_cluster_id"].isna()
        if missing.any():
            trainval_df.loc[missing, "_cluster_id"] = range(
                int(max_cid) + 1, int(max_cid) + 1 + int(missing.sum())
            )
        trainval_df["_cluster_id"] = trainval_df["_cluster_id"].astype(int)
    else:
        # Fallback: each protein is its own cluster
        print(f"Warning: cluster TSV not found at {cluster_tsv}, using per-protein clusters")
        trainval_df["_cluster_id"] = range(len(trainval_df))

    # Build cluster-level DataFrame for stratified k-fold
    cluster_groups = trainval_df.groupby("_cluster_id")
    cluster_df = pd.DataFrame({
        "_cluster_id": list(cluster_groups.groups.keys()),
        "families": [
            list(g["Protein families"].unique()) for _, g in cluster_groups
        ],
    })

    mlb = MultiLabelBinarizer()
    Y_clusters = mlb.fit_transform(cluster_df["families"])

    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []

    for fold_idx, (train_cidx, val_cidx) in enumerate(mskf.split(cluster_df, Y_clusters)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")

        train_cluster_ids = set(cluster_df.iloc[train_cidx]["_cluster_id"])

        # Assign splits
        fold_trainval = trainval_df.copy()
        fold_trainval["Split"] = fold_trainval["_cluster_id"].apply(
            lambda cid: "train" if cid in train_cluster_ids else "val"
        )

        # Combine with test
        fold_df = pd.concat(
            [fold_trainval.drop(columns=["_cluster_id"]), test_df],
            ignore_index=True,
        )

        # Write temporary CSV for this fold
        fold_dir = kfold_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_csv = fold_dir / "training_data.csv"
        fold_df.to_csv(fold_csv, index=False)

        # Run training with modified config
        fold_config = config.model_copy(update={
            "output_dir": fold_dir,
            "input_csv": fold_csv,
            "n_folds": 1,  # prevent recursion
        })

        run_training(fold_config)

        # Collect metrics
        metrics_file = fold_dir / "metrics" / "binary_test_calibrated_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                fold_metrics.append(json.load(f))

    # Aggregate
    summary = _aggregate_fold_metrics(fold_metrics)
    summary["n_folds"] = n_folds
    summary["per_fold"] = fold_metrics

    summary_path = kfold_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n{'='*60}")
    print(f"k-FOLD SUMMARY ({n_folds} folds)")
    print(f"{'='*60}")
    for metric in ("roc_auc", "pr_auc", "f1", "mcc", "accuracy"):
        if f"{metric}_mean" in summary:
            print(
                f"  {metric}: {summary[f'{metric}_mean']:.4f} "
                f"+/- {summary[f'{metric}_std']:.4f}"
            )

    print(f"\nSaved to {summary_path}")
    return summary


def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Compute mean and std across folds for numeric metrics."""
    if not fold_metrics:
        return {}

    keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))]
    result = {}
    for key in keys:
        values = [m[key] for m in fold_metrics if key in m]
        if values:
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
    return result
