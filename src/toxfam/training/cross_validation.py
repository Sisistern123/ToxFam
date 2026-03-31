"""k-Fold cross-validation at cluster level.

Holds out the test set, then performs k-fold stratified splitting of
train+val. Each fold trains independently and results are aggregated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from toxfam.config import TrainConfig

console = Console()


def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Compute mean and std across fold metric dicts.

    Parameters
    ----------
    fold_metrics : list of dicts, each with numeric metric values.

    Returns
    -------
    dict with {key}_mean and {key}_std for each numeric key.
    """
    if not fold_metrics:
        return {}
    keys = fold_metrics[0].keys()
    result = {}
    for key in keys:
        values = [
            m[key] for m in fold_metrics if isinstance(m.get(key), (int, float))
        ]
        if values:
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
    return result


def run_kfold_training(config: TrainConfig) -> None:
    """Run k-fold cross-validation, holding out the test set.

    For each fold, train+val is re-split using stratified sampling,
    the model is trained via run_training, and test metrics are collected.
    After all folds, mean ± std are reported and saved.
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedShuffleSplit

    from toxfam.data.dataset import analyze_data_splits
    from toxfam.training.orchestrator import run_training

    n_folds = config.n_folds
    if n_folds <= 1:
        console.print("n_folds <= 1, running single training.")
        run_training(config)
        return

    console.print(f"[bold]Running {n_folds}-fold cross-validation[/bold]")

    df = pd.read_csv(config.input_csv)
    train_df, val_df, test_df = analyze_data_splits(df)

    # Combine train + val for re-splitting per fold
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    fold_metrics: list[dict] = []
    out_root = Path(config.output_dir)

    for fold_idx in range(n_folds):
        console.print(f"\n[bold]--- Fold {fold_idx + 1}/{n_folds} ---[/bold]")

        fold_dir = out_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Stratified split for this fold
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.15, random_state=config.seed + fold_idx
        )
        train_idx, val_idx = next(
            splitter.split(trainval_df, trainval_df["Protein families"])
        )

        fold_train = trainval_df.iloc[train_idx].copy()
        fold_val = trainval_df.iloc[val_idx].copy()
        fold_train["Split"] = "train"
        fold_val["Split"] = "val"
        test_copy = test_df.copy()
        test_copy["Split"] = "test"

        fold_df = pd.concat([fold_train, fold_val, test_copy], ignore_index=True)

        # Write fold-specific CSV
        fold_csv = fold_dir / "training_data.csv"
        fold_df.to_csv(fold_csv, index=False)

        fold_config = config.model_copy(
            update={"output_dir": fold_dir, "input_csv": fold_csv, "n_folds": 1}
        )

        run_training(fold_config)

        # Collect test metrics
        metrics_path = fold_dir / "metrics" / "test_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
                fold_metrics.append(data.get("numeric_metrics", {}))

    # Aggregate
    summary = _aggregate_fold_metrics(fold_metrics)
    summary_path = out_root / "kfold_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    console.print(f"\n[bold]k-Fold summary saved to {summary_path}[/bold]")
    for k, v in summary.items():
        console.print(f"  {k}: {v:.4f}")
