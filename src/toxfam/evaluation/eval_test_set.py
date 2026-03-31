"""Evaluate model on test set: compares NN vs HBI sequence similarity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console

from toxfam._paths import benchmark_dir, get_project_root, processed_dir
from toxfam.evaluation.hbi import NO_HIT_LABEL, run_hbi_search, write_fasta_from_df
from toxfam.evaluation.metrics import (
    MetricsResult,
    calculate_metrics,
    print_metrics_table,
)
from toxfam.visualization.plots import plot_confusion_matrix

console = Console()


def _default_paths():
    proc = processed_dir()
    return {
        "results_dir": benchmark_dir() / "test_set",
        "training_data": proc / "training_data.csv",
        "train_hbi_data": proc / "hbi_train_all.csv",
        "train_hbi_fasta": proc / "hbi_train_all.fasta",
    }


def _load_split(training_data_path: Path, split: str) -> pd.DataFrame:
    """Load a specific split from training_data.csv."""
    df = pd.read_csv(training_data_path)
    return df[df["Split"] == split].reset_index(drop=True)


def load_data(training_data_path: Path, train_hbi_path: Path):
    console.print("Loading data...")
    test_df = _load_split(training_data_path, "test")
    train_df = pd.read_csv(train_hbi_path)

    q_labels = set(train_df["Protein families"].unique())
    t_labels = set(test_df["Protein families"].unique())
    only_in_train = q_labels - t_labels

    if only_in_train:
        console.print(
            f"   Found {len(only_in_train)} labels in train not in test. "
            f"Mapping to 'other'..."
        )
        repl_map = {lbl: "other" for lbl in only_in_train}
        train_df["Protein families"] = train_df["Protein families"].replace(repl_map)

    return test_df, train_df


def load_nn_precomputed(nn_preds_path: Path, nn_metrics_path: Path) -> dict[str, Any]:
    if not (nn_metrics_path.exists() and nn_preds_path.exists()):
        console.print("[yellow]Warning: NN pre-calculated files missing![/]")
        return {"has_nn": False}

    console.print("Loading pre-calculated NN results...")

    with open(nn_metrics_path, "r") as f:
        nn_json = json.load(f)

    nn_preds = pd.read_csv(nn_preds_path)

    if "predicted_label" in nn_preds.columns:
        pred_col = "predicted_label"
    elif "Predicted" in nn_preds.columns:
        pred_col = "Predicted"
    else:
        candidates = [c for c in nn_preds.columns if c != "identifier"]
        pred_col = candidates[0] if candidates else None

    if pred_col is None:
        console.print("[yellow]Warning: Could not find prediction column in NN CSV![/]")
        return {"has_nn": False}

    return {
        "has_nn": True,
        "nn_json": nn_json,
        "nn_preds_df": nn_preds,
        "nn_pred_col": pred_col,
    }


def run_eval_test_set(model_dir: Path | None = None) -> None:
    """Run full test-set evaluation (HBI + NN comparison)."""
    paths = _default_paths()
    results_dir = paths["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if model_dir is None:
        model_dir = (
            get_project_root() / "model" / "model_output" / "calibrated_combined"
        )

    nn_preds_path = model_dir / "predictions" / "test_calibrated_predictions.csv"
    nn_metrics_path = model_dir / "metrics" / "test_calibrated_metrics.json"

    # 1) Load data
    test_df, train_df = load_data(paths["training_data"], paths["train_hbi_data"])

    # 2) Generate test FASTA and run HBI search
    test_fasta = results_dir / "tmp" / "test_query.fasta"
    test_fasta.parent.mkdir(parents=True, exist_ok=True)
    write_fasta_from_df(test_df, test_fasta)

    hbi_result = run_hbi_search(
        query_fasta=test_fasta,
        target_fasta=paths["train_hbi_fasta"],
        target_labels_df=train_df,
        work_dir=results_dir / "tmp",
    )
    console.print(
        f"   HBI Coverage: {hbi_result.coverage:.1%} "
        f"({hbi_result.n_with_hits}/{hbi_result.n_queries})"
    )

    # 3) Combine with ground truth
    combined = test_df[["identifier", "Sequence", "Protein families"]].copy()
    combined.rename(columns={"Protein families": "ground_truth"}, inplace=True)
    combined = combined.merge(hbi_result.predictions, on="identifier", how="left")
    combined["hbi_prediction"] = combined["hbi_prediction"].fillna(NO_HIT_LABEL)

    # Handle HBI labels not in ground truth
    valid_labels = set(combined["ground_truth"].unique())
    hbi_labels = set(combined["hbi_prediction"].unique())
    unknown_labels = hbi_labels - valid_labels - {NO_HIT_LABEL}
    if unknown_labels:
        console.print(
            f"   Mapping {len(unknown_labels)} HBI labels not in ground truth "
            f"to 'other'"
        )
        combined["hbi_prediction"] = combined["hbi_prediction"].replace(
            {lbl: "other" for lbl in unknown_labels}
        )

    # 4) Compute HBI metrics (class_list from ground truth only)
    class_list = sorted(combined["ground_truth"].unique())
    hbi_metrics = calculate_metrics(
        combined["ground_truth"],
        combined["hbi_prediction"],
        class_list=class_list,
    )

    # 5) Load NN precomputed results
    nn_bundle = load_nn_precomputed(nn_preds_path, nn_metrics_path)
    has_model = nn_bundle.get("has_nn", False)
    nn_metrics: MetricsResult | None = None

    if has_model:
        nn_preds = nn_bundle["nn_preds_df"]
        pred_col = nn_bundle["nn_pred_col"]
        console.print(f"   Using '{pred_col}' column for NN predictions")

        model_subset = nn_preds[["identifier", pred_col]].rename(
            columns={pred_col: "model_prediction"}
        )
        combined = combined.merge(model_subset, on="identifier", how="left")
        combined["model_prediction"] = combined["model_prediction"].fillna(NO_HIT_LABEL)

        # Handle model labels not in ground truth
        model_labels = set(combined["model_prediction"].unique())
        unknown_model = model_labels - valid_labels - {NO_HIT_LABEL}
        if unknown_model:
            console.print(
                f"   Mapping {len(unknown_model)} model labels not in ground "
                f"truth to 'other'"
            )
            combined["model_prediction"] = combined["model_prediction"].replace(
                {lbl: "other" for lbl in unknown_model}
            )

        nn_metrics = calculate_metrics(
            combined["ground_truth"],
            combined["model_prediction"],
            class_list=class_list,
        )

    # 6) Print summary
    summary = {"HBI": hbi_metrics}
    if nn_metrics is not None:
        summary["Neural Network"] = nn_metrics
    print_metrics_table(summary)

    # 7) Save outputs
    combined.to_csv(results_dir / "test_comparison_results.csv", index=False)

    summary_rows = [hbi_metrics.to_summary_dict("HBI (Sequence Similarity)")]
    if nn_metrics is not None:
        summary_rows.append(nn_metrics.to_summary_dict("Neural Network (Calibrated)"))
    pd.DataFrame(summary_rows).to_csv(
        results_dir / "metric_comparison.csv", index=False
    )

    report_data: dict[str, Any] = {"HBI": hbi_metrics.to_json_dict()}
    if nn_metrics is not None:
        report_data["Neural_Network"] = nn_metrics.to_json_dict()
    elif has_model:
        report_data["Neural_Network"] = nn_bundle.get("nn_json", {})
    else:
        report_data["Neural_Network"] = "Missing precomputed files"

    with open(results_dir / "full_classification_report.json", "w") as f:
        json.dump(report_data, f, indent=4)

    # 8) Confusion matrices
    plot_confusion_matrix(
        hbi_metrics.y_true_encoded,
        hbi_metrics.y_pred_encoded,
        class_list,
        str(results_dir / "hbi_confusion_matrix.png"),
    )

    if nn_metrics is not None:
        plot_confusion_matrix(
            nn_metrics.y_true_encoded,
            nn_metrics.y_pred_encoded,
            class_list,
            str(results_dir / "model_confusion_matrix.png"),
        )

    console.print(f"\nResults saved to: {results_dir}")
