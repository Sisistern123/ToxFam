"""Evaluate model on test set: compares NN vs HBI sequence similarity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from pymmseqs.commands import createdb, search

from toxfam._paths import get_project_root
from toxfam.evaluation.metrics import calculate_multiclass_metrics
from toxfam.visualization.plots import plot_confusion_matrix


def _default_paths():
    root = get_project_root()
    bench_dir = root / "benchmark"
    new_dir = bench_dir / "new"
    return {
        "eval_dir": new_dir / "evaluation" / "test_set",
        "results_dir": new_dir / "evaluation" / "test_set" / "results",
        "test_data": bench_dir / "test_data.csv",
        "test_fasta": bench_dir / "test_data.fasta",
        "train_data": bench_dir / "HBI" / "train_all_df.csv",
        "train_fasta": bench_dir / "HBI" / "train_all_members.fasta",
    }


def load_data(test_data_path: Path, train_data_path: Path):
    print("Loading data...")
    test_df = pd.read_csv(test_data_path)
    train_df = pd.read_csv(train_data_path)

    q_labels = set(train_df["Protein families"].unique())
    t_labels = set(test_df["Protein families"].unique())
    only_in_train = q_labels - t_labels

    if only_in_train:
        print(
            f"Found {len(only_in_train)} labels in train not in test. "
            f"Mapping to 'other'..."
        )
        repl_map = {lbl: "other" for lbl in only_in_train}
        train_df["Protein families"] = train_df["Protein families"].replace(repl_map)

    return test_df, train_df


def run_hbi_search(
    test_df, train_df, test_fasta: Path, train_fasta: Path, results_dir: Path
):
    print("Running HBI evaluation...")
    tmp_dir = results_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    query_db = createdb(str(test_fasta), str(tmp_dir / "query_db"))
    target_db = createdb(str(train_fasta), str(tmp_dir / "train_db"))

    search_res = search(
        query_db.to_path(),
        target_db.to_path(),
        str(tmp_dir / "search_res"),
        str(tmp_dir / "tmp"),
        s=9,
        e="inf",
        min_seq_id=0.0,
        max_seqs=100_000,
    )

    res = search_res.to_pandas()
    if res.empty:
        print("No hits found.")
        return pd.DataFrame(columns=["identifier", "hbi_prediction", "hbi_confidence"])

    best_hits = res.loc[res.groupby("query")["evalue"].idxmin()].reset_index(drop=True)

    train_labels = train_df[["identifier", "Protein families"]].rename(
        columns={"identifier": "target", "Protein families": "hbi_prediction"}
    )
    hbi_results = best_hits.merge(train_labels, on="target", how="left")

    hbi_results = hbi_results.rename(columns={"query": "identifier"})
    hbi_results["hbi_confidence"] = hbi_results["fident"]

    return hbi_results[["identifier", "hbi_prediction", "hbi_confidence"]]


def load_nn_precomputed(nn_preds_path: Path, nn_metrics_path: Path) -> Dict[str, Any]:
    if not (nn_metrics_path.exists() and nn_preds_path.exists()):
        print("Warning: NN pre-calculated files missing!")
        return {"has_nn": False}

    print("Loading pre-calculated NN results...")

    with open(nn_metrics_path, "r") as f:
        nn_json = json.load(f)

    numeric = nn_json.get("numeric_metrics", {})
    nn_acc = numeric.get("accuracy", numeric.get("Test_Accuracy", None))

    report = nn_json.get("classification_report", {})
    nn_mcc_proxy = report.get("macro avg", {}).get("f1-score", 0.0)

    nn_preds = pd.read_csv(nn_preds_path)

    if "predicted_label" in nn_preds.columns:
        pred_col = "predicted_label"
    elif "Predicted" in nn_preds.columns:
        pred_col = "Predicted"
    else:
        candidates = [c for c in nn_preds.columns if c != "identifier"]
        pred_col = candidates[0] if candidates else None

    if pred_col is None:
        print("Warning: Could not find prediction column in NN predictions CSV!")
        return {"has_nn": False}

    return {
        "has_nn": True,
        "nn_json": nn_json,
        "nn_acc": nn_acc if nn_acc is not None else 0.0,
        "nn_mcc_proxy": nn_mcc_proxy,
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

    summary_metrics = []

    # 1) Load data
    test_df, train_df = load_data(paths["test_data"], paths["train_data"])

    # 2) HBI search
    hbi_results = run_hbi_search(
        test_df, train_df, paths["test_fasta"], paths["train_fasta"], results_dir
    )

    # 3) Combine + align
    combined = test_df[["identifier", "Sequence", "Protein families"]].copy()
    combined.rename(columns={"Protein families": "ground_truth"}, inplace=True)

    combined = combined.merge(hbi_results, on="identifier", how="left")
    combined["hbi_prediction"] = combined["hbi_prediction"].fillna("no hit")

    valid_labels = set(combined["ground_truth"].unique())
    hbi_labels = set(combined["hbi_prediction"].unique())
    labels_not_in_ground_truth = hbi_labels - valid_labels - {"no hit"}

    if labels_not_in_ground_truth:
        print(
            f"Found {len(labels_not_in_ground_truth)} HBI labels not in ground truth. "
            f"Mapping to 'other'..."
        )
        repl_map = {lbl: "other" for lbl in labels_not_in_ground_truth}
        combined["hbi_prediction"] = combined["hbi_prediction"].replace(repl_map)

    # 4) NN
    nn_bundle = load_nn_precomputed(nn_preds_path, nn_metrics_path)
    has_model = nn_bundle.get("has_nn", False)

    if has_model:
        nn_preds = nn_bundle["nn_preds_df"]
        pred_col = nn_bundle["nn_pred_col"]
        print(f"Using '{pred_col}' column for NN predictions")

        model_subset = nn_preds[["identifier", pred_col]].rename(
            columns={pred_col: "model_prediction"}
        )
        combined = combined.merge(model_subset, on="identifier", how="left")
        combined["model_prediction"] = combined["model_prediction"].fillna("no hit")

        model_labels = set(combined["model_prediction"].unique())
        labels_not_in_ground_truth_model = model_labels - valid_labels - {"no hit"}

        if labels_not_in_ground_truth_model:
            print(
                f"Found {len(labels_not_in_ground_truth_model)} model labels "
                f"not in ground truth. Mapping to 'other'..."
            )
            repl_map_model = {lbl: "other" for lbl in labels_not_in_ground_truth_model}
            combined["model_prediction"] = combined["model_prediction"].replace(
                repl_map_model
            )

    # 5) Shared class list
    shared_class_list = sorted(list(combined["ground_truth"].unique()))
    if "no hit" in set(combined["hbi_prediction"].unique()) or (
        "model_prediction" in combined
        and "no hit" in set(combined["model_prediction"].unique())
    ):
        shared_class_list = shared_class_list + ["no hit"]
    print(f"\nShared class list created: {len(shared_class_list)} classes")

    # 6) HBI metrics
    print("\nCalculating HBI Metrics...")
    hbi_m = calculate_multiclass_metrics(
        combined, "ground_truth", "hbi_prediction", shared_class_list=shared_class_list
    )
    summary_metrics.append(
        {
            "Method": "HBI (Sequence Similarity)",
            "Accuracy": hbi_m["acc"],
            "MCC": hbi_m["mcc"],
            "Micro_MCC": hbi_m["micro_mcc"],
            "Std_Error": hbi_m["std_error"],
            "Sample_Size": hbi_m["n_samples"],
        }
    )
    print(
        f"   -> Accuracy: {hbi_m['acc']:.4f} (+/-{hbi_m['std_error']:.4f}) "
        f"| MCC: {hbi_m['mcc']:.4f} | Micro-MCC: {hbi_m['micro_mcc']:.4f}"
    )

    # 7) NN metrics
    nn_json = None
    if has_model:
        nn_json = nn_bundle["nn_json"]
        nn_acc = float(nn_bundle["nn_acc"])
        nn_mcc_proxy = float(nn_bundle["nn_mcc_proxy"])

        n_samples = len(combined)
        nn_std_error = (
            np.sqrt((nn_acc * (1 - nn_acc)) / n_samples)
            if n_samples > 0
            else float("nan")
        )

        summary_metrics.append(
            {
                "Method": "Neural Network (Calibrated, precomputed)",
                "Accuracy": nn_acc,
                "MCC": nn_mcc_proxy,
                "Micro_MCC": np.nan,
                "Std_Error": nn_std_error,
                "Sample_Size": n_samples,
            }
        )
        print(
            f"NN (precomputed) -> Accuracy: {nn_acc:.4f} "
            f"(+/-{nn_std_error:.4f}) | 'MCC' (macro-F1 proxy): {nn_mcc_proxy:.4f}"
        )

    # 8) Save outputs
    combined.to_csv(results_dir / "test_comparison_results.csv", index=False)

    metrics_df = pd.DataFrame(summary_metrics)
    metrics_path = results_dir / "metric_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)

    report_data = {
        "HBI": {
            "numeric_metrics": {
                "Test_Accuracy": hbi_m["acc"],
                "Test_MCC": hbi_m["mcc"],
                "Test_Micro_MCC": hbi_m["micro_mcc"],
                "Test_Std_Error": hbi_m["std_error"],
            },
            "classification_report": hbi_m["report"],
        },
        "Neural_Network": nn_json if has_model else "Missing precomputed files",
    }
    with open(results_dir / "full_classification_report.json", "w") as f:
        json.dump(report_data, f, indent=4)

    # 9) Visualizations
    plot_confusion_matrix(
        all_labels=hbi_m["y_true_encoded"],
        all_preds=hbi_m["y_pred_encoded"],
        label_encoder=type("obj", (object,), {"classes_": hbi_m["class_list"]})(),
        output_path=str(results_dir / "hbi_confusion_matrix.png"),
    )

    if has_model:
        cls2idx = {cls_name: i for i, cls_name in enumerate(shared_class_list)}
        nn_y_true = combined["ground_truth"].map(cls2idx).to_numpy()
        nn_y_pred = combined["model_prediction"].map(cls2idx).to_numpy()

        if np.isnan(nn_y_true).sum() == 0 and np.isnan(nn_y_pred).sum() == 0:
            plot_confusion_matrix(
                all_labels=nn_y_true,
                all_preds=nn_y_pred,
                label_encoder=type("obj", (object,), {"classes_": shared_class_list})(),
                output_path=str(results_dir / "model_confusion_matrix.png"),
            )
        else:
            print(
                "Skipping NN confusion matrix due to unmapped labels after alignment."
            )

    print(f"\nDone! Summary saved to: {metrics_path}")
    print(f"Detailed Results saved to: {results_dir}")
