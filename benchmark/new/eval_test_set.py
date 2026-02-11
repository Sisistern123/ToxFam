#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report
from sklearn.preprocessing import label_binarize
from pymmseqs.commands import createdb, search

from model.visualization import plot_confusion_matrix

# ---------- Constants & Paths ----------
BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "evaluation" / "test_set"
RESULTS_DIR = EVAL_DIR / "results"
TMP_DIR = Path("tmp")

# Data Paths
TEST_DATA = BASE_DIR.parent / "test_data.csv"
TEST_FASTA = BASE_DIR.parent / "test_data.fasta"
TRAIN_DATA = BASE_DIR.parent / "HBI" / "train_all_df.csv"
TRAIN_FASTA = BASE_DIR.parent / "HBI" / "train_all_members.fasta"

# NN precomputed outputs (script 2 style)
MODEL_DIR = BASE_DIR.parent.parent / "model" / "model_output" / "calibrated_combined"
NN_PREDS_PATH = MODEL_DIR / "test_calibrated_predictions.csv"
NN_METRICS_PATH = MODEL_DIR / "test_calibrated_metrics.json"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Step 1: Load and Prepare Data ----------
def load_data():
    """Loads CSVs and aligns labels according to notebook logic."""
    print("📂 Loading data...")
    test_df = pd.read_csv(TEST_DATA)
    train_df = pd.read_csv(TRAIN_DATA)

    # Notebook Logic: Replace training labels not in validation/test with "other"
    q_labels = set(train_df["Protein families"].unique())
    t_labels = set(test_df["Protein families"].unique())
    only_in_train = q_labels - t_labels

    if only_in_train:
        print(f"⚠️  Found {len(only_in_train)} labels in train not in test. Mapping to 'other'...")
        print(f"Labels: {only_in_train}")
        repl_map = {lbl: "other" for lbl in only_in_train}
        train_df["Protein families"] = train_df["Protein families"].replace(repl_map)

    return test_df, train_df


# ---------- Step 2: HBI Search (Notebook Flow) ----------
def run_hbi_search(test_df, train_df):
    """Runs MMseqs2 search exactly as the notebook does."""
    print("🔍 Running HBI evaluation (Notebook Style)...")

    query_db = createdb(str(TEST_FASTA), str(TMP_DIR / "query_db"))
    target_db = createdb(str(TRAIN_FASTA), str(TMP_DIR / "train_db"))

    search_res = search(
        query_db.to_path(),
        target_db.to_path(),
        str(TMP_DIR / "search_res"),
        str(TMP_DIR / "tmp"),
        s=9,
        e="inf",
        min_seq_id=0.0,
        max_seqs=100_000
    )

    res = search_res.to_pandas()
    if res.empty:
        print("❌ No hits found.")
        return pd.DataFrame(columns=["identifier", "hbi_prediction", "hbi_confidence"])

    # Best hits by minimal E-value
    best_hits = res.loc[res.groupby("query")["evalue"].idxmin()].reset_index(drop=True)

    train_labels = train_df[["identifier", "Protein families"]].rename(
        columns={"identifier": "target", "Protein families": "hbi_prediction"}
    )
    hbi_results = best_hits.merge(train_labels, on="target", how="left")

    hbi_results = hbi_results.rename(columns={"query": "identifier"})
    # Notebook uses fident as "confidence"
    hbi_results["hbi_confidence"] = hbi_results["fident"]

    return hbi_results[["identifier", "hbi_prediction", "hbi_confidence"]]


# ---------- Step 3: Metrics (Notebook Style) ----------
def calculate_metrics_bundle(
    df: pd.DataFrame,
    pred_col: str,
    truth_col: str = "ground_truth",
    shared_class_list: Optional[list] = None
) -> Dict[str, Any]:
    """Calculates comprehensive metrics EXACTLY like the notebook."""
    if shared_class_list is not None:
        class_list = shared_class_list
        print(f"   Using shared class list with {len(class_list)} classes")
    else:
        class_list = sorted(list(set(df[truth_col].unique()) | set(df[pred_col].unique())))
        print(f"   Created class list with {len(class_list)} classes")

    cls2idx = {cls_name: i for i, cls_name in enumerate(class_list)}

    # Encode
    y_true = df[truth_col].map(cls2idx).to_numpy()
    y_pred = df[pred_col].map(cls2idx).to_numpy()

    # Guard: any unmapped => NaN
    unmapped_true = np.isnan(y_true).sum()
    unmapped_pred = np.isnan(y_pred).sum()
    if unmapped_true > 0 or unmapped_pred > 0:
        print(f"   ⚠️ WARNING: {unmapped_true} unmapped true labels, {unmapped_pred} unmapped predictions")

    n_samples = len(y_true)
    n_classes = len(class_list)

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Micro-MCC (flattened one-hots)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_pred_bin = label_binarize(y_pred, classes=range(n_classes))

    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        y_pred_bin = np.hstack((1 - y_pred_bin, y_pred_bin))

    micro_mcc = matthews_corrcoef(y_true_bin.ravel(), y_pred_bin.ravel())

    std_error = np.sqrt((acc * (1 - acc)) / n_samples) if n_samples > 0 else float("nan")

    report = classification_report(
        y_true,
        y_pred,
        labels=range(n_classes),  # explicit labels, like notebook
        target_names=class_list,
        output_dict=True,
        zero_division=0
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
        "y_pred_encoded": y_pred
    }


# ---------- Step 4: NN logic (Script 2 style) ----------
def load_nn_precomputed() -> Dict[str, Any]:
    """
    Script-2 style:
      - load metrics.json and take numeric_metrics["accuracy"]
      - use macro avg f1-score as MCC proxy (if real MCC not present)
      - load predictions CSV for merged comparison file
    """
    if not (NN_METRICS_PATH.exists() and NN_PREDS_PATH.exists()):
        print("⚠️ Warning: NN pre-calculated files missing!")
        return {"has_nn": False}

    print("✅ Loading pre-calculated NN results...")

    with open(NN_METRICS_PATH, "r") as f:
        nn_json = json.load(f)

    # Expect script-2 style keys; handle a couple variants defensively
    numeric = nn_json.get("numeric_metrics", {})
    nn_acc = numeric.get("accuracy", numeric.get("Test_Accuracy", None))

    # Script 2 behavior: MCC proxy from macro avg F1 if MCC missing
    report = nn_json.get("classification_report", {})
    nn_mcc_proxy = report.get("macro avg", {}).get("f1-score", 0.0)

    # Load predictions
    nn_preds = pd.read_csv(NN_PREDS_PATH)

    # Most common column name in your scripts
    if "predicted_label" in nn_preds.columns:
        pred_col = "predicted_label"
    elif "Predicted" in nn_preds.columns:
        pred_col = "Predicted"
    else:
        # fallback: pick a non-identifier column
        candidates = [c for c in nn_preds.columns if c != "identifier"]
        pred_col = candidates[0] if candidates else None

    if pred_col is None:
        print("⚠️ Warning: Could not find prediction column in NN predictions CSV!")
        return {"has_nn": False}

    return {
        "has_nn": True,
        "nn_json": nn_json,
        "nn_acc": nn_acc if nn_acc is not None else 0.0,
        "nn_mcc_proxy": nn_mcc_proxy,
        "nn_preds_df": nn_preds,
        "nn_pred_col": pred_col
    }


def main():
    ensure_dirs()
    summary_metrics = []

    # 1) Load data
    test_df, train_df = load_data()

    # 2) HBI search
    hbi_results = run_hbi_search(test_df, train_df)

    # 3) Combine + align
    combined = test_df[["identifier", "Sequence", "Protein families"]].copy()
    combined.rename(columns={"Protein families": "ground_truth"}, inplace=True)

    combined = combined.merge(hbi_results, on="identifier", how="left")
    combined["hbi_prediction"] = combined["hbi_prediction"].fillna("no hit")

    # Align HBI prediction labels to ground-truth label space
    valid_labels = set(combined["ground_truth"].unique())
    hbi_labels = set(combined["hbi_prediction"].unique())

    labels_not_in_ground_truth = hbi_labels - valid_labels

    # IMPORTANT: exclude "no hit" from mapping — it must stay a "no prediction" bucket
    labels_not_in_ground_truth = labels_not_in_ground_truth - {"no hit"}

    if labels_not_in_ground_truth:
        print(f"⚠️  Found {len(labels_not_in_ground_truth)} HBI labels not in ground truth. Mapping to 'other'...")
        print(f"   Labels: {labels_not_in_ground_truth}")
        repl_map = {lbl: "other" for lbl in labels_not_in_ground_truth}
        combined["hbi_prediction"] = combined["hbi_prediction"].replace(repl_map)

    # 4) NN (script 2 logic)
    nn_bundle = load_nn_precomputed()
    has_model = nn_bundle.get("has_nn", False)

    if has_model:
        nn_preds = nn_bundle["nn_preds_df"]
        pred_col = nn_bundle["nn_pred_col"]
        print(f"🤖 Using '{pred_col}' column for NN predictions")

        model_subset = nn_preds[["identifier", pred_col]].rename(columns={pred_col: "model_prediction"})
        combined = combined.merge(model_subset, on="identifier", how="left")
        combined["model_prediction"] = combined["model_prediction"].fillna("no hit")

        # Align NN prediction labels to ground-truth label space (same as script 1)
        model_labels = set(combined["model_prediction"].unique())
        labels_not_in_ground_truth_model = model_labels - valid_labels
        labels_not_in_ground_truth_model = labels_not_in_ground_truth_model - {"no hit"}

        if labels_not_in_ground_truth_model:
            print(
                f"⚠️  Found {len(labels_not_in_ground_truth_model)} model labels not in ground truth. Mapping to 'other'...")
            print(f"   Labels: {labels_not_in_ground_truth_model}")
            repl_map_model = {lbl: "other" for lbl in labels_not_in_ground_truth_model}
            combined["model_prediction"] = combined["model_prediction"].replace(repl_map_model)

    # 5) Shared class list (ground-truth only)
    shared_class_list = sorted(list(combined["ground_truth"].unique()))
    if "no hit" in set(combined["hbi_prediction"].unique()) or (
            "model_prediction" in combined and "no hit" in set(combined["model_prediction"].unique())):
        shared_class_list = shared_class_list + ["no hit"]
    print(f"\n🏷️  Shared class list created: {len(shared_class_list)} classes")
    print("   (All predictions aligned to ground truth label space)")
    print(shared_class_list)

    # 6) HBI metrics (computed fresh, script 1)
    print("\n📊 Calculating HBI Metrics...")
    hbi_m = calculate_metrics_bundle(combined, "hbi_prediction", shared_class_list=shared_class_list)
    summary_metrics.append({
        "Method": "HBI (Sequence Similarity)",
        "Accuracy": hbi_m["acc"],
        "MCC": hbi_m["mcc"],
        "Micro_MCC": hbi_m["micro_mcc"],
        "Std_Error": hbi_m["std_error"],
        "Sample_Size": hbi_m["n_samples"]
    })
    print(f"   -> Accuracy: {hbi_m['acc']:.4f} (±{hbi_m['std_error']:.4f}) | MCC: {hbi_m['mcc']:.4f} | Micro-MCC: {hbi_m['micro_mcc']:.4f}")

    # 7) NN metrics (LOADED, script 2)
    nn_json = None
    if has_model:
        nn_json = nn_bundle["nn_json"]
        nn_acc = float(nn_bundle["nn_acc"])
        nn_mcc_proxy = float(nn_bundle["nn_mcc_proxy"])

        # You can compute std error from loaded accuracy and current sample size for consistency in the summary
        n_samples = len(combined)
        nn_std_error = np.sqrt((nn_acc * (1 - nn_acc)) / n_samples) if n_samples > 0 else float("nan")

        summary_metrics.append({
            "Method": "Neural Network (Calibrated, precomputed)",
            "Accuracy": nn_acc,
            "MCC": nn_mcc_proxy,          # NOTE: macro-F1 proxy, script-2 style
            "Micro_MCC": np.nan,          # not provided in precomputed metrics
            "Std_Error": nn_std_error,
            "Sample_Size": n_samples
        })
        print(f"📊 NN (precomputed) -> Accuracy: {nn_acc:.4f} (±{nn_std_error:.4f}) | 'MCC' (macro-F1 proxy): {nn_mcc_proxy:.4f}")

    # 8) Save outputs
    combined.to_csv(RESULTS_DIR / "test_comparison_results.csv", index=False)

    metrics_df = pd.DataFrame(summary_metrics)
    metrics_path = RESULTS_DIR / "metric_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Full JSON report (HBI computed + NN loaded)
    report_data = {
        "HBI": {
            "numeric_metrics": {
                "Test_Accuracy": hbi_m["acc"],
                "Test_MCC": hbi_m["mcc"],
                "Test_Micro_MCC": hbi_m["micro_mcc"],
                "Test_Std_Error": hbi_m["std_error"]
            },
            "classification_report": hbi_m["report"]
        },
        "Neural_Network": nn_json if has_model else "Missing precomputed files"
    }
    with open(RESULTS_DIR / "full_classification_report.json", "w") as f:
        json.dump(report_data, f, indent=4)

    # 9) Visualizations
    plot_confusion_matrix(
        all_labels=hbi_m["y_true_encoded"],
        all_preds=hbi_m["y_pred_encoded"],
        label_encoder=type("obj", (object,), {"classes_": hbi_m["class_list"]})(),
        output_path=str(RESULTS_DIR / "hbi_confusion_matrix.png")
    )

    # Optional: generate NN confusion matrix using predictions (NOT recomputing metrics)
    if has_model:
        # Encode NN preds using the same shared class list
        cls2idx = {cls_name: i for i, cls_name in enumerate(shared_class_list)}
        nn_y_true = combined["ground_truth"].map(cls2idx).to_numpy()
        nn_y_pred = combined["model_prediction"].map(cls2idx).to_numpy()

        # Guard: if any unmapped slipped through
        if np.isnan(nn_y_true).sum() == 0 and np.isnan(nn_y_pred).sum() == 0:
            plot_confusion_matrix(
                all_labels=nn_y_true,
                all_preds=nn_y_pred,
                label_encoder=type("obj", (object,), {"classes_": shared_class_list})(),
                output_path=str(RESULTS_DIR / "model_confusion_matrix.png")
            )
        else:
            print("⚠️ Skipping NN confusion matrix due to unmapped labels after alignment.")

    print(f"\n✅ Done! Summary saved to: {metrics_path}")
    print(f"✅ Detailed Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()