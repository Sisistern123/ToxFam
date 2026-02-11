#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference and evaluation script for preprocessed metazoan toxins.
Uses existing FASTA, TSV, and H5 embedding files.

Pipeline:
1. Load preprocessed data (TSV + H5 embeddings)
2. Run neural network inference using embeddings
3. Run HBI evaluation against training data
4. Compare performance metrics
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import h5py
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report
from sklearn.preprocessing import label_binarize
from pymmseqs.commands import createdb, search

# ---------- Constants ----------
BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "evaluation" / "unreviewed_metazoan"
RESULTS_DIR = EVAL_DIR / "results"

# Input data paths - ADJUST THESE TO YOUR ACTUAL FILES
DATA_DIR = BASE_DIR / "data" / "unreviewed_metazoan"  # Adjust this
INPUT_TSV = DATA_DIR / "unreviewed_metazoan.tsv"  # Your TSV file
INPUT_FASTA = DATA_DIR / "unreviewed_metazoan.fasta"  # Your FASTA file
INPUT_H5 = DATA_DIR / "unreviewed_metazoan_embeds.h5"  # Your H5 embeddings

# Training data reference
TRAIN_DATA = BASE_DIR / "benchmark" / "HBI" / "train_all_df.csv"
TRAIN_FASTA = BASE_DIR / "benchmark" / "HBI" / "train_all_members.fasta"

# Model paths (for neural network inference)
MODEL_DIR = BASE_DIR / "model"
MODEL_CHECKPOINT = MODEL_DIR / "best_model.pth"  # Adjust path as needed


# ---------- Protein Family Normalization ----------
def normalize_protein_families(df: pd.DataFrame, column: str = "Protein families") -> pd.DataFrame:
    """
    Normalize protein family names using standardized regex mapping.
    Everything that doesn't match a known pattern goes to 'other'.

    Args:
        df: DataFrame with protein families column
        column: Name of the column containing protein families

    Returns:
        DataFrame with normalized protein families
    """
    df = df.copy()

    # First, handle basic normalization (take first entry from semicolon/comma separated)
    df[column] = df[column].str.split(";").str[0]
    df[column] = df[column].str.split(",").str[0]

    # Specific replacements for conotoxin superfamilies
    conotoxin_repl = {
        "I1 superfamily": "Conotoxin I1 superfamily",
        "O1 superfamily": "Conotoxin O1 superfamily",
        "O2 superfamily": "Conotoxin O2 superfamily",
        "E superfamily": "Conotoxin E superfamily",
        "F superfamily": "Conotoxin F superfamily",
    }
    df[column] = df[column].replace(conotoxin_repl)

    # Standardized regex-based mapping
    mapping = {
        r"Conotoxin.*": "Conotoxin family",
        r"Neurotoxin.*": "Neurotoxin family",
        r"Scoloptoxin.*|Scolopendra.*": "Scoloptoxin family",
        r"Caterpillar.*": "Caterpillar family",
        r"Teretoxin.*": "Teretoxin family",
        r"Limacoditoxin.*": "Limacoditoxin family",
        r"Scutigerotoxin.*": "Scutigerotoxin family",
        r"Cationic peptide.*": "Cationic peptide family",
        r"Formicidae venom.*": "Formicidae venom family",
        r"Bradykinin-potentiating peptide family|Natriuretic peptide family|Natriuretic":
            "Natriuretic, Bradykinin potentiating peptide family",
        r".*phospholipase.*|.*Phospholipase.*": "Phospholipase family",
    }

    for pattern, replacement in mapping.items():
        df[column] = df[column].str.replace(pattern, replacement, regex=True)

    # Get known families from the mapping
    known_families = set(mapping.values())

    # Count family occurrences after mapping
    family_counts = df[column].value_counts()

    # Map families with <10 occurrences to "other" (except already known standard families)
    def should_keep_family(family):
        if family in known_families:
            return True
        if family_counts.get(family, 0) >= 10:
            return True
        return False

    df[column] = df[column].apply(lambda x: x if should_keep_family(x) else "other")

    return df


def ensure_dirs() -> None:
    """Create necessary directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Step 1: Load Preprocessed Data ----------
def load_preprocessed_data() -> pd.DataFrame:
    """
    Load preprocessed data from TSV file.

    Expected TSV columns:
    - identifier (or Entry, accession, etc.)
    - Sequence
    - Protein families (or Protein_families, etc.)

    Returns:
        DataFrame with normalized protein families
    """
    print("📂 Loading preprocessed data...")

    if not INPUT_TSV.exists():
        raise FileNotFoundError(
            f"TSV file not found: {INPUT_TSV}\n"
            f"Please update INPUT_TSV path in the script."
        )

    # Load TSV
    df = pd.read_csv(INPUT_TSV, sep="\t")
    print(f"   Loaded {len(df)} sequences from {INPUT_TSV}")

    # Normalize column names (handle different naming conventions)
    column_mapping = {
        "Entry": "identifier",
        "entry": "identifier",
        "accession": "identifier",
        "Accession": "identifier",
        "Protein_families": "Protein families",
        "protein_families": "Protein families",
    }
    df = df.rename(columns=column_mapping)

    # Ensure required columns exist
    required_cols = ["identifier", "Sequence", "Protein families"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"⚠️  Available columns: {list(df.columns)}")
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Please ensure TSV has: identifier, Sequence, Protein families"
        )

    # Drop rows without protein families (if any)
    initial_len = len(df)
    df = df.dropna(subset=["Protein families"]).copy()
    if len(df) < initial_len:
        print(f"   Dropped {initial_len - len(df)} entries without protein family annotation")

    # Normalize protein families
    print("🔄 Normalizing protein families...")
    df = normalize_protein_families(df)

    print(f"✅ Loaded {len(df)} sequences")
    print(f"\n📊 Protein family distribution (top 10):")
    print(df["Protein families"].value_counts().head(10))

    return df


def load_embeddings(identifiers: list) -> Dict[str, np.ndarray]:
    """
    Load protein embeddings from H5 file.

    Args:
        identifiers: List of protein identifiers to load

    Returns:
        Dictionary mapping identifier -> embedding array
    """
    print("\n📦 Loading embeddings from H5 file...")

    if not INPUT_H5.exists():
        raise FileNotFoundError(
            f"H5 file not found: {INPUT_H5}\n"
            f"Please update INPUT_H5 path in the script."
        )

    embeddings = {}

    with h5py.File(INPUT_H5, 'r') as f:
        # Check structure of H5 file
        if len(f.keys()) == 0:
            raise ValueError(f"H5 file is empty: {INPUT_H5}")

        # Load embeddings for each identifier
        for identifier in identifiers:
            # Try different possible key formats
            possible_keys = [
                identifier,
                identifier.split('|')[-1],  # Handle UniProt format
                f"protein_{identifier}",
            ]

            found = False
            for key in possible_keys:
                if key in f:
                    embeddings[identifier] = f[key][:]
                    found = True
                    break

            if not found:
                print(f"⚠️  Warning: Embedding not found for {identifier}")

    print(f"✅ Loaded {len(embeddings)} embeddings")

    if len(embeddings) == 0:
        print(f"\n⚠️  Available keys in H5 file (first 5): {list(f.keys())[:5]}")
        raise ValueError("No embeddings loaded. Check identifier format in H5 file.")

    return embeddings


# ---------- Step 2: Neural Network Inference ----------
def run_model_inference(df: pd.DataFrame, embeddings: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Run neural network inference using pre-computed embeddings.

    Args:
        df: DataFrame with sequences and identifiers
        embeddings: Dictionary of identifier -> embedding

    Returns:
        DataFrame with columns: identifier, model_prediction, model_confidence
    """
    print("\n🤖 Running neural network inference...")

    # Check if model exists
    if not MODEL_CHECKPOINT.exists():
        print(f"⚠️  Model not found at {MODEL_CHECKPOINT}")
        print("   Skipping neural network inference.")
        return pd.DataFrame(columns=["identifier", "model_prediction", "model_confidence"])

    try:
        import torch
        import torch.nn.functional as F

        # Load your model - ADJUST THIS TO YOUR MODEL ARCHITECTURE
        # Example for a classifier that takes embeddings directly:
        from model.classifier import ToxinClassifier  # Adjust import

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {device}")

        # Load model
        model = ToxinClassifier.load_from_checkpoint(MODEL_CHECKPOINT)  # Adjust to your loading method
        model = model.to(device)
        model.eval()

        # Prepare embeddings in order
        identifiers_with_embeds = [id for id in df["identifier"] if id in embeddings]
        embed_matrix = np.stack([embeddings[id] for id in identifiers_with_embeds])

        print(f"   Processing {len(identifiers_with_embeds)} sequences with embeddings...")

        # Run inference in batches
        batch_size = 32
        all_predictions = []
        all_confidences = []

        with torch.no_grad():
            for i in range(0, len(embed_matrix), batch_size):
                batch_embeds = embed_matrix[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch_embeds).to(device)

                # Get predictions - ADJUST THIS TO YOUR MODEL'S OUTPUT FORMAT
                logits = model(batch_tensor)  # Adjust method call as needed
                probs = F.softmax(logits, dim=1)

                confidences, pred_indices = torch.max(probs, dim=1)

                all_predictions.extend(pred_indices.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

        # Map indices to labels - ADJUST THIS TO YOUR LABEL ENCODER
        # Example:
        label_encoder = model.label_encoder  # Or however you access it
        predicted_labels = [label_encoder.classes_[idx] for idx in all_predictions]

        # Create results DataFrame
        results = pd.DataFrame({
            "identifier": identifiers_with_embeds,
            "model_prediction": predicted_labels,
            "model_confidence": all_confidences
        })

        # Normalize predictions
        results = results.rename(columns={"model_prediction": "Protein families"})
        results = normalize_protein_families(results)
        results = results.rename(columns={"Protein families": "model_prediction"})

        print(f"✅ Model inference complete: {len(results)} predictions")
        return results

    except Exception as e:
        print(f"⚠️  Error during model inference: {e}")
        print("   Skipping neural network inference.")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=["identifier", "model_prediction", "model_confidence"])


# ---------- Step 3: HBI Evaluation ----------
def write_fasta(df: pd.DataFrame, filename: Path) -> None:
    """Write dataframe to FASTA format."""
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['identifier']}\n{row['Sequence']}\n")


def run_hbi_evaluation(query_df: pd.DataFrame, train_df: pd.DataFrame) -> Dict:
    """
    Run HBI search against training data and calculate metrics.

    Args:
        query_df: Query sequences with ground truth labels
        train_df: Training sequences with labels

    Returns:
        Dictionary with predictions and metrics
    """
    print("\n🔍 Running HBI Evaluation...")

    # Align training labels to query label space
    q_labels = set(query_df["Protein families"].unique())
    t_labels = set(train_df["Protein families"].unique())
    only_in_train = t_labels - q_labels

    if only_in_train:
        print(f"⚠️  Found {len(only_in_train)} train labels not in query set. Mapping to 'other'...")
        print(f"   Labels: {only_in_train}")
        repl_map = {lbl: "other" for lbl in only_in_train}
        train_df = train_df.copy()
        train_df["Protein families"] = train_df["Protein families"].replace(repl_map)

    # Prepare files for MMseqs2 search
    tmp_dir = RESULTS_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    query_fasta = tmp_dir / "query.fasta"
    write_fasta(query_df, query_fasta)

    # Create MMseqs2 databases
    print("   Creating databases...")
    query_db = createdb(str(query_fasta), str(tmp_dir / "queryDB"))
    target_db = createdb(str(TRAIN_FASTA), str(tmp_dir / "targetDB"))

    print("   Running sequence search...")
    search_res = search(
        query_db.to_path(),
        target_db.to_path(),
        str(tmp_dir / "resultDB"),
        str(tmp_dir / "search_tmp"),
        s=9,
        e="inf",
        min_seq_id=0.0,
        max_seqs=100_000
    )

    # Parse results
    df_search = search_res.to_pandas()

    if df_search.empty:
        print("❌ No search hits found.")
        predictions = query_df.copy()
        predictions["hbi_prediction"] = "no hit"
        predictions["hbi_confidence"] = 0.0
        predictions["evalue"] = np.nan
    else:
        # Get best hit per query (minimal E-value)
        best_hits = df_search.loc[df_search.groupby("query")["evalue"].idxmin()].copy()

        # Map to labels
        train_label_map = dict(zip(train_df["identifier"], train_df["Protein families"]))
        best_hits["hbi_prediction"] = best_hits["target"].map(train_label_map)
        best_hits["hbi_confidence"] = best_hits["fident"]

        # Merge with query data
        predictions = query_df.merge(
            best_hits[["query", "hbi_prediction", "hbi_confidence", "evalue"]],
            left_on="identifier",
            right_on="query",
            how="left"
        )
        predictions.drop(columns="query", inplace=True, errors="ignore")

        # Fill no hits
        predictions["hbi_prediction"] = predictions["hbi_prediction"].fillna("no hit")
        predictions["hbi_confidence"] = predictions["hbi_confidence"].fillna(0.0)

    # Align HBI predictions to query label space
    valid_labels = set(query_df["Protein families"].unique())
    hbi_labels = set(predictions["hbi_prediction"].unique())
    labels_not_in_ground_truth = hbi_labels - valid_labels

    if labels_not_in_ground_truth:
        print(f"⚠️  Found {len(labels_not_in_ground_truth)} HBI labels not in ground truth. Mapping to 'other'...")
        print(f"   Labels: {labels_not_in_ground_truth}")
        repl_map_hbi = {lbl: "other" for lbl in labels_not_in_ground_truth}
        predictions["hbi_prediction"] = predictions["hbi_prediction"].replace(repl_map_hbi)

    # Calculate metrics
    metrics = calculate_metrics(
        predictions,
        truth_col="Protein families",
        pred_col="hbi_prediction"
    )

    return {
        "predictions": predictions,
        "metrics": metrics
    }


def calculate_metrics(df: pd.DataFrame, truth_col: str, pred_col: str) -> Dict:
    """Calculate comprehensive metrics."""
    # Create shared class list from ground truth
    class_list = sorted(list(df[truth_col].unique()))

    # Create mapping
    cls2idx = {cls_name: i for i, cls_name in enumerate(class_list)}

    # Encode
    y_true = df[truth_col].map(cls2idx).to_numpy()
    y_pred = df[pred_col].map(cls2idx).to_numpy()

    n_samples = len(y_true)
    n_classes = len(class_list)

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Micro-MCC
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_pred_bin = label_binarize(y_pred, classes=range(n_classes))

    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        y_pred_bin = np.hstack((1 - y_pred_bin, y_pred_bin))

    micro_mcc = matthews_corrcoef(y_true_bin.ravel(), y_pred_bin.ravel())

    # Standard error
    std_error = np.sqrt((acc * (1 - acc)) / n_samples)

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=range(n_classes),
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
        "y_true_encoded": y_true,
        "y_pred_encoded": y_pred
    }


# ---------- Main Pipeline ----------
def main():
    ensure_dirs()

    # Step 1: Load preprocessed data
    df = load_preprocessed_data()

    # Step 2: Load embeddings
    embeddings = load_embeddings(df["identifier"].tolist())

    # Step 3: Run model inference
    model_preds = run_model_inference(df, embeddings)
    has_model = len(model_preds) > 0

    # Step 4: Load and normalize training data
    print("\n🔄 Loading training data...")
    if not TRAIN_DATA.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_DATA}")

    train_df = pd.read_csv(TRAIN_DATA)
    train_df = normalize_protein_families(train_df)
    print(f"✅ Training data: {len(train_df)} sequences")

    # Step 5: Run HBI evaluation
    hbi_results = run_hbi_evaluation(df, train_df)

    # Step 6: Combine results
    combined = hbi_results["predictions"].copy()

    if has_model:
        combined = combined.merge(
            model_preds,
            on="identifier",
            how="left"
        )

        # Align model predictions to ground truth label space
        valid_labels = set(combined["Protein families"].unique())
        model_labels = set(combined["model_prediction"].unique())
        labels_not_in_ground_truth = model_labels - valid_labels

        if labels_not_in_ground_truth:
            print(
                f"\n⚠️  Found {len(labels_not_in_ground_truth)} model labels not in ground truth. Mapping to 'other'...")
            repl_map = {lbl: "other" for lbl in labels_not_in_ground_truth}
            combined["model_prediction"] = combined["model_prediction"].replace(repl_map)

        # Calculate model metrics
        model_metrics = calculate_metrics(
            combined,
            truth_col="Protein families",
            pred_col="model_prediction"
        )

    # Step 7: Save results
    # A. All predictions
    combined.to_csv(RESULTS_DIR / "all_predictions.csv", index=False)
    print(f"\n💾 Saved predictions to: {RESULTS_DIR / 'all_predictions.csv'}")

    # B. HBI metrics
    with open(RESULTS_DIR / "hbi_metrics.json", "w") as f:
        json.dump({
            "numeric_metrics": {
                "Accuracy": hbi_results["metrics"]["acc"],
                "MCC": hbi_results["metrics"]["mcc"],
                "Micro_MCC": hbi_results["metrics"]["micro_mcc"],
                "Std_Error": hbi_results["metrics"]["std_error"],
                "Sample_Size": hbi_results["metrics"]["n_samples"]
            },
            "classification_report": hbi_results["metrics"]["report"]
        }, f, indent=4)

    # C. Model metrics (if available)
    if has_model:
        with open(RESULTS_DIR / "model_metrics.json", "w") as f:
            json.dump({
                "numeric_metrics": {
                    "Accuracy": model_metrics["acc"],
                    "MCC": model_metrics["mcc"],
                    "Micro_MCC": model_metrics["micro_mcc"],
                    "Std_Error": model_metrics["std_error"],
                    "Sample_Size": model_metrics["n_samples"]
                },
                "classification_report": model_metrics["report"]
            }, f, indent=4)

    # D. Summary comparison
    summary_data = [{
        "Method": "HBI (Sequence Similarity)",
        "Accuracy": hbi_results["metrics"]["acc"],
        "MCC": hbi_results["metrics"]["mcc"],
        "Micro_MCC": hbi_results["metrics"]["micro_mcc"],
        "Std_Error": hbi_results["metrics"]["std_error"],
        "Sample_Size": hbi_results["metrics"]["n_samples"]
    }]

    if has_model:
        summary_data.append({
            "Method": "Neural Network",
            "Accuracy": model_metrics["acc"],
            "MCC": model_metrics["mcc"],
            "Micro_MCC": model_metrics["micro_mcc"],
            "Std_Error": model_metrics["std_error"],
            "Sample_Size": model_metrics["n_samples"]
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(RESULTS_DIR / "metric_comparison.csv", index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n📈 HBI Performance:")
    print(f"   Accuracy:  {hbi_results['metrics']['acc']:.4f} (±{hbi_results['metrics']['std_error']:.4f})")
    print(f"   MCC:       {hbi_results['metrics']['mcc']:.4f}")
    print(f"   Micro-MCC: {hbi_results['metrics']['micro_mcc']:.4f}")
    print(f"   Samples:   {hbi_results['metrics']['n_samples']}")

    if has_model:
        print(f"\n🤖 Neural Network Performance:")
        print(f"   Accuracy:  {model_metrics['acc']:.4f} (±{model_metrics['std_error']:.4f})")
        print(f"   MCC:       {model_metrics['mcc']:.4f}")
        print(f"   Micro-MCC: {model_metrics['micro_mcc']:.4f}")
        print(f"   Samples:   {model_metrics['n_samples']}")

    print(f"\n✅ Evaluation complete! Results saved to {RESULTS_DIR}")
    print(f"   - All predictions: all_predictions.csv")
    print(f"   - HBI metrics: hbi_metrics.json")
    if has_model:
        print(f"   - Model metrics: model_metrics.json")
    print(f"   - Comparison: metric_comparison.csv")


if __name__ == "__main__":
    main()
