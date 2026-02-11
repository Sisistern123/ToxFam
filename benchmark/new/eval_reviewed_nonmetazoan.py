#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Evaluation Pipeline: HBI vs. ML Model (Post-SignalP6)
Matches H5 embeddings to UniProt metadata, NO CLUSTERING, Binary Classification (Toxin/Nontoxin).
"""

from __future__ import annotations
import os
import json
import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report
from pymmseqs.commands import createdb, search

# ---------- Configuration ----------
BASE_DIR = Path(__file__).resolve().parent

# --- PATHS ---
H5_PATH = Path("/Users/selin/PycharmProjects/ToxFam/benchmark/new/evaluation/non-metazoa/non-metazoa.h5")
MODEL_PATH = Path("/Users/selin/PycharmProjects/ToxFam/model/model_output/calibrated_combined/best_model_calibrated.pt")
CLASS_MAP_PATH = Path("/Users/selin/PycharmProjects/ToxFam/model/model_output/calibrated_combined/class_indices.json")
nonmetazoa = BASE_DIR / "evaluation" / "non-metazoa" / "non-metazoa.tsv"
nonmetazoa_fasta = BASE_DIR / "evaluation" / "non-metazoa" / "non-metazoa_finalfilter.fasta"

# ---------------------------

# HBI Reference Data (must exist)
TRAIN_DATA = BASE_DIR.parent / "HBI" / "train_all_df.csv"
TRAIN_FASTA = BASE_DIR.parent / "HBI" / "train_all_members.fasta"

# Output Locations
RESULTS_DIR = BASE_DIR / "evaluation" / "non-metazoa" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Define what counts as "Nontoxin" (Case insensitive check used later)
NONTOXIN_LABELS = {"nontox"}


# ---------- Helper Functions ----------

def write_fasta(df: pd.DataFrame, filename: str | Path) -> None:
    """Write dataframe to FASTA format."""
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['Entry']}\n{row['Sequence']}\n")


def to_binary_class(label: str) -> str:
    """Map specific protein families to binary Toxin/Nontoxin classes."""
    # check if label is in the nontoxin set (normalized to lowercase)
    if str(label).lower() in NONTOXIN_LABELS:
        return "nontoxin"
    return "toxin"


# ---------- Logic Core ----------

def calculate_metrics(df: pd.DataFrame, truth_col: str, pred_col: str) -> Dict:
    """
    Calculate metrics based on binary classification (Toxin vs Nontoxin).
    Maps raw family labels to binary before scoring.
    """
    # 1. Map columns to binary 'toxin' / 'nontoxin'
    y_true = df[truth_col].apply(to_binary_class).to_numpy()
    y_pred = df[pred_col].apply(to_binary_class).to_numpy()

    # 2. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Standard Error
    n_samples = len(y_true)
    std_error = np.sqrt((acc * (1 - acc)) / n_samples)

    # Full report
    report = classification_report(
        y_true,
        y_pred,
        target_names=["nontoxin", "toxin"],  # Alphabetical order usually, but report handles strings
        output_dict=True,
        zero_division=0
    )

    return {
        "acc": acc,
        "mcc": mcc,
        "std_error": std_error,
        "n_samples": n_samples,
        "report": report
    }


def run_hbi_evaluation(query_df: pd.DataFrame, train_df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Run sequence-similarity search (HBI) against training data."""
    query_fasta = nonmetazoa_fasta
    write_fasta(query_df, query_fasta)
    tmp_dir = results_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    query_db = createdb(str(query_fasta), str(tmp_dir / "queryDB"))
    target_db = createdb(str(TRAIN_FASTA), str(tmp_dir / "targetDB"))

    # Removed 's' and 'e' params to use defaults or adjust as needed, kept your previous setting
    search_res = search(query_db.to_path(), target_db.to_path(), str(tmp_dir / "resultDB"), str(tmp_dir / "search_tmp"),
                        s=9, e="inf")
    df_search = search_res.to_pandas()

    if df_search.empty:
        query_df["hbi_prediction"] = "no hit"
    else:
        best_hits = df_search.loc[df_search.groupby("query")["evalue"].idxmin()].copy()
        train_map = dict(zip(train_df["identifier"], train_df["Protein families"]))
        best_hits["hbi_prediction"] = best_hits["target"].map(train_map)
        query_df = query_df.merge(best_hits[["query", "hbi_prediction"]], left_on="Entry", right_on="query",
                                  how="left")
        query_df.drop(columns="query", inplace=True)
        query_df["hbi_prediction"] = query_df["hbi_prediction"].fillna("no hit")

    # We do NOT map to 'other' here anymore, because we want to see the specific family
    # to decide if it is toxin or nontoxin in the metric step.
    return query_df


def load_calibrated_model(model_path: Path, class_map_path: Path, h5_path: Path, device: str = 'cpu'):
    """
    Load a calibrated model saved from ModelWithTemperature state dict.

    The state dict contains:
    - model.* : the underlying base model weights
    - temperature: the temperature parameter

    We need to:
    1. Determine the model architecture (inspect H5 to get embedding dim)
    2. Load class mapping to get num_classes
    3. Instantiate the base model
    4. Wrap it in ModelWithTemperature
    5. Load the state dict

    Returns the loaded model ready for inference.
    """
    # Import here to avoid circular dependencies
    from model.model_architecture import MultiInputMLP
    from model.calibration import ModelWithTemperature

    # Load class mapping to get num_classes
    with open(class_map_path, 'r') as f:
        class_indices = json.load(f)
    num_classes = len(class_indices)

    # Get embedding dimension from H5 file (check first entry)
    with h5py.File(h5_path, 'r') as f:
        first_key = list(f.keys())[0]
        embedding_dim = f[first_key][:].shape[0]

    print(f"Reconstructing model: embedding_dim={embedding_dim}, num_classes={num_classes}")

    # Reconstruct the base model architecture
    # Based on your training code, the calibrated model is MultiInputMLP for combined strategy
    # You'll need to adjust these parameters to match your config
    base_model = MultiInputMLP(
        embed_dim=embedding_dim,  # e.g., 1024
        tax_dim=7,  # This should match your CONFIG["tax_dim"]
        hidden_dims=[512, 256],  # This should match your CONFIG["hidden_dims"]
        num_classes=num_classes,
        dropout=0.3  # This should match your CONFIG["dropout"]
    )

    # Wrap in ModelWithTemperature
    scaled_model = ModelWithTemperature(base_model, torch.device(device))

    # Load the state dict
    state_dict = torch.load(model_path, map_location=torch.device(device))
    scaled_model.load_state_dict(state_dict)

    scaled_model.eval()
    print(f"✅ Loaded calibrated model with temperature: {scaled_model.temperature.item():.3f}")

    return scaled_model


def run_model_inference(df: pd.DataFrame, h5_path: Path, model_path: Path) -> pd.DataFrame:
    """Run ML Model inference using pre-computed H5 embeddings."""

    # Load the calibrated model
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_calibrated_model(model_path, CLASS_MAP_PATH, h5_path, device=device)

    # Load class mapping
    with open(CLASS_MAP_PATH, 'r') as f:
        idx_to_label = {int(k): v for k, v in json.load(f).items()}

    preds = []
    with h5py.File(h5_path, 'r') as f:
        for ident in df['Entry']:
            emb = torch.tensor(f[ident][:]).unsqueeze(0).to(device)

            # IMPORTANT: MultiInputMLP expects TWO inputs: (embeddings, taxonomy)
            # Since we don't have taxonomy data during inference, we create a dummy zero vector
            # This is a limitation - ideally you'd have the taxonomy H5 file for inference too
            tax_dim = 7  # Should match your CONFIG["tax_dim"]
            dummy_tax = torch.zeros(1, tax_dim).to(device)

            with torch.no_grad():
                # The calibrated model's forward pass goes through ModelWithTemperature
                # which calls the underlying MultiInputMLP with both inputs
                outputs = model(emb, dummy_tax)
                pred_idx = torch.argmax(outputs, dim=1).item()
            # Append the raw family name (e.g., "Snake Toxin A", "nontox", etc.)
            preds.append(idx_to_label.get(pred_idx, "other"))

    df['model_prediction'] = preds
    return df


# ---------- Main Execution ----------

def main():
    # 1. Load Data
    df_all = pd.read_csv(nonmetazoa, sep="\t")

    # 2. Filter by H5 presence (The post-SignalP6 filter)
    with h5py.File(H5_PATH, 'r') as f: h5_keys = set(f.keys())
    df = df_all[df_all['Entry'].isin(h5_keys)].copy()
    print(f"✅ Synced {len(df)} records with H5 embeddings.")

    df_eval = df.copy()

    # 4. HBI Baseline
    print("🔍 Running HBI Evaluation...")
    train_df = pd.read_csv(TRAIN_DATA)
    df_eval = run_hbi_evaluation(df_eval, train_df, RESULTS_DIR / "hbi")

    # 5. Model Inference
    print("🤖 Running Model Inference...")
    df_eval = run_model_inference(df_eval, H5_PATH, MODEL_PATH)

    # 6. Binary Metrics & Save
    print("📊 Calculating Binary Metrics (Toxin vs Nontoxin)...")

    # Note: We pass the raw columns. The function maps them to binary internally.
    hbi_m = calculate_metrics(df_eval, "Protein families", "hbi_prediction")
    mod_m = calculate_metrics(df_eval, "Protein families", "model_prediction")

    # Save Metrics
    with open(RESULTS_DIR / "final_metrics.json", "w") as f:
        json.dump({"HBI": hbi_m, "Model": mod_m}, f, indent=4)

    # Save Predictions (Add binary columns to CSV for manual check)
    df_eval["binary_ground_truth"] = df_eval["Protein families"].apply(to_binary_class)
    df_eval["binary_hbi"] = df_eval["hbi_prediction"].apply(to_binary_class)
    df_eval["binary_model"] = df_eval["model_prediction"].apply(to_binary_class)

    df_eval.to_csv(RESULTS_DIR / "final_results.csv", index=False)

    print("-" * 30)
    print(f"✅ Success! (Binary Toxin/Nontoxin)")
    print(f"HBI   -> Acc: {hbi_m['acc']:.4f} | MCC: {hbi_m['mcc']:.4f}")
    print(f"Model -> Acc: {mod_m['acc']:.4f} | MCC: {mod_m['mcc']:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    main()