"""Evaluation pipeline for non-metazoan reviewed proteins.

Binary classification: Toxin vs Nontoxin. Compares HBI vs ML model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd
import torch
from pymmseqs.commands import createdb, search
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef

from toxfam._paths import get_project_root

NONTOXIN_LABELS = {"nontox"}


def _write_fasta(df: pd.DataFrame, filename: str | Path) -> None:
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['Entry']}\n{row['Sequence']}\n")


def to_binary_class(label: str) -> str:
    if str(label).lower() in NONTOXIN_LABELS:
        return "nontoxin"
    return "toxin"


def calculate_metrics(df: pd.DataFrame, truth_col: str, pred_col: str) -> Dict:
    y_true = df[truth_col].apply(to_binary_class).to_numpy()
    y_pred = df[pred_col].apply(to_binary_class).to_numpy()

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    n_samples = len(y_true)
    std_error = np.sqrt((acc * (1 - acc)) / n_samples)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["nontoxin", "toxin"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "acc": acc,
        "mcc": mcc,
        "std_error": std_error,
        "n_samples": n_samples,
        "report": report,
    }


def run_hbi_evaluation(
    query_df: pd.DataFrame,
    train_df: pd.DataFrame,
    query_fasta: Path,
    train_fasta: Path,
    results_dir: Path,
) -> pd.DataFrame:
    _write_fasta(query_df, query_fasta)
    tmp_dir = results_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    query_db = createdb(str(query_fasta), str(tmp_dir / "queryDB"))
    target_db = createdb(str(train_fasta), str(tmp_dir / "targetDB"))

    search_res = search(
        query_db.to_path(),
        target_db.to_path(),
        str(tmp_dir / "resultDB"),
        str(tmp_dir / "search_tmp"),
        s=9,
        e="inf",
    )
    df_search = search_res.to_pandas()

    if df_search.empty:
        query_df["hbi_prediction"] = "no hit"
    else:
        best_hits = df_search.loc[df_search.groupby("query")["evalue"].idxmin()].copy()
        train_map = dict(zip(train_df["identifier"], train_df["Protein families"]))
        best_hits["hbi_prediction"] = best_hits["target"].map(train_map)
        query_df = query_df.merge(
            best_hits[["query", "hbi_prediction"]],
            left_on="Entry",
            right_on="query",
            how="left",
        )
        query_df.drop(columns="query", inplace=True)
        query_df["hbi_prediction"] = query_df["hbi_prediction"].fillna("no hit")

    return query_df


def load_calibrated_model(
    model_path: Path, class_map_path: Path, h5_path: Path, device: str = "cpu"
):
    from toxfam.model.architectures import ModularMLP, MultiInputMLP
    from toxfam.model.calibration import ModelWithTemperature

    with open(class_map_path, "r") as f:
        class_indices = json.load(f)
    num_classes = len(class_indices)

    with h5py.File(h5_path, "r") as f:
        first_key = list(f.keys())[0]
        embedding_dim = f[first_key][:].shape[0]

    state_dict = torch.load(model_path, map_location=torch.device(device))

    # Detect model architecture from state dict keys
    is_multi_input = any(k.startswith("model.tax_net.") for k in state_dict)

    if is_multi_input:
        # Infer tax_dim from the first tax_net layer weight shape
        tax_dim = state_dict["model.tax_net.0.weight"].shape[1]
        tax_hidden_dim = state_dict["model.tax_net.0.weight"].shape[0]

        # Infer hidden_dims from joint layer weights
        hidden_dims = []
        i = 0
        while f"model.joint.{i}.weight" in state_dict:
            hidden_dims.append(state_dict[f"model.joint.{i}.weight"].shape[0])
            i += 3  # Linear + ReLU + Dropout
        # Last entry is the output layer, not a hidden dim
        if hidden_dims:
            hidden_dims.pop()

        print(
            f"Reconstructing MultiInputMLP: embedding_dim={embedding_dim}, "
            f"tax_dim={tax_dim}, hidden_dims={hidden_dims}, num_classes={num_classes}"
        )
        base_model = MultiInputMLP(
            embed_dim=embedding_dim,
            tax_dim=tax_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            tax_hidden_dim=tax_hidden_dim,
        )
    else:
        # Infer hidden_dims from projector + backbone weights
        hidden_dims = [state_dict["model.projector.0.weight"].shape[0]]
        i = 0
        while f"model.backbone.{i}.weight" in state_dict:
            hidden_dims.append(state_dict[f"model.backbone.{i}.weight"].shape[0])
            i += 3
        if hidden_dims and len(hidden_dims) > 1:
            hidden_dims.pop()  # last is output layer

        print(
            f"Reconstructing ModularMLP: embedding_dim={embedding_dim}, "
            f"hidden_dims={hidden_dims}, num_classes={num_classes}"
        )
        base_model = ModularMLP(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
        )

    scaled_model = ModelWithTemperature(base_model, torch.device(device))
    scaled_model.load_state_dict(state_dict)

    scaled_model.eval()
    print(
        f"Loaded calibrated model with temperature: "
        f"{scaled_model.temperature.item():.3f}"
    )

    return scaled_model, is_multi_input


def run_model_inference(
    df: pd.DataFrame,
    h5_path: Path,
    model_path: Path,
    class_map_path: Path,
) -> pd.DataFrame:
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, is_multi_input = load_calibrated_model(
        model_path, class_map_path, h5_path, device=device
    )

    with open(class_map_path, "r") as f:
        idx_to_label = {int(k): v for k, v in json.load(f).items()}

    # Infer tax_dim from the model if it's a MultiInputMLP
    if is_multi_input:
        tax_dim = model.model.tax_net[0].in_features
    else:
        tax_dim = None

    preds = []
    with h5py.File(h5_path, "r") as f:
        for ident in df["Entry"]:
            emb = torch.tensor(f[ident][:]).unsqueeze(0).to(device)

            with torch.no_grad():
                if is_multi_input:
                    dummy_tax = torch.zeros(1, tax_dim).to(device)
                    outputs = model(emb, dummy_tax)
                else:
                    outputs = model(emb)
                pred_idx = torch.argmax(outputs, dim=1).item()
            preds.append(idx_to_label.get(pred_idx, "other"))

    df["model_prediction"] = preds
    return df


def run_eval_nonmetazoan(
    h5_path: Path,
    model_path: Path,
    class_map_path: Path,
) -> None:
    """Run non-metazoan evaluation."""
    root = get_project_root()
    base_dir = root / "benchmark" / "new"

    nonmetazoa_tsv = base_dir / "evaluation" / "non-metazoa" / "non-metazoa.tsv"
    nonmetazoa_fasta = (
        base_dir / "evaluation" / "non-metazoa" / "non-metazoa_finalfilter.fasta"
    )
    train_data = root / "benchmark" / "HBI" / "train_all_df.csv"
    train_fasta = root / "benchmark" / "HBI" / "train_all_members.fasta"
    results_dir = base_dir / "evaluation" / "non-metazoa" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df_all = pd.read_csv(nonmetazoa_tsv, sep="\t")

    # 2. Filter by H5 presence
    with h5py.File(h5_path, "r") as f:
        h5_keys = set(f.keys())
    df = df_all[df_all["Entry"].isin(h5_keys)].copy()
    print(f"Synced {len(df)} records with H5 embeddings.")

    df_eval = df.copy()

    # 4. HBI Baseline
    print("Running HBI Evaluation...")
    train_df = pd.read_csv(train_data)
    df_eval = run_hbi_evaluation(
        df_eval, train_df, nonmetazoa_fasta, train_fasta, results_dir / "hbi"
    )

    # 5. Model Inference
    print("Running Model Inference...")
    df_eval = run_model_inference(df_eval, h5_path, model_path, class_map_path)

    # 6. Binary Metrics & Save
    print("Calculating Binary Metrics (Toxin vs Nontoxin)...")

    hbi_m = calculate_metrics(df_eval, "Protein families", "hbi_prediction")
    mod_m = calculate_metrics(df_eval, "Protein families", "model_prediction")

    with open(results_dir / "final_metrics.json", "w") as f:
        json.dump({"HBI": hbi_m, "Model": mod_m}, f, indent=4)

    df_eval["binary_ground_truth"] = df_eval["Protein families"].apply(to_binary_class)
    df_eval["binary_hbi"] = df_eval["hbi_prediction"].apply(to_binary_class)
    df_eval["binary_model"] = df_eval["model_prediction"].apply(to_binary_class)

    df_eval.to_csv(results_dir / "final_results.csv", index=False)

    print("-" * 30)
    print("Success! (Binary Toxin/Nontoxin)")
    print(f"HBI   -> Acc: {hbi_m['acc']:.4f} | MCC: {hbi_m['mcc']:.4f}")
    print(f"Model -> Acc: {mod_m['acc']:.4f} | MCC: {mod_m['mcc']:.4f}")
    print("-" * 30)
