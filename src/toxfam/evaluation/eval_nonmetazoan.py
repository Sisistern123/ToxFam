"""Evaluation pipeline for non-metazoan reviewed proteins.

Binary classification: Toxin vs Nontoxin. Compares HBI vs ML model.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pandas as pd
import torch
from rich.console import Console

from toxfam._paths import benchmark_dir, evaluation_data_dir, processed_dir
from toxfam.evaluation.hbi import NO_HIT_LABEL, run_hbi_search, write_fasta_from_df
from toxfam.evaluation.metrics import (
    calculate_binary_metrics,
    print_metrics_table,
    to_binary_class,
)

console = Console()


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

    is_multi_input = any(k.startswith("model.tax_net.") for k in state_dict)

    if is_multi_input:
        tax_dim = state_dict["model.tax_net.0.weight"].shape[1]
        tax_hidden_dim = state_dict["model.tax_net.0.weight"].shape[0]

        hidden_dims = []
        i = 0
        while f"model.joint.{i}.weight" in state_dict:
            hidden_dims.append(state_dict[f"model.joint.{i}.weight"].shape[0])
            i += 3
        if hidden_dims:
            hidden_dims.pop()

        console.print(
            f"   Reconstructing MultiInputMLP: embedding_dim={embedding_dim}, "
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
        hidden_dims = [state_dict["model.projector.0.weight"].shape[0]]
        i = 0
        while f"model.backbone.{i}.weight" in state_dict:
            hidden_dims.append(state_dict[f"model.backbone.{i}.weight"].shape[0])
            i += 3
        if hidden_dims and len(hidden_dims) > 1:
            hidden_dims.pop()

        console.print(
            f"   Reconstructing ModularMLP: embedding_dim={embedding_dim}, "
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

    console.print(
        f"   Loaded calibrated model with temperature: "
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

    if is_multi_input:
        tax_dim = model.model.tax_net[0].in_features
    else:
        tax_dim = None

    preds = []
    with h5py.File(h5_path, "r") as f:
        for ident in df["identifier"]:
            emb = torch.tensor(f[ident][:]).unsqueeze(0).to(device)

            with torch.no_grad():
                if is_multi_input:
                    dummy_tax = torch.zeros(1, tax_dim).to(device)
                    outputs = model(emb, dummy_tax)
                else:
                    outputs = model(emb)
                pred_idx = torch.argmax(outputs, dim=1).item()
            preds.append(idx_to_label.get(pred_idx, "other"))

    df = df.copy()
    df["model_prediction"] = preds
    return df


def run_eval_nonmetazoan(
    h5_path: Path,
    model_path: Path,
    class_map_path: Path,
) -> None:
    """Run non-metazoan evaluation."""
    eval_data = evaluation_data_dir() / "non_metazoan"
    proc = processed_dir()

    nonmetazoa_tsv = eval_data / "non_metazoan.tsv"
    train_data = proc / "hbi_train_all.csv"
    train_fasta = proc / "hbi_train_all.fasta"
    results_dir = benchmark_dir() / "non_metazoan"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and normalize column names
    df_all = pd.read_csv(nonmetazoa_tsv, sep="\t")
    if "Entry" in df_all.columns and "identifier" not in df_all.columns:
        df_all = df_all.rename(columns={"Entry": "identifier"})

    # 2. Filter by H5 presence
    with h5py.File(h5_path, "r") as f:
        h5_keys = set(f.keys())
    df = df_all[df_all["identifier"].isin(h5_keys)].copy()
    console.print(f"Synced {len(df)} records with H5 embeddings.")

    # 3. HBI search
    console.print("Running HBI Evaluation...")
    train_df = pd.read_csv(train_data)

    query_fasta = results_dir / "hbi" / "query.fasta"
    write_fasta_from_df(df, query_fasta)

    hbi_result = run_hbi_search(
        query_fasta=query_fasta,
        target_fasta=train_fasta,
        target_labels_df=train_df,
        work_dir=results_dir / "hbi" / "tmp",
    )
    console.print(
        f"   HBI Coverage: {hbi_result.coverage:.1%} "
        f"({hbi_result.n_with_hits}/{hbi_result.n_queries})"
    )

    df_eval = df.merge(hbi_result.predictions, on="identifier", how="left")
    df_eval["hbi_prediction"] = df_eval["hbi_prediction"].fillna(NO_HIT_LABEL)

    # 4. Model inference
    console.print("Running Model Inference...")
    df_eval = run_model_inference(df_eval, h5_path, model_path, class_map_path)

    # 5. Binary metrics
    console.print("Calculating Binary Metrics (Toxin vs Nontoxin)...")
    hbi_m = calculate_binary_metrics(
        df_eval["Protein families"], df_eval["hbi_prediction"]
    )
    mod_m = calculate_binary_metrics(
        df_eval["Protein families"], df_eval["model_prediction"]
    )

    # 6. Save results
    with open(results_dir / "final_metrics.json", "w") as f:
        json.dump(
            {"HBI": hbi_m.to_json_dict(), "Model": mod_m.to_json_dict()}, f, indent=4
        )

    df_eval["binary_ground_truth"] = df_eval["Protein families"].apply(to_binary_class)
    df_eval["binary_hbi"] = df_eval["hbi_prediction"].apply(to_binary_class)
    df_eval["binary_model"] = df_eval["model_prediction"].apply(to_binary_class)
    df_eval.to_csv(results_dir / "final_results.csv", index=False)

    # 7. Print summary
    print_metrics_table({"HBI": hbi_m, "Model": mod_m})
