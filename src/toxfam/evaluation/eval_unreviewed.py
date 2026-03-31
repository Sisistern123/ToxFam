"""Evaluation script for unreviewed metazoan proteins."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rich.console import Console

from toxfam._paths import benchmark_dir, processed_dir
from toxfam.data.preprocessing import normalize_protein_families
from toxfam.evaluation.hbi import NO_HIT_LABEL, run_hbi_search, write_fasta_from_df
from toxfam.evaluation.metrics import calculate_metrics, print_metrics_table

console = Console()


# ---------- Data Loading ----------


def load_preprocessed_data(input_tsv: Path) -> pd.DataFrame:
    console.print("Loading preprocessed data...")

    if not input_tsv.exists():
        raise FileNotFoundError(f"TSV file not found: {input_tsv}")

    df = pd.read_csv(input_tsv, sep="\t")
    console.print(f"   Loaded {len(df)} sequences from {input_tsv}")

    column_mapping = {
        "Entry": "identifier",
        "entry": "identifier",
        "accession": "identifier",
        "Accession": "identifier",
        "Protein_families": "Protein families",
        "protein_families": "Protein families",
    }
    df = df.rename(columns=column_mapping)

    required_cols = ["identifier", "Sequence", "Protein families"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        console.print(f"   Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    initial_len = len(df)
    df = df.dropna(subset=["Protein families"]).copy()
    if len(df) < initial_len:
        console.print(
            f"   Dropped {initial_len - len(df)} entries without "
            f"protein family annotation"
        )

    console.print("Normalizing protein families...")
    df = normalize_protein_families(df)

    console.print(f"Loaded {len(df)} sequences")
    console.print("\nProtein family distribution (top 10):")
    console.print(str(df["Protein families"].value_counts().head(10)))

    return df


# ---------- Main Pipeline ----------


def run_eval_unreviewed(
    input_tsv: Path,
    input_h5: Path,
    train_data: Path | None = None,
    train_fasta: Path | None = None,
) -> None:
    """Run unreviewed metazoan evaluation pipeline."""
    _ = input_h5  # Reserved for future model inference

    proc = processed_dir()
    if train_data is None:
        train_data = proc / "hbi_train_all.csv"
    if train_fasta is None:
        train_fasta = proc / "hbi_train_all.fasta"

    results_dir = benchmark_dir() / "unreviewed"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load preprocessed data
    df = load_preprocessed_data(input_tsv)

    # 2. Load training data
    console.print("\nLoading training data...")
    if not train_data.exists():
        raise FileNotFoundError(f"Training data not found: {train_data}")

    train_df = pd.read_csv(train_data)
    train_df = normalize_protein_families(train_df)
    console.print(f"Training data: {len(train_df)} sequences")

    # 3. Harmonize train labels to query label space
    q_labels = set(df["Protein families"].unique())
    t_labels = set(train_df["Protein families"].unique())
    only_in_train = t_labels - q_labels
    if only_in_train:
        console.print(
            f"   Found {len(only_in_train)} train labels not in query set. "
            f"Mapping to 'other'..."
        )
        train_df = train_df.copy()
        train_df["Protein families"] = train_df["Protein families"].replace(
            {lbl: "other" for lbl in only_in_train}
        )

    # 4. Write query FASTA and run HBI
    console.print("\nRunning HBI Evaluation...")
    query_fasta = results_dir / "tmp" / "query.fasta"
    write_fasta_from_df(df, query_fasta)

    hbi_result = run_hbi_search(
        query_fasta=query_fasta,
        target_fasta=train_fasta,
        target_labels_df=train_df,
        work_dir=results_dir / "tmp",
    )
    console.print(
        f"   HBI Coverage: {hbi_result.coverage:.1%} "
        f"({hbi_result.n_with_hits}/{hbi_result.n_queries})"
    )

    # 5. Merge predictions and handle label mismatches
    predictions = df.merge(hbi_result.predictions, on="identifier", how="left")
    predictions["hbi_prediction"] = predictions["hbi_prediction"].fillna(NO_HIT_LABEL)

    valid_labels = set(df["Protein families"].unique())
    hbi_labels = set(predictions["hbi_prediction"].unique())
    unknown = hbi_labels - valid_labels - {NO_HIT_LABEL}
    if unknown:
        console.print(
            f"   Mapping {len(unknown)} HBI labels not in ground truth to 'other'"
        )
        predictions["hbi_prediction"] = predictions["hbi_prediction"].replace(
            {lbl: "other" for lbl in unknown}
        )

    # 6. Calculate metrics
    hbi_metrics = calculate_metrics(
        predictions["Protein families"],
        predictions["hbi_prediction"],
    )

    # 7. Save results
    predictions.to_csv(results_dir / "all_predictions.csv", index=False)

    with open(results_dir / "hbi_metrics.json", "w") as f:
        json.dump(hbi_metrics.to_json_dict(), f, indent=4)

    summary_df = pd.DataFrame(
        [hbi_metrics.to_summary_dict("HBI (Sequence Similarity)")]
    )
    summary_df.to_csv(results_dir / "metric_comparison.csv", index=False)

    # 8. Print summary
    print_metrics_table({"HBI": hbi_metrics})
    console.print(f"\nEvaluation complete! Results saved to {results_dir}")
