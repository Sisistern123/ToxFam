"""Evaluation runner — run one method at a time, compare later.

Each method (HBI, NN model) writes results to a standard directory:
    benchmark/{dataset}/{method}/
        predictions.csv       — identifier, actual_label, predicted_label, confidence
        metrics.json          — MetricsResult.to_json_dict()
        run_metadata.json     — method, dataset, timestamp, git commit, parameters
        confusion_matrix.png

The ``compare`` function scans all method directories for a dataset and
produces a side-by-side comparison table.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console

from toxfam._paths import benchmark_dir, evaluation_data_dir, processed_dir
from toxfam.data.normalization import normalize_protein_families
from toxfam.evaluation.hbi import NO_HIT_LABEL, run_hbi_search, write_fasta_from_df
from toxfam.evaluation.metrics import (
    MetricsResult,
    calculate_binary_metrics,
    calculate_metrics,
    print_metrics_table,
)
from toxfam.visualization.plots import plot_confusion_matrix

console = Console()

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "test_set": {
        "source": "training_data",
        "split": "test",
        "task": "multiclass",
    },
    "val_set": {
        "source": "training_data",
        "split": "val",
        "task": "multiclass",
    },
    "non_metazoan": {
        "source": "evaluation",
        "tsv": "non_metazoan.tsv",
        "h5": "non_metazoan.h5",
        "task": "binary",
    },
    "unreviewed": {
        "source": "evaluation",
        "tsv": "unreviewed.tsv",
        "h5": "unreviewed.h5",
        "task": "multiclass",
    },
}


def list_datasets() -> list[str]:
    return list(DATASETS.keys())


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(dataset: str) -> pd.DataFrame:
    """Load ground-truth DataFrame for a dataset.

    Returns DataFrame with at least ``identifier``, ``Sequence``,
    ``Protein families`` columns.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list_datasets()}")

    cfg = DATASETS[dataset]

    if cfg["source"] == "training_data":
        training_csv = processed_dir() / "training_data.csv"
        if not training_csv.exists():
            raise FileNotFoundError(
                f"{training_csv} not found. Run 'toxfam download-data' first."
            )
        df = pd.read_csv(training_csv)
        df = df[df["Split"] == cfg["split"]].reset_index(drop=True)
        console.print(f"   Loaded {len(df)} sequences from {cfg['split']} split")
        return df

    # evaluation datasets
    eval_dir = evaluation_data_dir() / dataset
    tsv_name = cfg.get("tsv")
    if tsv_name is None:
        raise ValueError(f"Dataset '{dataset}' requires --input-tsv")

    tsv_path = eval_dir / tsv_name
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"{tsv_path} not found. Run 'toxfam download-data' first."
        )

    df = pd.read_csv(tsv_path, sep="\t")
    # Normalize column names
    if "Entry" in df.columns and "identifier" not in df.columns:
        df = df.rename(columns={"Entry": "identifier"})

    df = df.dropna(subset=["Protein families"]).copy()
    df = normalize_protein_families(df)

    console.print(f"   Loaded {len(df)} sequences from {tsv_path.name}")
    return df


def _get_task(dataset: str) -> str:
    return DATASETS[dataset]["task"]


# ---------------------------------------------------------------------------
# Save run results (standard format)
# ---------------------------------------------------------------------------


def git_commit_short() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _save_run(
    predictions_df: pd.DataFrame,
    metrics: MetricsResult,
    method: str,
    dataset: str,
    params: dict,
    output_dir: Path,
) -> None:
    """Write standard run outputs to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # predictions.csv
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    # metrics.json
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics.to_json_dict(), f, indent=4)

    # run_metadata.json
    metadata = {
        "method": method,
        "dataset": dataset,
        "task": _get_task(dataset),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_short(),
        "n_samples": metrics.n_samples,
        "parameters": params,
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # confusion_matrix.png
    plot_confusion_matrix(
        metrics.y_true_encoded,
        metrics.y_pred_encoded,
        metrics.class_list,
        str(output_dir / "confusion_matrix.png"),
    )


# ---------------------------------------------------------------------------
# HBI evaluation
# ---------------------------------------------------------------------------


def run_hbi_evaluation(dataset: str) -> MetricsResult:
    """Run HBI on a dataset and save results."""
    console.print(f"\n[bold]Running HBI evaluation on '{dataset}'[/]")

    df = load_dataset(dataset)
    proc = processed_dir()
    output_dir = benchmark_dir() / dataset / "hbi"

    # Build query FASTA
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    query_fasta = tmp_dir / "query.fasta"
    write_fasta_from_df(df, query_fasta)

    # Load HBI reference and harmonize labels
    train_df = pd.read_csv(proc / "hbi_train_all.csv")
    query_labels = set(df["Protein families"].unique())
    train_labels = set(train_df["Protein families"].unique())
    only_in_train = train_labels - query_labels
    if only_in_train:
        console.print(f"   Mapping {len(only_in_train)} train-only labels to 'other'")
        train_df["Protein families"] = train_df["Protein families"].replace(
            {lbl: "other" for lbl in only_in_train}
        )

    # Run search
    hbi_result = run_hbi_search(
        query_fasta=query_fasta,
        target_fasta=proc / "hbi_train_all.fasta",
        target_labels_df=train_df,
        work_dir=tmp_dir,
    )
    console.print(
        f"   Coverage: {hbi_result.coverage:.1%} "
        f"({hbi_result.n_with_hits}/{hbi_result.n_queries})"
    )

    # Merge predictions with ground truth
    merged = df[["identifier", "Protein families"]].merge(
        hbi_result.predictions, on="identifier", how="left"
    )
    merged["hbi_prediction"] = merged["hbi_prediction"].fillna(NO_HIT_LABEL)

    # Map unknown HBI labels to "other"
    valid_labels = set(df["Protein families"].unique())
    unknown = set(merged["hbi_prediction"].unique()) - valid_labels - {NO_HIT_LABEL}
    if unknown:
        console.print(f"   Mapping {len(unknown)} unknown HBI labels to 'other'")
        merged["hbi_prediction"] = merged["hbi_prediction"].replace(
            {lbl: "other" for lbl in unknown}
        )

    # Compute metrics
    task = _get_task(dataset)
    if task == "binary":
        metrics = calculate_binary_metrics(
            merged["Protein families"], merged["hbi_prediction"]
        )
    else:
        metrics = calculate_metrics(
            merged["Protein families"], merged["hbi_prediction"]
        )

    # Build standard predictions CSV
    predictions_df = pd.DataFrame(
        {
            "identifier": merged["identifier"],
            "actual_label": merged["Protein families"],
            "predicted_label": merged["hbi_prediction"],
            "confidence": hbi_result.predictions.set_index("identifier")
            .reindex(merged["identifier"])["hbi_confidence"]
            .values,
        }
    )

    _save_run(
        predictions_df,
        metrics,
        method="hbi",
        dataset=dataset,
        params={
            "sensitivity": 9.0,
            "evalue": "inf",
            "max_seqs": 100_000,
        },
        output_dir=output_dir,
    )

    print_metrics_table({"HBI": metrics})
    console.print(f"   Results saved to: {output_dir}")
    return metrics


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def run_model_evaluation(
    dataset: str,
    model_dir: str | Path,
) -> MetricsResult:
    """Run a trained model on a dataset and save results."""
    from toxfam.model.inference import run_inference

    model_dir = Path(model_dir)
    method_name = f"nn_{model_dir.name}"
    console.print(f"\n[bold]Running model evaluation '{method_name}' on '{dataset}'[/]")

    df = load_dataset(dataset)
    output_dir = benchmark_dir() / dataset / method_name

    # Check model dir has required files
    config_path = model_dir / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"model_config.json not found in {model_dir}. "
            "Re-run training or generate it from config.yaml."
        )

    # Find embeddings H5
    cfg = DATASETS[dataset]
    if cfg["source"] == "evaluation":
        h5_path = evaluation_data_dir() / dataset / cfg["h5"]
    else:
        h5_path = processed_dir() / "embeddings.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {h5_path}")

    # Filter to proteins present in H5
    import h5py

    with h5py.File(h5_path, "r") as f:
        h5_keys = set(f.keys())
    n_before = len(df)
    df = df[df["identifier"].isin(h5_keys)].copy()
    if len(df) < n_before:
        console.print(f"   Filtered to {len(df)}/{n_before} sequences with embeddings")

    # Run inference
    inference_df = run_inference(df, h5_path, model_dir)

    # Compute metrics
    task = _get_task(dataset)
    if task == "binary":
        metrics = calculate_binary_metrics(
            df["Protein families"], inference_df["predicted_label"]
        )
    else:
        metrics = calculate_metrics(
            df["Protein families"], inference_df["predicted_label"]
        )

    # Build standard predictions CSV
    predictions_df = pd.DataFrame(
        {
            "identifier": df["identifier"].values,
            "actual_label": df["Protein families"].values,
            "predicted_label": inference_df["predicted_label"].values,
            "confidence": inference_df["confidence"].values,
        }
    )

    _save_run(
        predictions_df,
        metrics,
        method=method_name,
        dataset=dataset,
        params={"model_dir": str(model_dir)},
        output_dir=output_dir,
    )

    print_metrics_table({method_name: metrics})
    console.print(f"   Results saved to: {output_dir}")
    return metrics


# ---------------------------------------------------------------------------
# Compare methods
# ---------------------------------------------------------------------------


def compare_methods(dataset: str) -> pd.DataFrame:
    """Compare all methods that have been run for a dataset."""
    dataset_dir = benchmark_dir() / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"No results for dataset '{dataset}'. Run evaluations first."
        )

    console.print(f"\n[bold]Comparing methods for '{dataset}'[/]")

    results: dict[str, MetricsResult] = {}
    summary_rows: list[dict] = []

    for method_dir in sorted(dataset_dir.iterdir()):
        metrics_path = method_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        method_name = method_dir.name
        if method_name == "comparison":
            continue

        with open(metrics_path) as f:
            data = json.load(f)

        nm = data["numeric_metrics"]
        # Reconstruct a lightweight MetricsResult for the table
        from types import SimpleNamespace

        m = SimpleNamespace(
            accuracy=nm["Test_Accuracy"],
            mcc=nm["Test_MCC"],
            micro_mcc=nm["Test_Micro_MCC"],
            std_error=nm["Test_Std_Error"],
            n_samples=0,
        )

        # Load predictions to get n_samples
        preds_path = method_dir / "predictions.csv"
        if preds_path.exists():
            n = len(pd.read_csv(preds_path))
            m.n_samples = n

        results[method_name] = m
        summary_rows.append(
            {
                "Method": method_name,
                "Accuracy": m.accuracy,
                "MCC": m.mcc,
                "Micro_MCC": m.micro_mcc,
                "Std_Error": m.std_error,
                "Sample_Size": m.n_samples,
            }
        )

    if not results:
        console.print("[yellow]No method results found.[/]")
        return pd.DataFrame()

    print_metrics_table(results)

    # Save comparison
    comparison_dir = dataset_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(comparison_dir / "metric_comparison.csv", index=False)

    # Build full report with all classification reports
    full_report = {}
    for method_dir in sorted(dataset_dir.iterdir()):
        metrics_path = method_dir / "metrics.json"
        if metrics_path.exists() and method_dir.name != "comparison":
            with open(metrics_path) as f:
                full_report[method_dir.name] = json.load(f)

    with open(comparison_dir / "full_report.json", "w") as f:
        json.dump(full_report, f, indent=4)

    console.print(f"   Comparison saved to: {comparison_dir}")
    return summary_df
