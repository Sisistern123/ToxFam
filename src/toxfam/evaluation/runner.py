"""Evaluation runner — run one method at a time, compare later.

Each method (HBI, NN model) writes results to a standard directory:
    benchmark/{dataset}/{method}/
        predictions.csv       — identifier, actual_label, predicted_label, confidence,
                                confidence_uncalibrated (NN models only),
                                p_toxic (EAT only)
        metrics.json          — MetricsResult.to_json_dict()
        run_metadata.json     — method, dataset, timestamp, git commit, parameters
        confusion_matrix.png

The ``compare`` function scans all method directories for a dataset and
produces a side-by-side comparison table.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console

from toxfam._paths import benchmark_dir, processed_dir
from toxfam.data._fasta import write_fasta
from toxfam.data.registry import (
    DATASETS,
    load_dataset,
    resolve_embeddings_h5,
)
from toxfam.data.split_manifest import apply_manifest, verify_split_provenance
from toxfam.evaluation.eat import run_eat_search
from toxfam.evaluation.hbi import NO_HIT_LABEL, run_hbi_search
from toxfam.evaluation.metrics import (
    MetricsResult,
    calculate_binary_metrics,
    calculate_metrics,
    print_metrics_table,
)
from toxfam.visualization.plots import plot_confusion_matrix

console = Console()

ORGANISM_COL = "Organism (ID)"

# The dataset registry + loaders live in toxfam.data.registry so `toxfam predict`
# can resolve a dataset name without importing this evaluation/plotting module.
# DATASETS, load_dataset and resolve_embeddings_h5 are used by the runners below.


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


def git_dirty() -> bool:
    """True if the working tree has uncommitted changes.

    A bare short SHA silently hides that results were produced from a modified
    tree; this flags it so provenance is not misleading.
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except Exception:
        return False


def _environment() -> dict:
    """Key Python + package versions for reproducibility (best-effort)."""
    import platform
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    def _v(pkg: str) -> str:
        try:
            return _pkg_version(pkg)
        except PackageNotFoundError:
            return "unknown"

    return {
        "python": platform.python_version(),
        "torch": _v("torch"),
        "transformers": _v("transformers"),
        "scikit-learn": _v("scikit-learn"),
        "numpy": _v("numpy"),
    }


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
        "git_dirty": git_dirty(),
        "environment": _environment(),
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
    write_fasta(df, query_fasta)

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
# EAT evaluation (embedding annotation transfer — the embedding-space analog of HBI)
# ---------------------------------------------------------------------------


def run_eat_evaluation(dataset: str, *, metric: str = "cosine") -> MetricsResult:
    """Run EAT (embedding-based annotation transfer) on a dataset and save results.

    For each query protein, transfer the family label of its nearest ProtT5
    neighbour (k=1) among the *training split* — the same data the MLP trains on,
    and disjoint from val/test (no leakage). ``metric`` is ``"cosine"`` (default,
    selected on val_set) or ``"euclidean"``. Mirrors ``run_hbi_evaluation`` and
    writes the standard ``benchmark/{dataset}/eat/`` outputs, so
    ``toxfam eval compare`` picks it up automatically.
    """
    import h5py

    console.print(f"\n[bold]Running EAT evaluation on '{dataset}'[/]")

    df = load_dataset(dataset)
    proc = processed_dir()
    output_dir = benchmark_dir() / dataset / "eat"

    # Reference = the training split (what the MLP trained on). Its identifiers are
    # disjoint from val/test (the Split column partitions them), so EAT never looks
    # a query up against itself.
    ref_h5 = proc / "embeddings.h5"
    train_df = apply_manifest(pd.read_csv(proc / "training_data.csv"))
    train_df = train_df[train_df["Split"] == "train"].reset_index(drop=True)
    # NB: do NOT collapse train-only family labels to "other" here. run_eat_search
    # derives the toxic/non-toxin mask for p_toxic from these labels, and collapsing
    # "nontox" -> "other" would mark every non-toxin reference as toxic (degenerate
    # p_toxic on datasets whose queries lack the "nontox" label, e.g. non_metazoan).
    # Family-label comparability is instead handled AFTER prediction (unknown
    # predicted labels -> "other", below), which is behaviour-identical for the
    # family metric since relabelling a reference never changes which one is nearest.

    # Query embeddings H5 (evaluation datasets carry their own; the training
    # splits reuse the processed embeddings, i.e. the same file as ref_h5).
    query_h5 = resolve_embeddings_h5(dataset)
    for h5 in (ref_h5, query_h5):
        if not h5.exists():
            raise FileNotFoundError(f"Embeddings not found: {h5}")

    # Filter reference + queries to identifiers actually present in their H5.
    with h5py.File(ref_h5, "r") as f:
        ref_keys = set(f.keys())
    with h5py.File(query_h5, "r") as f:
        query_keys = set(f.keys())
    n_ref_before = len(train_df)
    train_df = train_df[train_df["identifier"].isin(ref_keys)].reset_index(drop=True)
    if len(train_df) < n_ref_before:
        console.print(
            f"   Reference: {len(train_df)}/{n_ref_before} train proteins have embeddings"
        )
    n_q_before = len(df)
    df = df[df["identifier"].isin(query_keys)].reset_index(drop=True)
    if len(df) < n_q_before:
        console.print(f"   Filtered to {len(df)}/{n_q_before} queries with embeddings")

    # 1-NN embedding annotation transfer.
    eat_result = run_eat_search(
        query_h5=query_h5,
        ref_h5=ref_h5,
        reference_df=train_df,
        query_ids=df["identifier"].tolist(),
        metric=metric,
    )
    console.print(
        f"   Reference: {eat_result.n_reference} proteins; queries: {eat_result.n_queries}"
    )

    # Merge predictions with ground truth.
    merged = df[["identifier", "Protein families"]].merge(
        eat_result.predictions, on="identifier", how="left"
    )

    # Map unknown predicted labels to "other" (mirror HBI).
    valid_labels = set(df["Protein families"].unique())
    unknown = set(merged["eat_prediction"].unique()) - valid_labels
    if unknown:
        console.print(f"   Mapping {len(unknown)} unknown EAT labels to 'other'")
        merged["eat_prediction"] = merged["eat_prediction"].replace(
            {lbl: "other" for lbl in unknown}
        )

    # Compute metrics (task-gated, identical to HBI).
    task = _get_task(dataset)
    if task == "binary":
        metrics = calculate_binary_metrics(
            merged["Protein families"], merged["eat_prediction"]
        )
    else:
        metrics = calculate_metrics(
            merged["Protein families"], merged["eat_prediction"]
        )

    # Build standard predictions CSV (+ p_toxic for the binary toxicity comparison).
    predictions_df = pd.DataFrame(
        {
            "identifier": merged["identifier"],
            "actual_label": merged["Protein families"],
            "predicted_label": merged["eat_prediction"],
            "confidence": merged["eat_confidence"],
            "p_toxic": merged["p_toxic"],
        }
    )

    _save_run(
        predictions_df,
        metrics,
        method="eat",
        dataset=dataset,
        params={"k": 1, "metric": metric, "reference": "training_data[train]"},
        output_dir=output_dir,
    )

    print_metrics_table({"EAT": metrics})
    console.print(f"   Results saved to: {output_dir}")
    return metrics


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def _needs_built_taxonomy(model_dir: Path, df: pd.DataFrame) -> bool:
    """True when a combined model needs taxonomy vectors this dataset's H5 lacks.

    Returns False for single-branch models, and for datasets the model's training
    taxonomy H5 already covers (test_set / val_set) — those keep using the stored
    vectors, so their numbers are unaffected.
    """
    import h5py

    from toxfam.model.inference import _resolve_tax_h5
    from toxfam.model.model_config import ModelConfig

    cfg = ModelConfig.load(model_dir / "model_config.json")
    if cfg.architecture != "MultiInputMLP":
        return False

    if ORGANISM_COL not in df.columns:
        console.print(
            f"   [bold red]{model_dir.name} is a combined model but '{ORGANISM_COL}' is "
            "absent from this dataset — its taxonomy branch will contribute nothing.[/]"
        )
        return False

    resolved = _resolve_tax_h5(model_dir)
    if resolved is None:
        return True
    with h5py.File(resolved, "r") as f:
        covered = set(f.keys())
    return bool(set(df["identifier"]) - covered)


def run_model_evaluation(
    dataset: str,
    model_dir: str | Path,
) -> MetricsResult:
    """Run a trained model on a dataset and save results."""
    from toxfam.model.inference import run_inference

    model_dir = Path(model_dir)
    # Every eval scores predictions against ground-truth labels, and for
    # test_set/val_set those labels come from the split. Refuse a checkpoint that
    # was not trained against the manifest on disk.
    verify_split_provenance(model_dir)

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
    h5_path = resolve_embeddings_h5(dataset)
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

    # Run inference. A combined (two-branch) model needs taxonomy vectors for *these*
    # proteins. The training taxonomy H5 only covers the 65,179 training reps, so on any
    # other dataset every protein would silently fall back to a zero vector and we would
    # be scoring a taxonomy-ablated model. Build the vectors from the dataset's own
    # organism IDs, exactly as `toxfam predict` does.
    tax_h5 = None
    tmp_tax_dir = None
    if _needs_built_taxonomy(model_dir, df):
        from toxfam.data.taxonomy import build_taxonomy_h5

        tmp_tax_dir = Path(tempfile.mkdtemp(prefix="toxfam_eval_tax_"))
        console.print("   Building taxonomy vectors from the dataset's organism IDs")
        tax_h5 = build_taxonomy_h5(df, tmp_tax_dir)

    try:
        inference_df = run_inference(df, h5_path, model_dir, tax_h5_path=tax_h5)
    finally:
        if tmp_tax_dir is not None:
            shutil.rmtree(tmp_tax_dir, ignore_errors=True)

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
    pred_cols: dict[str, object] = {
        "identifier": df["identifier"].values,
        "actual_label": df["Protein families"].values,
        "predicted_label": inference_df["predicted_label"].values,
        "confidence": inference_df["confidence"].values,
    }
    if "confidence_uncalibrated" in inference_df.columns:
        pred_cols["confidence_uncalibrated"] = inference_df[
            "confidence_uncalibrated"
        ].values
    predictions_df = pd.DataFrame(pred_cols)

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
# Binary evaluation from a trained model directory
# ---------------------------------------------------------------------------


def run_binary_evaluation_from_dir(model_dir: str | Path) -> dict:
    """Re-compute binary toxic/nontoxin metrics from a trained model directory.

    Loads the calibrated model and its saved ``config.yaml``, computes P(toxic)
    on the val and test splits, optimizes the threshold on val (Youden's J), and
    evaluates on test with both default and optimized thresholds — writing
    ``binary_metrics.json`` + ROC/PR plots into ``model_dir``. Returns the
    binary-metrics dict. This is the library entrypoint behind ``toxfam eval
    binary`` (which is a thin delegator to it).
    """
    from toxfam.config import TrainConfig
    from toxfam.data.dataset import ToxDataset, analyze_data_splits
    from toxfam.evaluation.binary import run_binary_evaluation
    from toxfam.model.inference import load_calibrated_model

    model_dir = Path(model_dir)
    # Scores P(toxic) on the val and test splits, so the checkpoint must be pinned
    # to the split on disk.
    verify_split_provenance(model_dir)

    config = TrainConfig.from_yaml(model_dir / "config.yaml")
    config = config.model_copy(update={"output_dir": model_dir})

    df = apply_manifest(pd.read_csv(config.input_csv))
    train_df, val_df, test_df = analyze_data_splits(df)

    h5_paths = [str(p) for p in config.h5_paths]
    train_ds = ToxDataset(train_df, h5_paths, is_train=True)
    try:
        scaled_model, _, idx_to_label = load_calibrated_model(model_dir)
        # P(toxic) sums the model's non-toxin *output columns*, which are frozen at
        # training time (class_indices.json). Here the LabelEncoder is refit from
        # config.input_csv; if that CSV's train-split labels have drifted since
        # training, the refit order would misalign with the model's neurons and
        # yield silently-wrong metrics. Fail loudly instead.
        frozen = [idx_to_label[i] for i in sorted(idx_to_label)]
        if list(train_ds.le.classes_) != frozen:
            raise ValueError(
                "Class order from config.input_csv does not match the model's "
                "class_indices.json — the training data has drifted since this model "
                "was trained. Re-run eval against the CSV the model was trained on."
            )
        results = run_binary_evaluation(
            scaled_model, train_ds.le, val_df, test_df, config, model_dir,
        )
    finally:
        train_ds.close()

    console.print("Binary metrics saved.")
    return results


# ---------------------------------------------------------------------------
# Compare methods
# ---------------------------------------------------------------------------


def _assert_methods_scored_same_proteins(scored_ids: dict[str, set[str]]) -> None:
    """Refuse to tabulate methods that were run against different protein sets.

    Comparing row counts is not enough: two methods evaluated on two different
    versions of the "test set" both report 9779 samples while sharing a fraction of
    their proteins. Only set equality catches a stale benchmark directory.
    """
    if len(scored_ids) < 2:
        return

    reference_method, reference = next(iter(scored_ids.items()))
    for method, ids in scored_ids.items():
        if ids == reference:
            continue
        shared = len(ids & reference)
        raise ValueError(
            f"'{method}' and '{reference_method}' were evaluated on different protein "
            f"sets, so their metrics are not comparable.\n"
            f"  {method}: {len(ids)} proteins\n"
            f"  {reference_method}: {len(reference)} proteins\n"
            f"  in common: {shared}\n"
            "A benchmark directory is stale. Re-run every method on the current split, "
            "e.g. 'toxfam eval hbi <dataset>' and 'toxfam eval model <dataset> "
            "--model-dir ...', then compare again."
        )


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
    scored_ids: dict[str, set[str]] = {}

    for method_dir in sorted(dataset_dir.iterdir()):
        metrics_path = method_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        method_name = method_dir.name
        if method_name == "comparison":
            continue

        with open(metrics_path) as f:
            data = json.load(f)

        # Skip foreign metrics.json (e.g. the score-based external-tool benchmark,
        # which shares benchmark/{dataset}/ but uses a different schema).
        if "numeric_metrics" not in data:
            console.print(
                f"   [yellow]Skipping '{method_name}': metrics.json has no "
                "'numeric_metrics' (not a toxfam eval method)[/]"
            )
            continue

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

        # Load predictions to get n_samples + the identifiers actually scored
        preds_path = method_dir / "predictions.csv"
        if preds_path.exists():
            preds = pd.read_csv(preds_path)
            m.n_samples = len(preds)
            scored_ids[method_name] = set(preds["identifier"])

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

    _assert_methods_scored_same_proteins(scored_ids)
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
                data = json.load(f)
            if "numeric_metrics" in data:  # skip foreign (external-tool) metrics.json
                full_report[method_dir.name] = data

    with open(comparison_dir / "full_report.json", "w") as f:
        json.dump(full_report, f, indent=4)

    console.print(f"   Comparison saved to: {comparison_dir}")
    return summary_df
