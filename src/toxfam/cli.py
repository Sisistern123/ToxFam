"""Unified CLI for ToxFam."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Optional,
)  # Required by Typer's runtime type resolution (cannot use X | None syntax)

import typer
from rich.console import Console

from toxfam import __version__

# Status/narration to stdout; errors to stderr (where scripts expect them).
console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="toxfam",
    help="Animal toxin protein family classification using MLP on ProtT5 embeddings.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    # Suppress Typer's local-variable dump on uncaught exceptions — command bodies
    # hold live tensors / DataFrames / models that would flood the terminal; the
    # final traceback line (the actionable one) is still shown.
    pretty_exceptions_show_locals=False,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"toxfam {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the toxfam version and exit.",
        ),
    ] = None,
) -> None:
    # No docstring / help: keep the Typer(help=...) above as the top-level description.
    pass


# Shared embedding-batch option types (reused by `embed` and `predict`). Typer
# requires the default on the parameter itself, so only the help/metadata is
# centralized here; the numeric defaults stay at each call site.
MaxResiduesOpt = Annotated[int, typer.Option(help="Max residues per embedding batch")]
MaxBatchOpt = Annotated[int, typer.Option(help="Max sequences per embedding batch")]


class Dataset(str, Enum):
    """The evaluation datasets accepted by `toxfam eval`.

    Typing the CLI ``dataset`` argument as this enum gives parse-time validation,
    a choices list in ``--help``, and shell completion. Kept in sync with
    ``toxfam.data.registry.DATASETS`` by a test; the runners keep their own
    ValueError as defense-in-depth.
    """

    test_set = "test_set"
    val_set = "val_set"
    non_metazoan = "non_metazoan"
    unreviewed = "unreviewed"


class EatMetric(str, Enum):
    """Distance metric for `toxfam eval eat`."""

    cosine = "cosine"
    euclidean = "euclidean"


# ---------- toxfam download-data ----------


GITHUB_REPO = "Sisistern123/ToxFam"
RELEASE_TAG = "data-v1"

_RAW = "raw"
_PROCESSED = "processed"
_INTERMEDIATE = "intermediate"
_EVALUATION = "evaluation"

DATA_ASSETS: list[tuple[str, str, str, str]] = [
    # (release asset name, target dir, relative path inside target, file to check for skip)
    ("0800.tsv", _RAW, "0800.tsv", "0800.tsv"),
    ("nontox.tsv", _RAW, "nontox.tsv", "nontox.tsv"),
    ("training_data.csv", _PROCESSED, "training_data.csv", "training_data.csv"),
    ("embeddings.h5", _PROCESSED, "embeddings.h5", "embeddings.h5"),
    ("hbi_train_all.csv", _PROCESSED, "hbi_train_all.csv", "hbi_train_all.csv"),
    ("hbi_train_all.fasta", _PROCESSED, "hbi_train_all.fasta", "hbi_train_all.fasta"),
    ("sp6_cache.zip", _INTERMEDIATE, "sp6", "sp6/sp6_cache.json"),
    ("evaluation_data.zip", _EVALUATION, ".", "non_metazoan/non_metazoan.tsv"),
]


def _download_with_progress(url: str, dest: Path, label: str) -> None:
    """Download a file with a rich progress bar."""
    import urllib.request

    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TransferSpeedColumn,
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with Progress(
            TextColumn("  {task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as progress:
            task = progress.add_task(label, total=total or None)
            with open(dest, "wb") as f:
                while chunk := resp.read(1024 * 64):
                    f.write(chunk)
                    progress.advance(task, len(chunk))


@app.command("download-data")
def download_data(
    tag: Annotated[str, typer.Option(help="GitHub release tag")] = RELEASE_TAG,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Re-download even if files exist")
    ] = False,
) -> None:
    """Download raw and processed data from GitHub Releases.

    Fetches UniProt TSVs (data/raw/), training splits and ProtT5 embeddings
    (data/processed/), and the SignalP6 per-sequence cache
    (data/intermediate/sp6/). Taxonomy vectors are not included — regenerate
    them with `toxfam taxonomy`. Existing files are skipped unless --force
    is set.
    """
    import os
    import tempfile
    import zipfile

    from toxfam._paths import (
        evaluation_data_dir,
        intermediate_dir,
        processed_dir,
        raw_dir,
    )

    dirs = {
        _RAW: raw_dir(),
        _PROCESSED: processed_dir(),
        _INTERMEDIATE: intermediate_dir(),
        _EVALUATION: evaluation_data_dir(),
    }
    base_url = f"https://github.com/{GITHUB_REPO}/releases/download/{tag}"

    for asset_name, dir_key, rel_path, skip_file in DATA_ASSETS:
        target_dir = dirs[dir_key]
        skip_path = target_dir / skip_file
        url = f"{base_url}/{asset_name}"

        if skip_path.exists() and not force:
            console.print(f"  skip {rel_path} (exists)")
            continue

        try:
            if asset_name.endswith(".zip"):
                extract_dir = target_dir / rel_path
                extract_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    _download_with_progress(url, tmp_path, asset_name)
                    with zipfile.ZipFile(tmp_path, "r") as zf:
                        zf.extractall(extract_dir)
                finally:
                    # Always clean up the temp archive, even if download/extract fails.
                    tmp_path.unlink(missing_ok=True)
            else:
                # Stream to a sibling .part file and atomically rename on success,
                # so an interrupted download can't leave a truncated file that the
                # skip-if-exists check above would treat as complete.
                dest = target_dir / rel_path
                tmp_dest = dest.with_name(dest.name + ".part")
                try:
                    _download_with_progress(url, tmp_dest, asset_name)
                    os.replace(tmp_dest, dest)
                finally:
                    tmp_dest.unlink(missing_ok=True)
        except Exception as e:
            err_console.print(f"  FAILED: {e}", style="red")
            raise typer.Exit(code=1)

    _verify_training_csv_against_manifest(processed_dir() / "training_data.csv")
    console.print("Done.")


def _verify_training_csv_against_manifest(training_csv: Path) -> None:
    """Check the downloaded release CSV describes the proteins the manifest pins.

    The split itself is always read from the git-tracked manifest, so a release CSV
    cannot redefine it. But a CSV describing a *different protein set* means the
    release and the checkout disagree about the dataset, and every downstream
    command would fail later and less clearly than here.
    """
    import pandas as pd

    from toxfam.data.split_manifest import SplitManifestError, load_manifest

    if not training_csv.exists():
        return
    try:
        manifest_ids = set(load_manifest()["identifier"])
    except SplitManifestError as e:
        err_console.print(f"  [yellow]Could not verify split manifest: {e}[/]")
        return

    csv_ids = set(pd.read_csv(training_csv, usecols=["identifier"])["identifier"])
    if csv_ids == manifest_ids:
        return

    err_console.print(
        f"  [red]{training_csv.name} does not match data/splits/split_manifest.csv[/]\n"
        f"    release CSV: {len(csv_ids)} proteins\n"
        f"    manifest:    {len(manifest_ids)} proteins\n"
        f"    only in CSV: {len(csv_ids - manifest_ids)}, "
        f"only in manifest: {len(manifest_ids - csv_ids)}\n"
        "  The data release and this checkout describe different protein sets. "
        "Check out the commit matching the release, or re-run 'toxfam preprocess'.",
        style="red",
    )
    raise typer.Exit(code=1)


# ---------- Step 1: toxfam preprocess ----------


@app.command()
def preprocess(
    signalp6_extra: Annotated[
        str, typer.Option(help="Extra args for SignalP6")
    ] = "--organism eukarya",
    min_seq_id: Annotated[
        float, typer.Option(help="MMseqs2 clustering identity threshold")
    ] = 0.9,
) -> None:
    """Run the full data preprocessing pipeline.

    Reads raw UniProt TSVs from data/raw/, normalizes family labels, removes
    signal peptides via SignalP6 (with MD5-based per-sequence caching), clusters
    sequences per family with MMseqs2, creates multilabel-stratified
    train/val/test splits, and writes the final training CSV to
    data/processed/training_data.csv.
    """
    from toxfam.data.preprocessing import run_preprocessing_pipeline

    run_preprocessing_pipeline(
        signalp6_extra=signalp6_extra,
        min_seq_id=min_seq_id,
    )


# ---------- Step 2: toxfam embed ----------


@app.command()
def embed(
    input_fasta: Annotated[
        Optional[Path],
        typer.Option(
            "-i", "--input", help="Input FASTA file", exists=True,
            show_default="data/intermediate/mmseqs/representatives/all.fasta",
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o", "--output", help="Output H5 file",
            show_default="data/processed/embeddings.h5",
        ),
    ] = None,
    model_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Cache directory for model",
            show_default="~/.cache/huggingface/hub/",
        ),
    ] = None,
    model_name: Annotated[
        str, typer.Option(help="HuggingFace model name")
    ] = "Rostlab/prot_t5_xl_half_uniref50-enc",
    max_residues: MaxResiduesOpt = 4000,
    max_batch: MaxBatchOpt = 100,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing H5 instead of resuming"),
    ] = False,
) -> None:
    """Generate per-protein ProtT5 embeddings from a FASTA file.

    Loads the ProtT5-XL-U50 encoder, batches sequences by length, and writes
    1024-dim mean-pool embeddings to an HDF5 file (one dataset per protein).
    Already-embedded sequences are skipped unless --force is set. Automatically
    selects the best available device (CUDA > MPS > CPU).
    """
    from toxfam._paths import intermediate_dir, processed_dir
    from toxfam.data.embedding import generate_embeddings

    if input_fasta is None:
        input_fasta = intermediate_dir() / "mmseqs" / "representatives" / "all.fasta"
    if output is None:
        output = processed_dir() / "embeddings.h5"
    # Fail fast (before loading ProtT5) if the resolved default input is missing —
    # an explicit --input is already validated by exists=True at parse time.
    if not input_fasta.exists():
        raise typer.BadParameter(f"Input FASTA not found: {input_fasta}")

    generate_embeddings(
        input_fasta=input_fasta,
        output_h5=output,
        model_dir=str(model_dir) if model_dir else None,
        model_name=model_name,
        max_residues=max_residues,
        max_batch=max_batch,
        force=force,
    )


# ---------- Step 3: toxfam taxonomy ----------


@app.command()
def taxonomy(
    input_csv: Annotated[
        Optional[Path],
        typer.Option(
            help="Training CSV with 'Organism (ID)' column", exists=True,
            show_default="data/processed/training_data.csv",
        ),
    ] = None,
    input_h5: Annotated[
        Optional[Path],
        typer.Option(
            help="Input H5 with protein embeddings", exists=True,
            show_default="data/processed/embeddings.h5",
        ),
    ] = None,
    output_h5: Annotated[
        Optional[Path],
        typer.Option(
            help="Output H5 for multi-hot taxonomy vectors",
            show_default="data/processed/taxonomy_vectors.h5",
        ),
    ] = None,
) -> None:
    """Generate multi-hot taxonomy vectors for the combined training strategy.

    Reads NCBI taxon IDs from the training CSV ('Organism (ID)' column),
    resolves full lineage via taxopy, and encodes membership in 50
    predefined animal taxa as multi-hot vectors stored in HDF5. Only
    proteins present in the input embeddings H5 are included.
    """
    from toxfam._paths import processed_dir
    from toxfam.data.taxonomy import run_multi_hot_taxonomy_pipeline

    proc = processed_dir()
    if input_csv is None:
        input_csv = proc / "training_data.csv"
    if input_h5 is None:
        input_h5 = proc / "embeddings.h5"
    if output_h5 is None:
        output_h5 = proc / "taxonomy_vectors.h5"

    # Fail fast (before creating the output dir) if a resolved default input is
    # missing — explicit inputs are already validated by exists=True at parse time.
    for hint, path in (("--input-csv", input_csv), ("--input-h5", input_h5)):
        if not path.exists():
            raise typer.BadParameter(f"{hint} file not found: {path}")

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    run_multi_hot_taxonomy_pipeline(
        input_csv=str(input_csv),
        input_h5_path=str(input_h5),
        output_h5_path=str(output_h5),
    )


# ---------- Step 4: toxfam train ----------


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to training config YAML (e.g. configs/standard.yaml)",
            exists=True,
            readable=True,
        ),
    ],
) -> None:
    """Train a toxin family classifier from a YAML config file.

    Loads training splits and embeddings, builds DataLoaders with class
    weighting, and trains either a standard MLP (embeddings only) or a
    combined two-branch MLP (embeddings + taxonomy vectors) depending on the
    training_strategy in the config. After training, applies temperature
    scaling calibration on the validation set. Outputs the best model,
    calibrated model, metrics JSON, predictions CSV, and plots to the
    configured output directory.
    """
    from toxfam.config import TrainConfig
    from toxfam.training.orchestrator import run_training

    cfg = TrainConfig.from_yaml(config)
    run_training(cfg)


# ---------- toxfam predict ----------


@app.command()
def predict(
    input_tsv: Annotated[
        Path,
        typer.Argument(
            help="Input TSV path, or a registered dataset name (e.g. non_metazoan, "
            "unreviewed). TSV needs 'identifier' (+ 'Organism (ID)' for combined "
            "models, + 'Sequence' if embeddings must be generated). A dataset name "
            "also auto-selects its embeddings H5.",
        ),
    ],
    model_dir: Annotated[
        Path,
        typer.Option(
            help="Primary model directory (combined or standard, auto-detected)",
            exists=True,
        ),
    ],
    standard_model_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Standard (ModularMLP) model for proteins without an organism ID. "
            "Only used when --model-dir is a combined model.",
            exists=True,
        ),
    ] = None,
    embeddings: Annotated[
        Optional[Path],
        typer.Option(
            "--embeddings",
            help="Precomputed ProtT5 embeddings H5 (keyed by identifier). Missing "
            "identifiers are embedded from the 'Sequence' column.",
            exists=True,
        ),
    ] = None,
    output: Annotated[
        Path, typer.Option("-o", "--output", help="Output TSV path")
    ] = Path("predictions.tsv"),
    top_k: Annotated[int, typer.Option(help="Number of top family predictions")] = 3,
    toxicity_only: Annotated[
        bool,
        typer.Option(
            "--toxicity-only",
            help="Only predict toxic/non-toxic (skip family prediction columns)",
        ),
    ] = False,
    max_residues: MaxResiduesOpt = 4000,
    max_batch: MaxBatchOpt = 100,
) -> None:
    """Predict toxin family and toxicity for arbitrary proteins (no labels needed).

    Reads a TSV of proteins and runs a trained model to produce the top-K family
    predictions with calibrated confidences plus a binary toxic/non-toxic call.
    ProtT5 embeddings are reused from --embeddings when available and generated
    on demand otherwise. Three usages, dispatched on the primary model:

    \b
    * Combined model only: proteins without an organism ID are excluded.
    * Combined + --standard-model-dir: proteins with an organism ID go to the
      combined model, those without to the standard model (two output TSVs).
    * Standard model: organism IDs ignored, all proteins predicted.
    """
    from toxfam.prediction import run_prediction

    run_prediction(
        input_tsv,
        model_dir,
        standard_model_dir=standard_model_dir,
        embeddings_h5=embeddings,
        output=output,
        top_k=top_k,
        toxicity_only=toxicity_only,
        max_residues=max_residues,
        max_batch=max_batch,
    )


# ---------- Step 5: toxfam eval {hbi,model,compare} ----------

eval_app = typer.Typer(
    help="Run evaluations and compare methods.",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval")


@eval_app.command("hbi")
def eval_hbi(
    dataset: Annotated[
        Dataset,
        typer.Argument(help="Dataset to evaluate"),
    ],
) -> None:
    """Run HBI (homology-based inference) on a dataset.

    Searches query sequences against the training database via MMseqs2 and
    transfers the family label from the best hit. Results are saved to
    benchmark/{dataset}/hbi/.
    """
    from toxfam.evaluation.runner import run_hbi_evaluation

    run_hbi_evaluation(dataset.value)


@eval_app.command("eat")
def eval_eat(
    dataset: Annotated[
        Dataset,
        typer.Argument(help="Dataset to evaluate"),
    ],
    metric: Annotated[
        EatMetric,
        typer.Option(help="Distance metric (cosine selected on val_set)"),
    ] = EatMetric.cosine,
) -> None:
    """Run EAT (embedding-based annotation transfer) on a dataset.

    Transfers the family label of each query's nearest ProtT5 neighbour (k=1)
    among the training split — the embedding-space analog of HBI. Results are
    saved to benchmark/{dataset}/eat/.
    """
    from toxfam.evaluation.runner import run_eat_evaluation

    run_eat_evaluation(dataset.value, metric=metric.value)


@eval_app.command("model")
def eval_model(
    dataset: Annotated[
        Dataset,
        typer.Argument(help="Dataset to evaluate"),
    ],
    model_dir: Annotated[
        Path,
        typer.Option(
            help="Training output directory containing models/ and class_indices.json"
        ),
    ],
) -> None:
    """Run a trained neural network model on a dataset.

    Loads the calibrated model from the training output directory, runs
    inference on all sequences, and saves results to
    benchmark/{dataset}/nn_{model_dir_name}/.
    """
    from toxfam.evaluation.runner import run_model_evaluation

    run_model_evaluation(dataset.value, model_dir)


@eval_app.command("compare")
def eval_compare(
    dataset: Annotated[
        Dataset,
        typer.Argument(help="Dataset to compare methods for"),
    ],
) -> None:
    """Compare all evaluated methods for a dataset.

    Scans benchmark/{dataset}/ for method results and prints a side-by-side
    comparison table. Writes metric_comparison.csv and full_report.json to
    benchmark/{dataset}/comparison/.
    """
    from toxfam.evaluation.runner import compare_methods

    compare_methods(dataset.value)


@eval_app.command("binary")
def eval_binary(
    model_dir: Annotated[
        Path,
        typer.Argument(
            help="Model output directory containing config.yaml and models/",
            exists=True,
        ),
    ],
) -> None:
    """Re-compute binary toxic/nontoxin metrics from a trained model.

    Loads the calibrated model and config from the model output directory,
    computes P(toxic) for val and test sets, optimizes the threshold on val
    (Youden's J), and evaluates on test with both default and optimized
    thresholds. Saves binary_metrics.json and ROC/PR plots.
    """
    from toxfam.evaluation.runner import run_binary_evaluation_from_dir

    run_binary_evaluation_from_dir(model_dir)


# ---------- toxfam plot ----------

plot_app = typer.Typer(
    help="Generate plots and visualizations.",
    no_args_is_help=True,
)
app.add_typer(plot_app, name="plot")


@plot_app.command("taxonomy")
def plot_taxonomy() -> None:
    """Generate taxonomy sunburst plots for toxin and non-toxin proteins.

    Reads the training CSV, resolves NCBI lineages via taxopy, and creates
    interactive sunburst charts (HTML + PNG) showing the taxonomic distribution
    of toxin and non-toxin proteins. Outputs to figures/taxonomy/.
    """
    from toxfam.visualization.taxonomy_sunburst import main as _main

    _main()


def main():
    app()
