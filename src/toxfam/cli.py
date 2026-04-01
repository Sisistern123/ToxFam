"""Unified CLI for ToxFam."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="toxfam",
    help="Animal toxin protein family classification using MLP on ProtT5 embeddings.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


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
            typer.echo(f"  skip {rel_path} (exists)")
            continue

        try:
            if asset_name.endswith(".zip"):
                extract_dir = target_dir / rel_path
                extract_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                _download_with_progress(url, tmp_path, asset_name)
                with zipfile.ZipFile(tmp_path, "r") as zf:
                    zf.extractall(extract_dir)
                tmp_path.unlink()
            else:
                dest = target_dir / rel_path
                _download_with_progress(url, dest, asset_name)
        except Exception as e:
            typer.echo(f"  FAILED: {e}", err=True)
            raise typer.Exit(code=1)

    typer.echo("Done.")


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
    input: Annotated[
        Path,
        typer.Option("-i", "--input", help="Input FASTA file", exists=True),
    ] = Path("data/intermediate/mmseqs/representatives/all.fasta"),
    output: Annotated[
        Path, typer.Option("-o", "--output", help="Output H5 file")
    ] = Path("data/processed/embeddings.h5"),
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
    max_residues: Annotated[int, typer.Option(help="Max residues per batch")] = 4000,
    max_batch: Annotated[int, typer.Option(help="Max sequences per batch")] = 100,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite existing H5 instead of resuming")
    ] = False,
) -> None:
    """Generate per-protein ProtT5 embeddings from a FASTA file.

    Loads the ProtT5-XL-U50 encoder, batches sequences by length, and writes
    1024-dim mean-pool embeddings to an HDF5 file (one dataset per protein).
    Already-embedded sequences are skipped unless --force is set. Automatically
    selects the best available device (CUDA > MPS > CPU).
    """
    from toxfam.data.embedding import generate_embeddings

    generate_embeddings(
        input_fasta=input,
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
        Path,
        typer.Option(help="Training CSV with 'Organism (ID)' column", exists=True),
    ] = Path("data/processed/training_data.csv"),
    input_h5: Annotated[
        Path,
        typer.Option(help="Input H5 with protein embeddings", exists=True),
    ] = Path("data/processed/embeddings.h5"),
    output_h5: Annotated[
        Path, typer.Option(help="Output H5 for multi-hot taxonomy vectors")
    ] = Path("data/processed/taxonomy_vectors.h5"),
) -> None:
    """Generate multi-hot taxonomy vectors for the combined training strategy.

    Reads NCBI taxon IDs from the training CSV ('Organism (ID)' column),
    resolves full lineage via taxopy, and encodes membership in 50
    predefined animal taxa as multi-hot vectors stored in HDF5. Only
    proteins present in the input embeddings H5 are included.
    """
    from toxfam.data.taxonomy import run_multi_hot_taxonomy_pipeline

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


# ---------- Step 5a: toxfam eval-test ----------


@app.command("eval-test")
def eval_test(
    model_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory containing model outputs"),
    ] = None,
) -> None:
    """Compare neural network vs homology-based inference on the test set.

    Runs an MMseqs2 sequence search (HBI) against training sequences and
    loads pre-computed NN predictions from the model directory. Computes
    accuracy, MCC, and per-class F1 for both methods, saves a side-by-side
    comparison CSV, confusion matrices, and a detailed classification report
    to benchmark/test_set/.
    """
    from toxfam.evaluation.eval_test_set import run_eval_test_set

    run_eval_test_set(model_dir=model_dir)


# ---------- Step 5b: toxfam eval-nonmetazoan ----------


@app.command("eval-nonmetazoan")
def eval_nonmetazoan(
    h5_path: Annotated[
        Path,
        typer.Option(help="H5 file with embeddings", exists=True),
    ],
    model_path: Annotated[
        Path,
        typer.Option(help="Path to calibrated model .pt file", exists=True),
    ],
    class_map: Annotated[
        Path,
        typer.Option(help="Path to class_indices.json", exists=True),
    ],
) -> None:
    """Binary toxin/non-toxin evaluation on non-metazoan reviewed proteins.

    Loads reviewed non-metazoan sequences, runs both an MMseqs2 homology
    search (HBI) and the trained model to predict toxin vs non-toxin, then
    computes binary accuracy and MCC for each method. Results are saved to
    benchmark/non_metazoan/.
    """
    from toxfam.evaluation.eval_nonmetazoan import run_eval_nonmetazoan

    run_eval_nonmetazoan(
        h5_path=h5_path,
        model_path=model_path,
        class_map_path=class_map,
    )


# ---------- Step 5c: toxfam eval-unreviewed ----------


@app.command("eval-unreviewed")
def eval_unreviewed(
    input_tsv: Annotated[
        Path,
        typer.Option(help="Input TSV with protein data", exists=True),
    ],
    input_h5: Annotated[
        Path,
        typer.Option(help="Input H5 embeddings", exists=True),
    ],
    train_data: Annotated[
        Optional[Path],
        typer.Option(help="Training data CSV"),
    ] = None,
    train_fasta: Annotated[
        Optional[Path],
        typer.Option(help="Training FASTA file"),
    ] = None,
) -> None:
    """Multi-class family evaluation on unreviewed metazoan proteins.

    Normalizes family labels (same rules as preprocessing), runs an MMseqs2
    homology search (HBI) against training sequences, and computes multi-class
    accuracy, MCC, and per-class F1. Results are saved to
    benchmark/unreviewed/.
    """
    from toxfam.evaluation.eval_unreviewed import run_eval_unreviewed

    run_eval_unreviewed(
        input_tsv=input_tsv,
        input_h5=input_h5,
        train_data=train_data,
        train_fasta=train_fasta,
    )


# ---------- Step 5d: toxfam eval-binary ----------


@app.command("eval-ensemble")
def eval_ensemble_cmd(
    model_dirs: Annotated[
        list[Path],
        typer.Argument(help="Model output directories to ensemble"),
    ],
) -> None:
    """Evaluate an ensemble of trained models.

    Averages softmax predictions from multiple calibrated models, then
    computes binary metrics with threshold optimization.
    """
    from toxfam.evaluation.ensemble import evaluate_ensemble

    evaluate_ensemble(model_dirs)


@app.command("profile-data")
def profile_data(
    input_csv: Annotated[
        Path,
        typer.Option(help="Training CSV to profile", exists=True),
    ] = Path("data/processed/training_data.csv"),
    h5_path: Annotated[
        Optional[Path],
        typer.Option(help="Embeddings H5 for similarity analysis"),
    ] = None,
) -> None:
    """Profile training data for biases and class imbalance.

    Analyzes class distribution, organism diversity, sequence lengths,
    and optionally embedding similarities. Flags potential biases.
    """
    from toxfam.evaluation.data_quality import profile_training_data

    profile_training_data(
        str(input_csv), h5_path=str(h5_path) if h5_path else None
    )


@app.command()
def cpp(
    training_csv: Annotated[
        Path,
        typer.Option(help="Training CSV with family labels", exists=True),
    ] = Path("data/processed/training_data.csv"),
    output: Annotated[
        Path,
        typer.Option(help="Output H5 for CPP features"),
    ] = Path("data/intermediate/cpp/cpp_features.h5"),
    n_filter: Annotated[
        int,
        typer.Option(help="Number of CPP features to select"),
    ] = 100,
) -> None:
    """Generate CPP (Comparative Physicochemical Profiling) features.

    Uses AAanalysis to compute discriminative physicochemical properties
    between toxic and non-toxic sequences. Outputs an HDF5 file with one
    feature vector per protein.
    """
    from toxfam.data.cpp_features import run_cpp_pipeline

    output.parent.mkdir(parents=True, exist_ok=True)
    run_cpp_pipeline(
        training_csv=str(training_csv),
        output_h5=str(output),
        n_filter=n_filter,
    )


@app.command("eval-binary")
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
    import json as _json

    import pandas as pd
    import torch

    from toxfam.config import TrainConfig
    from toxfam.data.dataset import ToxDataset, analyze_data_splits
    from toxfam.evaluation.metrics import (
        calculate_binary_metrics_with_scores,
        find_optimal_threshold,
    )
    from toxfam.model.architectures import ModularMLP, MultiInputMLP
    from toxfam.model.calibration import ModelWithTemperature
    from toxfam.training.orchestrator import (
        _compute_binary_labels,
        _compute_p_toxic,
    )
    from toxfam.training.trainer import get_device
    from toxfam.visualization.analysis import plot_binary_pr, plot_binary_roc

    config = TrainConfig.from_yaml(model_dir / "config.yaml")
    config = config.model_copy(update={"output_dir": model_dir})

    df = pd.read_csv(config.input_csv)
    train_df, val_df, test_df = analyze_data_splits(df)

    h5_paths = [str(p) for p in config.h5_paths]
    train_ds = ToxDataset(train_df, h5_paths, is_train=True)

    # Load calibrated model
    device = get_device()
    calibrated_path = model_dir / "models" / "best_model_calibrated.pt"
    if not calibrated_path.exists():
        typer.echo(f"Calibrated model not found at {calibrated_path}", err=True)
        raise typer.Exit(code=1)

    class_map_path = model_dir / "class_indices.json"
    with open(class_map_path) as f:
        class_map = _json.load(f)
    num_classes = len(class_map)

    if config.training_strategy == "combined":
        base_model = MultiInputMLP(
            embed_dim=config.effective_embedding_dim,
            tax_dim=config.tax_dim,
            hidden_dims=config.hidden_dims,
            num_classes=num_classes,
            dropout=config.dropout,
        )
    else:
        base_model = ModularMLP(
            input_dim=config.effective_embedding_dim,
            hidden_dims=config.hidden_dims,
            num_classes=num_classes,
            dropout=config.dropout,
        )

    scaled_model = ModelWithTemperature(base_model, device)
    scaled_model.load_state_dict(
        torch.load(calibrated_path, map_location=device, weights_only=True)
    )
    scaled_model.eval()

    label_col = "Protein families"
    val_y_true = _compute_binary_labels(val_df, label_col)
    val_p_toxic = _compute_p_toxic(scaled_model, val_df, config, train_ds.le, label_col)
    thresh_result = find_optimal_threshold(val_y_true, val_p_toxic)
    opt_threshold = thresh_result["optimal_threshold"]

    test_y_true = _compute_binary_labels(test_df, label_col)
    test_p_toxic = _compute_p_toxic(
        scaled_model, test_df, config, train_ds.le, label_col
    )

    test_default = calculate_binary_metrics_with_scores(
        test_y_true, test_p_toxic, threshold=0.5
    )
    test_opt = calculate_binary_metrics_with_scores(
        test_y_true, test_p_toxic, threshold=opt_threshold
    )

    typer.echo(f"Threshold (Youden): {opt_threshold:.4f}")
    typer.echo(
        f"Test (t=0.5):   ROC-AUC={test_default['roc_auc']:.4f} "
        f"PR-AUC={test_default['pr_auc']:.4f} MCC={test_default['mcc']:.4f}"
    )
    typer.echo(
        f"Test (t={opt_threshold:.3f}): ROC-AUC={test_opt['roc_auc']:.4f} "
        f"PR-AUC={test_opt['pr_auc']:.4f} MCC={test_opt['mcc']:.4f}"
    )

    metrics_dir = model_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    _curve_keys = {"fpr", "tpr", "precision_curve", "recall_curve", "roc_thresholds", "pr_thresholds"}
    results = {
        "optimized_threshold": opt_threshold,
        "test_default": {k: v for k, v in test_default.items() if k not in _curve_keys},
        "test_optimized": {k: v for k, v in test_opt.items() if k not in _curve_keys},
    }
    (metrics_dir / "binary_metrics.json").write_text(_json.dumps(results, indent=4))

    plots_dir = model_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_binary_roc(
        test_default["fpr"],
        test_default["tpr"],
        test_default["roc_auc"],
        plots_dir / "binary_roc.png",
    )
    plot_binary_pr(
        test_default["precision_curve"],
        test_default["recall_curve"],
        test_default["pr_auc"],
        plots_dir / "binary_pr.png",
    )

    train_ds.close()
    typer.echo("Binary metrics saved.")


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
