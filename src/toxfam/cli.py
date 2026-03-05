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

DATA_ASSETS: list[tuple[str, str, str, str]] = [
    # (release asset name, target dir, relative path inside target, file to check for skip)
    ("0800.tsv", _RAW, "0800.tsv", "0800.tsv"),
    ("nontox.tsv", _RAW, "nontox.tsv", "nontox.tsv"),
    ("training_data.csv", _PROCESSED, "training_data.csv", "training_data.csv"),
    ("embeddings.h5", _PROCESSED, "embeddings.h5", "embeddings.h5"),
    ("sp6_cache.zip", _INTERMEDIATE, "sp6", "sp6/sp6_cache.json"),
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
    tag: Annotated[
        str, typer.Option(help="GitHub release tag")
    ] = RELEASE_TAG,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Re-download even if files exist")
    ] = False,
) -> None:
    """Download raw and processed data from GitHub Releases."""
    import tempfile
    import zipfile

    from toxfam._paths import intermediate_dir, processed_dir, raw_dir

    dirs = {_RAW: raw_dir(), _PROCESSED: processed_dir(), _INTERMEDIATE: intermediate_dir()}
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
    """Run the data preprocessing pipeline."""
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
    """Generate per-protein ProtT5 embeddings from a FASTA file."""
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
        Path, typer.Option(help="Output H5 for binary taxonomy vectors")
    ] = Path("data/intermediate/taxonomy/binary_taxonomy_vectors.h5"),
) -> None:
    """Generate binary taxonomy vectors from training CSV with taxon IDs."""
    from toxfam.data.taxonomy import run_binary_taxonomy_pipeline

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    run_binary_taxonomy_pipeline(
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
    """Train a model using the specified config file."""
    from toxfam.config import TrainConfig
    from toxfam.training.orchestrator import run_training

    cfg = TrainConfig.from_yaml(config)
    if cfg.n_folds > 1:
        from toxfam.training.cross_validation import run_kfold_training

        run_kfold_training(cfg, n_folds=cfg.n_folds)
    else:
        run_training(cfg)


# ---------- Step 5a: toxfam eval-test ----------


@app.command("eval-test")
def eval_test(
    model_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory containing model outputs"),
    ] = None,
) -> None:
    """Evaluate on test set (HBI vs NN comparison)."""
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
    """Evaluate on non-metazoan reviewed proteins."""
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
    input_fasta: Annotated[
        Path,
        typer.Option(help="Input FASTA file", exists=True),
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
    """Evaluate on unreviewed metazoan proteins."""
    from toxfam.evaluation.eval_unreviewed import run_eval_unreviewed

    run_eval_unreviewed(
        input_tsv=input_tsv,
        input_fasta=input_fasta,
        input_h5=input_h5,
        train_data=train_data,
        train_fasta=train_fasta,
    )


# ---------- toxfam explore-data ----------


@app.command("explore-data")
def explore_data(
    xml: Annotated[
        Path,
        typer.Option(help="UniProt XML file", exists=True),
    ],
    h5: Annotated[
        Path,
        typer.Option(help="ProtT5 embeddings H5 file", exists=True),
    ],
) -> None:
    """Explore a new UniProt XML dataset: family distribution, overlap with existing data."""
    import subprocess
    import sys

    from toxfam._paths import get_project_root

    script = get_project_root() / "scripts" / "explore_new_data.py"
    subprocess.run(
        [sys.executable, str(script), "--xml", str(xml), "--h5", str(h5)],
        check=True,
    )


# ---------- toxfam parse-xml ----------


@app.command("parse-xml")
def parse_xml(
    xml: Annotated[
        Path,
        typer.Option(help="UniProt XML file", exists=True),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Output CSV file"),
    ] = Path("data/processed/parsed_toxins.csv"),
) -> None:
    """Parse a UniProt XML file into a CSV matching ToxFam conventions."""
    from toxfam.data.xml_parser import parse_uniprot_xml

    output.parent.mkdir(parents=True, exist_ok=True)
    df = parse_uniprot_xml(xml)
    df.to_csv(output, index=False)
    typer.echo(f"Parsed {len(df)} entries → {output}")


# ---------- toxfam hierarchical-preprocess ----------


@app.command("hierarchical-preprocess")
def hierarchical_preprocess(
    xml: Annotated[
        Optional[Path],
        typer.Option(help="UniProt XML file (defaults to ToxProtFeb2026 XML)"),
    ] = None,
    min_seq_id: Annotated[
        float, typer.Option(help="MMseqs2 clustering identity threshold")
    ] = 0.9,
    family_min_count: Annotated[
        int, typer.Option(help="Min members per family after clustering")
    ] = 10,
    max_nontox_per_family: Annotated[
        int, typer.Option(help="Max nontox proteins per family")
    ] = 200,
    skip_signalp6: Annotated[
        bool, typer.Option("--skip-signalp6", help="Skip SignalP6 step")
    ] = False,
    skip_mmseqs2_search: Annotated[
        bool, typer.Option("--skip-mmseqs2-search", help="Skip MMseqs2 nontox search")
    ] = False,
) -> None:
    """Run the hierarchical data assembly pipeline (merge sources, find nontox, cluster, split)."""
    from toxfam.data.hierarchical_preprocessing import run_hierarchical_preprocessing

    run_hierarchical_preprocessing(
        xml_path=xml,
        min_seq_id=min_seq_id,
        family_min_count=family_min_count,
        max_nontox_per_family=max_nontox_per_family,
        skip_signalp6=skip_signalp6,
        skip_mmseqs2_search=skip_mmseqs2_search,
    )


# ---------- toxfam cpp ----------


@app.command("cpp")
def cpp(
    training_csv: Annotated[
        Path,
        typer.Option(help="Training CSV with is_toxic column", exists=True),
    ] = Path("data/processed/hierarchical_training_data.csv"),
    output_h5: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Output H5 file"),
    ] = None,
    n_filter: Annotated[
        int, typer.Option(help="Number of CPP features to select")
    ] = 100,
    subsample_ratio: Annotated[
        float, typer.Option(help="Nontox-to-toxic ratio for feature selection (1.0 = balanced)")
    ] = 1.0,
) -> None:
    """Generate CPP physicochemical features (tox vs nontox) via AAanalysis."""
    from toxfam.data.cpp_features import run_cpp_pipeline

    run_cpp_pipeline(
        training_csv=training_csv,
        output_h5=output_h5,
        n_filter=n_filter,
        subsample_ratio=subsample_ratio,
    )


# ---------- toxfam eval-binary ----------


@app.command("eval-binary")
def eval_binary(
    model_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing trained model outputs", exists=True),
    ],
    config_path: Annotated[
        Optional[Path],
        typer.Option(help="Config YAML (defaults to model_dir/config.yaml)"),
    ] = None,
) -> None:
    """Re-compute binary toxic/non-toxic metrics for a trained model without retraining."""
    import json

    import torch

    from toxfam.config import TrainConfig
    from toxfam.data.dataset import analyze_data_splits
    from toxfam.device import get_device
    from toxfam.model.calibration import ModelWithTemperature

    # Load config
    if config_path is None:
        config_path = model_dir / "config.yaml"
    if not config_path.exists():
        typer.echo(f"Config not found at {config_path}", err=True)
        raise typer.Exit(code=1)

    cfg = TrainConfig.from_yaml(config_path)
    # Override output_dir to model_dir
    cfg = cfg.model_copy(update={"output_dir": model_dir})

    # Load class indices
    class_json = model_dir / "class_indices.json"
    if not class_json.exists():
        typer.echo(f"class_indices.json not found in {model_dir}", err=True)
        raise typer.Exit(code=1)

    with open(class_json) as f:
        class_map = json.load(f)
    classes = [class_map[str(i)] for i in range(len(class_map))]

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.classes_ = __import__("numpy").array(classes)

    # Load model
    device = get_device()
    calibrated_path = model_dir / "best_model_calibrated.pt"
    if not calibrated_path.exists():
        typer.echo(f"Calibrated model not found at {calibrated_path}", err=True)
        raise typer.Exit(code=1)

    # Determine label_col and model architecture based on strategy
    strategy = cfg.training_strategy
    label_col = "Protein families"
    if strategy in ("binary", "hierarchical"):
        label_col = "is_toxic_label"

    # Load test data
    import pandas as pd

    df = pd.read_csv(cfg.input_csv)
    _, _, test_df = analyze_data_splits(df)
    if "is_toxic" not in test_df.columns:
        test_df["is_toxic"] = test_df["Protein families"] != "nontox"
    if label_col == "is_toxic_label":
        test_df[label_col] = test_df["is_toxic"].map({True: "toxic", False: "nontoxic"})

    # Reconstruct model and load weights
    from toxfam.training.orchestrator import _compute_and_save_binary_metrics

    # Build the model architecture to load state dict
    num_classes = len(classes)
    if strategy == "multitask":
        from toxfam.model.architectures import MultiTaskMLP
        from toxfam.training.strategies import _MultiTaskFamilyWrapper

        base_model = MultiTaskMLP(
            input_dim=cfg.effective_embedding_dim,
            hidden_dims=cfg.hidden_dims,
            num_family_classes=num_classes,
            dropout=cfg.dropout,
        )
        base_model = _MultiTaskFamilyWrapper(base_model)
    elif strategy == "hierarchical":
        from toxfam.model.architectures import HierarchicalMLP, ModularMLP

        projector = ModularMLP(
            input_dim=cfg.effective_embedding_dim,
            hidden_dims=cfg.hidden_dims,
            num_classes=2,
            dropout=cfg.dropout,
        ).projector
        base_model = HierarchicalMLP(
            backbone=projector,
            backbone_out_dim=cfg.hidden_dims[0],
            num_classes=num_classes,
            dropout=cfg.dropout,
            freeze_backbone=cfg.stage2_freeze_backbone,
            head_hidden_dim=cfg.stage2_hidden_dim,
        )
    elif strategy == "combined":
        from toxfam.model.architectures import MultiInputMLP

        base_model = MultiInputMLP(
            embed_dim=cfg.effective_embedding_dim,
            tax_dim=cfg.tax_dim,
            hidden_dims=cfg.hidden_dims,
            num_classes=num_classes,
            dropout=cfg.dropout,
        )
    else:
        from toxfam.model.architectures import ModularMLP

        base_model = ModularMLP(
            input_dim=cfg.effective_embedding_dim,
            hidden_dims=cfg.hidden_dims,
            num_classes=num_classes,
            dropout=cfg.dropout,
        )

    scaled_model = ModelWithTemperature(base_model, device)
    scaled_model.load_state_dict(
        torch.load(calibrated_path, map_location=device, weights_only=True)
    )
    scaled_model.to(device)

    typer.echo("Re-computing binary metrics...")
    _compute_and_save_binary_metrics(
        scaled_model,
        test_df,
        label_col,
        le,
        cfg,
        model_dir,
        tag="Test_Calibrated",
    )
    typer.echo("Done.")


# ---------- toxfam eval-ensemble ----------


@app.command("eval-ensemble")
def eval_ensemble_cmd(
    model_dirs: Annotated[
        list[Path],
        typer.Argument(help="Directories containing trained models"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("-o", "--output-dir", help="Output directory for ensemble results"),
    ] = Path("model/model_output/ensemble"),
) -> None:
    """Evaluate an ensemble of trained models on the test set."""
    from toxfam.evaluation.ensemble import evaluate_ensemble

    result = evaluate_ensemble(model_dirs, output_dir=output_dir)
    typer.echo(f"Ensemble ROC-AUC: {result.get('roc_auc', 'N/A')}")
    typer.echo(f"Ensemble PR-AUC: {result.get('pr_auc', 'N/A')}")


# ---------- toxfam hbi-baseline ----------


@app.command("hbi-baseline")
def hbi_baseline_cmd(
    input_csv: Annotated[
        Path,
        typer.Option(help="Training CSV with Split column", exists=True),
    ] = Path("data/processed/training_data.csv"),
    output_dir: Annotated[
        Path,
        typer.Option("-o", "--output-dir", help="Output directory for HBI results"),
    ] = Path("model/model_output/hbi_baselines"),
) -> None:
    """Run homology-based inference binary baselines (MMseqs2)."""
    from toxfam.evaluation.hbi_binary_baseline import run_hbi_binary_baselines

    run_hbi_binary_baselines(input_csv=input_csv, output_dir=output_dir)


# ---------- toxfam profile-data ----------


@app.command("profile-data")
def profile_data(
    input_csv: Annotated[
        Path,
        typer.Option(help="Training CSV file", exists=True),
    ] = Path("data/processed/training_data.csv"),
    h5_path: Annotated[
        Optional[Path],
        typer.Option(help="Embeddings H5 file for similarity analysis"),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("-o", "--output-dir", help="Output directory for profile report"),
    ] = Path("data/profile"),
) -> None:
    """Profile training data for potential biases."""
    from toxfam.evaluation.data_quality import profile_training_data

    profile_training_data(
        input_csv=input_csv,
        h5_path=h5_path,
        output_dir=output_dir,
    )


# ---------- toxfam benchmark-external ----------


@app.command("benchmark-external")
def benchmark_external_cmd(
    input_csv: Annotated[
        Path,
        typer.Option(help="Training CSV with Split column", exists=True),
    ] = Path("data/processed/training_data.csv"),
    output_dir: Annotated[
        Path,
        typer.Option("-o", "--output-dir", help="Output directory"),
    ] = Path("model/model_output/external_benchmarks"),
) -> None:
    """Run external tool benchmarks (ToxinPred2, etc.) on the test set."""
    from toxfam.evaluation.external_benchmarks import run_all_external_benchmarks

    run_all_external_benchmarks(input_csv=input_csv, output_dir=output_dir)


# ---------- toxfam fetch-counterparts ----------


@app.command("fetch-counterparts")
def fetch_counterparts_cmd(
    include_trembl: Annotated[
        bool,
        typer.Option("--include-trembl", help="Also search TrEMBL (unreviewed, existence 1-3)"),
    ] = False,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("-o", "--output-dir", help="Output directory"),
    ] = None,
) -> None:
    """Fetch non-toxic structural counterparts from UniProt + compute embeddings."""
    from toxfam.data.counterpart_acquisition import run_counterpart_pipeline

    h5_path = run_counterpart_pipeline(
        include_trembl=include_trembl,
        output_dir=output_dir,
    )
    if h5_path:
        typer.echo(f"\nCounterpart embeddings saved to: {h5_path}")


# ---------- toxfam eval-comparison ----------


@app.command("eval-comparison")
def eval_comparison_cmd(
    training_csv: Annotated[
        Path,
        typer.Option(help="Training CSV with Split column", exists=True),
    ] = Path("data/processed/training_data.csv"),
    hbi_h5: Annotated[
        Optional[Path],
        typer.Option(help="HBI features H5 file"),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("-o", "--output-dir", help="Output directory"),
    ] = Path("model/model_output/comparison"),
) -> None:
    """Run full method comparison: per-family eval + publication figures."""
    from toxfam.evaluation.comparison import run_full_comparison

    run_full_comparison(
        training_csv=training_csv,
        hbi_h5_path=hbi_h5,
        output_dir=output_dir,
    )


# ---------- toxfam compute-hbi ----------


@app.command("compute-hbi")
def compute_hbi_cmd(
    training_csv: Annotated[
        Optional[Path],
        typer.Option("--training-csv", help="Training CSV with Split column"),
    ] = None,
    output_h5: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Output H5 file for HBI features"),
    ] = None,
) -> None:
    """Pre-compute HBI features for all sequences (leave-one-out for train)."""
    from toxfam.data.hbi_features import compute_hbi_features

    result_path = compute_hbi_features(
        training_csv=training_csv,
        output_h5=output_h5,
    )
    typer.echo(f"\nHBI features saved to: {result_path}")


# ---------- toxfam handcrafted-features ----------


@app.command("handcrafted-features")
def handcrafted_features_cmd(
    input_csv: Annotated[
        Path,
        typer.Option(help="Training CSV with identifier + Sequence columns", exists=True),
    ] = Path("data/processed/training_data.csv"),
    output_h5: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Output H5 file"),
    ] = None,
) -> None:
    """Compute handcrafted features (Atchley factors + cysteine patterns) for all sequences."""
    from toxfam.data.handcrafted_features import run_handcrafted_pipeline

    result = run_handcrafted_pipeline(input_csv=input_csv, output_h5=output_h5)
    typer.echo(f"\nHandcrafted features saved to: {result}")


def main():
    app()
