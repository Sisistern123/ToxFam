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

_PROCESSED = "processed"
_INTERMEDIATE = "intermediate"

DATA_ASSETS: list[tuple[str, str, str, str]] = [
    # (release asset name, target dir, relative path inside target, file to check for skip)
    ("training_data.csv", _PROCESSED, "training_data.csv", "training_data.csv"),
    ("training_data.h5", _PROCESSED, "embeddings.h5", "embeddings.h5"),
    ("sp6_cache.zip", _INTERMEDIATE, "sp6", "sp6/sp6_cache.json"),
]


@app.command("download-data")
def download_data(
    tag: Annotated[
        str, typer.Option(help="GitHub release tag")
    ] = RELEASE_TAG,
) -> None:
    """Download processed data (embeddings, training splits, SP6 cache) from GitHub Releases."""
    import tempfile
    import urllib.request
    import zipfile

    from toxfam._paths import intermediate_dir, processed_dir

    dirs = {_PROCESSED: processed_dir(), _INTERMEDIATE: intermediate_dir()}
    base_url = f"https://github.com/{GITHUB_REPO}/releases/download/{tag}"

    for asset_name, dir_key, rel_path, skip_file in DATA_ASSETS:
        target_dir = dirs[dir_key]
        skip_path = target_dir / skip_file

        if skip_path.exists():
            typer.echo(f"  skip {rel_path} (exists)")
            continue

        url = f"{base_url}/{asset_name}"
        typer.echo(f"  downloading {asset_name} ...")

        try:
            if asset_name.endswith(".zip"):
                extract_dir = target_dir / rel_path
                extract_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                urllib.request.urlretrieve(url, tmp_path)
                with zipfile.ZipFile(tmp_path, "r") as zf:
                    zf.extractall(extract_dir)
                tmp_path.unlink()
            else:
                dest = target_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, dest)
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


def main():
    app()
