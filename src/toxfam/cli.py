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


# ---------- Step 0: toxfam download-data ----------


GITHUB_REPO = "Sisistern123/ToxFam"
RELEASE_TAG = "data-v1"

# Files to download and their relative paths under data/processed/
DATA_ASSETS = {
    "training_data.csv": "training_data.csv",
    "training_data.h5": "embeddings/training_data.h5",
    "training_tax.csv": "taxonomy/training_tax.csv",
    "binary_taxonomy_vectors.h5": "taxonomy/binary_taxonomy_vectors.h5",
}


@app.command("download-data")
def download_data(
    tag: Annotated[
        str, typer.Option(help="GitHub release tag to download from")
    ] = RELEASE_TAG,
) -> None:
    """Download processed data files from GitHub Releases."""
    import urllib.request

    from toxfam._paths import processed_dir

    proc = processed_dir()
    base_url = f"https://github.com/{GITHUB_REPO}/releases/download/{tag}"

    for asset_name, rel_path in DATA_ASSETS.items():
        dest = proc / rel_path
        if dest.exists():
            typer.echo(f"  skip {rel_path} (exists)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base_url}/{asset_name}"
        typer.echo(f"  downloading {rel_path} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            typer.echo(f"  FAILED: {e}", err=True)
            raise typer.Exit(code=1)

    typer.echo("Done.")


# ---------- Step 1: toxfam preprocess ----------


@app.command()
def preprocess(
    run_signalp6: Annotated[
        bool, typer.Option("--run-signalp6/--no-signalp6", help="Run SignalP6 preprocessing")
    ] = True,
    signalp6_extra: Annotated[
        str, typer.Option(help="Extra args for SignalP6")
    ] = "--organism euk",
    min_seq_id: Annotated[
        float, typer.Option(help="MMseqs2 clustering identity threshold")
    ] = 0.9,
) -> None:
    """Run the data preprocessing pipeline."""
    from toxfam.data.preprocessing import run_preprocessing_pipeline

    run_preprocessing_pipeline(
        run_signalp6=run_signalp6,
        signalp6_extra=signalp6_extra,
        min_seq_id=min_seq_id,
    )


# ---------- Step 2: toxfam embed ----------


@app.command()
def embed(
    input: Annotated[
        Path,
        typer.Option("-i", "--input", help="Input FASTA file", exists=True),
    ],
    output: Annotated[
        Path, typer.Option("-o", "--output", help="Output H5 file")
    ],
    model_dir: Annotated[
        Optional[Path], typer.Option(help="Cache directory for model")
    ] = None,
    model_name: Annotated[
        str, typer.Option(help="HuggingFace model name")
    ] = "Rostlab/prot_t5_xl_half_uniref50-enc",
    max_residues: Annotated[
        int, typer.Option(help="Max residues per batch")
    ] = 4000,
    max_batch: Annotated[
        int, typer.Option(help="Max sequences per batch")
    ] = 100,
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
    )


# ---------- Step 3a: toxfam taxonomy ----------


@app.command()
def taxonomy(
    input_csv: Annotated[
        Path,
        typer.Option(help="Input CSV with 'identifier' column", exists=True),
    ],
    output_csv: Annotated[
        Path, typer.Option(help="Output annotated CSV")
    ],
) -> None:
    """Annotate proteins with NCBI taxonomy lineage."""
    from toxfam.data.taxonomy import annotate_csv_with_taxonomy

    annotate_csv_with_taxonomy(str(input_csv), str(output_csv))


# ---------- Step 3b: toxfam taxonomy-vectors ----------


@app.command("taxonomy-vectors")
def taxonomy_vectors(
    tax_csv: Annotated[
        Path,
        typer.Option(help="Taxonomy-annotated CSV", exists=True),
    ],
    input_h5: Annotated[
        Path,
        typer.Option(help="Input H5 with protein embeddings", exists=True),
    ],
    output_h5: Annotated[
        Path, typer.Option(help="Output H5 for binary taxonomy vectors")
    ],
) -> None:
    """Generate binary taxonomy vectors from annotated CSV."""
    from toxfam.data.taxonomy import run_binary_taxonomy_pipeline

    run_binary_taxonomy_pipeline(
        tax_csv_path=str(tax_csv),
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
