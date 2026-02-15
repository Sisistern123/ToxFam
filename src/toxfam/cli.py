"""Unified CLI for ToxFam."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="toxfam",
    help="Animal toxin protein family classification using MLP on ProtT5 embeddings.",
    no_args_is_help=True,
)


# ---------- toxfam train ----------


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


# ---------- toxfam preprocess ----------


@app.command()
def preprocess(
    run_signalp6: Annotated[
        bool, typer.Option("--run-signalp6", help="Run SignalP6 preprocessing")
    ] = False,
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


# ---------- toxfam embed ----------


@app.command()
def embed(
    input: Annotated[
        Path,
        typer.Option("-i", "--input", help="Input FASTA file", exists=True),
    ],
    output: Annotated[
        Path, typer.Option("-o", "--output", help="Output H5 file")
    ],
    per_protein: Annotated[
        bool, typer.Option("--per-protein", help="Mean-pooled per-protein embeddings")
    ] = False,
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
    """Generate ProtT5 embeddings from a FASTA file."""
    from toxfam.data.embedding import generate_embeddings

    generate_embeddings(
        input_fasta=input,
        output_h5=output,
        per_protein=per_protein,
        model_dir=str(model_dir) if model_dir else None,
        model_name=model_name,
        max_residues=max_residues,
        max_batch=max_batch,
    )


# ---------- toxfam taxonomy ----------


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


# ---------- toxfam taxonomy-vectors ----------


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
        Path, typer.Option(help="Output H5 for taxonomy vectors")
    ],
    mode: Annotated[
        str, typer.Option(help="Vector mode: 'binary' or 'numeric'")
    ] = "binary",
) -> None:
    """Generate taxonomy vectors from annotated CSV."""
    from toxfam.data.taxonomy import (
        run_binary_taxonomy_pipeline,
        run_numeric_taxonomy_pipeline,
    )

    if mode == "binary":
        run_binary_taxonomy_pipeline(
            tax_csv_path=str(tax_csv),
            input_h5_path=str(input_h5),
            output_h5_path=str(output_h5),
        )
    elif mode == "numeric":
        run_numeric_taxonomy_pipeline(
            tax_csv_path=str(tax_csv),
            input_h5_path=str(input_h5),
            output_h5_path=str(output_h5),
        )
    else:
        typer.echo(f"Unknown mode: {mode}. Use 'binary' or 'numeric'.", err=True)
        raise typer.Exit(code=1)


# ---------- toxfam eval-test ----------


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


# ---------- toxfam eval-nonmetazoan ----------


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


# ---------- toxfam eval-unreviewed ----------


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
