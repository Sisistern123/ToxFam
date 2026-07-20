"""Pure inference (`toxfam predict`).

Predicts toxin family + binary toxicity for arbitrary proteins, given a TSV of
``identifier`` (+ ``Organism (ID)`` for combined models, + ``Sequence`` if
embeddings must be generated) and one or two trained model directories.

No ground-truth labels, no metrics, no ``benchmark/`` writes — just predictions.

Three usages, dispatched on the primary model's architecture:

1. **Combined only** (``--model-dir`` is a ``MultiInputMLP``): proteins without an
   organism ID are *excluded*; the rest get combined-model predictions. One TSV.
2. **Combined + standard** (``--model-dir`` combined, ``--standard-model-dir``
   standard): proteins *with* an organism ID go to the combined model, those
   *without* go to the standard model. Two disjoint TSVs.
3. **Standard only** (``--model-dir`` is a ``ModularMLP``): organism IDs ignored,
   all proteins predicted by the standard model. One TSV.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import h5py
import pandas as pd
from rich.console import Console

from toxfam.data.normalization import ORGANISM_COL, ensure_identifier_column
from toxfam.model.inference import run_topk_inference
from toxfam.model.model_config import ModelConfig

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_input(input_tsv: str | Path) -> pd.DataFrame:
    """Read the prediction input TSV and normalize the identifier column."""
    df = pd.read_csv(input_tsv, sep="\t")
    df = ensure_identifier_column(df)
    if "identifier" not in df.columns:
        raise ValueError(
            "Input TSV must contain an 'identifier' (or 'Entry') column. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def _is_split_dataset(spec: str | Path) -> bool:
    """True when ``spec`` names a dataset carved out of the train/val/test split.

    Predicting on ``test_set``/``val_set`` reads the split, so the checkpoint must be
    pinned to it. Predicting on a user-supplied TSV, or on an external dataset such as
    ``non_metazoan``, involves no split and needs no manifest -- which is what lets the
    Colab notebook run against a pip-installed package with no repo checkout.
    """
    from toxfam.data.registry import DATASETS

    name = str(spec)
    if Path(spec).exists() or name not in DATASETS:
        return False
    return DATASETS[name].get("source") == "training_data"


def _resolve_input(spec: str | Path) -> tuple[pd.DataFrame, Path | None]:
    """Resolve the input spec to (DataFrame, default embeddings H5).

    ``spec`` may be a path to a TSV, or a registered dataset name (the same
    names accepted by ``toxfam eval``, e.g. ``non_metazoan``). For a registered
    dataset, the TSV and a default embeddings H5 are looked up from the registry;
    unlike eval, rows are *not* dropped for missing family labels.

    Resolves against the light ``toxfam.data.registry`` (not ``evaluation.runner``)
    so prediction does not import the evaluation/plotting stack.
    """
    from toxfam._paths import evaluation_data_dir
    from toxfam.data.registry import (
        DATASETS,
        list_datasets,
        load_dataset,
        resolve_embeddings_h5,
    )

    spec_path = Path(spec)
    name = str(spec)

    if name in DATASETS and not spec_path.exists():
        cfg = DATASETS[name]
        default_h5 = resolve_embeddings_h5(name)
        default_h5 = default_h5 if default_h5.exists() else None

        if cfg["source"] == "evaluation":
            tsv = evaluation_data_dir() / name / cfg["tsv"]
            if not tsv.exists():
                raise FileNotFoundError(
                    f"{tsv} not found. Run 'toxfam download-data' first."
                )
            console.print(f"   Resolved dataset '{name}' -> {tsv.name}")
            return _read_input(tsv), default_h5

        # training_data split (test_set / val_set) — training_data.csv is always
        # 'identifier'-keyed (renamed at preprocessing), so no rename is needed.
        return load_dataset(name), default_h5

    if not spec_path.exists():
        raise FileNotFoundError(
            f"Input '{spec}' is neither an existing file nor a known dataset. "
            f"Known datasets: {list_datasets()}"
        )
    return _read_input(spec_path), None


def _organism_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows that have a usable NCBI taxon ID."""
    if ORGANISM_COL not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[ORGANISM_COL], errors="coerce").notna()


def _read_optimized_threshold(model_dir: Path) -> float:
    """Read the deployed binary threshold; fall back to 0.5 if unavailable.

    When the checkpoint ships a deployed Platt calibrator
    (``models/binary_calibrator.json``), its ``threshold`` lives in calibrated
    score space and must be used against the calibrated P(toxic) that
    ``run_topk_inference`` now emits. Older checkpoints fall back to the raw
    Youden threshold in ``metrics/binary_metrics.json``.
    """
    from toxfam.model.inference import _load_binary_calibration

    binary_cal = _load_binary_calibration(model_dir)
    if binary_cal is not None:
        return binary_cal.threshold

    # Back-compat: older checkpoints (no calibrator) store a raw-space Youden
    # threshold here, consistent with the raw P(toxic) emitted in that case.
    path = model_dir / "metrics" / "binary_metrics.json"
    if not path.exists():
        return 0.5
    try:
        return float(json.loads(path.read_text())["optimized_threshold"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.5


def _ensure_embeddings(
    df: pd.DataFrame,
    embeddings_h5: str | Path | None,
    work_dir: Path,
    *,
    max_residues: int,
    max_batch: int,
) -> Path:
    """Return an H5 that holds an embedding for every input identifier.

    Reuses ``embeddings_h5`` as-is when it already covers all identifiers;
    otherwise embeds the missing sequences (from the ``Sequence`` column) via the
    standard ProtT5 pipeline, non-destructively (copies the supplied H5 first).
    """
    from toxfam.data._fasta import write_fasta
    from toxfam.data.embedding import generate_embeddings

    identifiers = set(df["identifier"])

    if embeddings_h5 is not None:
        with h5py.File(embeddings_h5, "r") as f:
            existing = set(f.keys())
        missing = identifiers - existing
        if not missing:
            console.print(f"   Using precomputed embeddings: {Path(embeddings_h5).name}")
            return Path(embeddings_h5)
        # Non-destructive: copy, then append only the missing ones.
        work_h5 = work_dir / "embeddings.h5"
        shutil.copy(embeddings_h5, work_h5)
        console.print(
            f"   {len(missing)} of {len(identifiers)} sequences missing embeddings; "
            "generating them"
        )
    else:
        work_h5 = work_dir / "embeddings.h5"
        missing = identifiers
        console.print(f"   No embeddings supplied; generating for {len(missing)} sequences")

    if "Sequence" not in df.columns:
        raise ValueError(
            f"{len(missing)} protein(s) have no precomputed embedding and the input "
            "TSV has no 'Sequence' column to embed from. Provide --embeddings that "
            "covers every identifier, or add a 'Sequence' column."
        )

    missing_df = df[df["identifier"].isin(missing)]
    if missing_df["Sequence"].isna().any():
        raise ValueError("Some rows needing embeddings have an empty 'Sequence'.")

    fasta = work_dir / "to_embed.fasta"
    write_fasta(missing_df, fasta)
    generate_embeddings(
        fasta, work_h5, force=False, max_residues=max_residues, max_batch=max_batch
    )
    return work_h5


def _report_taxonomy_coverage(
    df_pool: pd.DataFrame, tax_h5: Path, output_path: Path
) -> None:
    """Warn about organism IDs that produced no taxonomy signal.

    Two distinct reasons a combined-model protein can end up with an all-zero
    taxonomy vector (and therefore a weakened prediction):

    * **Unresolvable taxon ID** — taxopy could not look the ID up at all (it is
      obsolete/merged/deleted in NCBI taxonomy, malformed, or the local taxopy
      database is stale).
    * **Organism not among the model's 50 taxa** — the lineage resolved fine, but
      none of the 50 predefined ``TAXA`` appear in it, so the multi-hot vector is
      all zeros (the model simply has no taxonomy feature for this organism).

    Prints a summary and writes a ``*_unresolved_organisms.tsv`` sidecar listing
    the affected proteins and the reason.
    """
    from toxfam.data.taxonomy import _resolve_lineages

    zero_ids = []
    with h5py.File(tax_h5, "r") as f:
        for ident in df_pool["identifier"]:
            if not f[str(ident)][:].any():
                zero_ids.append(ident)
    if not zero_ids:
        return

    sub = df_pool[df_pool["identifier"].isin(zero_ids)][
        ["identifier", ORGANISM_COL]
    ].copy()
    sub["_taxid"] = pd.to_numeric(sub[ORGANISM_COL], errors="coerce")
    taxids = sub["_taxid"].dropna().astype(int).unique().tolist()
    resolved = _resolve_lineages(taxids) if taxids else {}

    def _reason(taxid: float) -> str:
        if pd.isna(taxid):
            return "no organism id"
        names = resolved.get(int(taxid), (None, set()))[1]
        return (
            "unresolvable taxon id"
            if not names
            else "organism not among model's 50 taxa"
        )

    sub["reason"] = sub["_taxid"].map(_reason)
    sub = sub.drop(columns="_taxid")

    console.print(
        f"   [yellow]{len(sub)} protein(s) got no taxonomy signal "
        "(zero taxonomy vector):[/]"
    )
    for reason, count in sub["reason"].value_counts().items():
        console.print(f"     {count}: {reason}")

    sidecar = _suffixed(output_path, "unresolved_organisms")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(sidecar, sep="\t", index=False)
    console.print(f"   Details written to [cyan]{sidecar}[/]")


def _predict_pool(
    df_pool: pd.DataFrame,
    embeddings_h5: Path,
    model_dir: Path,
    work_dir: Path,
    output_path: Path,
    *,
    is_combined: bool,
    top_k: int,
    toxicity_only: bool,
) -> pd.DataFrame:
    """Run inference on one pool and attach the binary toxicity call."""
    from toxfam.data.taxonomy import build_taxonomy_h5

    tax_h5 = build_taxonomy_h5(df_pool, work_dir) if is_combined else None
    if tax_h5 is not None:
        _report_taxonomy_coverage(df_pool, tax_h5, output_path)
    preds = run_topk_inference(
        df_pool, embeddings_h5, model_dir,
        tax_h5_path=tax_h5, top_k=top_k, binary_only=toxicity_only,
    )
    threshold = _read_optimized_threshold(model_dir)
    preds["predicted_toxic"] = preds["p_toxic"] >= threshold
    return preds


def _write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    console.print(f"   Wrote {len(df)} predictions to [cyan]{path}[/]")


def _suffixed(path: Path, tag: str) -> Path:
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_prediction(
    input_tsv: str | Path,
    model_dir: str | Path,
    *,
    standard_model_dir: str | Path | None = None,
    embeddings_h5: str | Path | None = None,
    output: str | Path = "predictions.tsv",
    top_k: int = 3,
    toxicity_only: bool = False,
    max_residues: int = 4000,
    max_batch: int = 100,
) -> list[Path]:
    """Predict toxin family + toxicity for the proteins in ``input_tsv``.

    Returns the list of written TSV paths.
    """
    model_dir = Path(model_dir)
    output = Path(output)

    # test_set / val_set come from the split, so the checkpoint must be pinned to it.
    # Any other input carries no split and needs no manifest.
    if _is_split_dataset(input_tsv):
        from toxfam.data.split_manifest import verify_split_provenance

        verify_split_provenance(model_dir)
        if standard_model_dir is not None:
            verify_split_provenance(standard_model_dir)

    df, default_h5 = _resolve_input(input_tsv)
    if embeddings_h5 is None and default_h5 is not None:
        embeddings_h5 = default_h5
    console.print(f"\n[bold]Predicting for {len(df)} proteins[/]")

    primary_cfg = ModelConfig.load(model_dir / "model_config.json")
    primary_combined = primary_cfg.architecture == "MultiInputMLP"

    work_dir = Path(tempfile.mkdtemp(prefix="toxfam_predict_"))
    try:
        # --- Mode 3: standard primary — predict everything, ignore organism IDs.
        if not primary_combined:
            if standard_model_dir is not None:
                console.print(
                    "   [yellow]--standard-model-dir is ignored: primary model is "
                    "already a standard (embeddings-only) model[/]"
                )
            emb = _ensure_embeddings(
                df, embeddings_h5, work_dir,
                max_residues=max_residues, max_batch=max_batch,
            )
            preds = _predict_pool(
                df, emb, model_dir, work_dir, output,
                is_combined=False, top_k=top_k, toxicity_only=toxicity_only,
            )
            _write_tsv(preds, output)
            return [output]

        # --- Primary is combined: split by organism-ID availability.
        mask = _organism_mask(df)
        pool_tax = df[mask].copy()
        pool_notax = df[~mask].copy()

        # --- Mode 1: combined only — exclude proteins without an organism ID.
        if standard_model_dir is None:
            if len(pool_notax):
                console.print(
                    f"   [yellow]Excluding {len(pool_notax)} protein(s) without an "
                    "organism ID (combined model requires taxonomy). Supply "
                    "--standard-model-dir to predict them with a standard model.[/]"
                )
            if pool_tax.empty:
                raise ValueError(
                    "No proteins have an organism ID, so the combined model cannot "
                    "predict any of them. Re-run with a standard model via "
                    "--model-dir, or supply organism IDs."
                )
            emb = _ensure_embeddings(
                pool_tax, embeddings_h5, work_dir,
                max_residues=max_residues, max_batch=max_batch,
            )
            preds = _predict_pool(
                pool_tax, emb, model_dir, work_dir, output,
                is_combined=True, top_k=top_k, toxicity_only=toxicity_only,
            )
            _write_tsv(preds, output)
            return [output]

        # --- Mode 2: combined + standard — route each pool, write two TSVs.
        standard_model_dir = Path(standard_model_dir)
        std_cfg = ModelConfig.load(standard_model_dir / "model_config.json")
        if std_cfg.architecture != "ModularMLP":
            raise ValueError(
                f"--standard-model-dir must be a standard (ModularMLP) model, but "
                f"'{standard_model_dir.name}' is a {std_cfg.architecture}."
            )

        emb = _ensure_embeddings(
            df, embeddings_h5, work_dir,
            max_residues=max_residues, max_batch=max_batch,
        )
        written: list[Path] = []

        out_c = _suffixed(output, "combined")
        out_s = _suffixed(output, "standard")

        if len(pool_tax):
            console.print(f"\n[bold]Combined pool[/]: {len(pool_tax)} proteins")
            preds_c = _predict_pool(
                pool_tax, emb, model_dir, work_dir, out_c,
                is_combined=True, top_k=top_k, toxicity_only=toxicity_only,
            )
            _write_tsv(preds_c, out_c)
            written.append(out_c)
        else:
            console.print("   [yellow]No proteins with an organism ID (combined pool empty)[/]")

        if len(pool_notax):
            console.print(f"\n[bold]Standard pool[/]: {len(pool_notax)} proteins")
            preds_s = _predict_pool(
                pool_notax, emb, standard_model_dir, work_dir, out_s,
                is_combined=False, top_k=top_k, toxicity_only=toxicity_only,
            )
            _write_tsv(preds_s, out_s)
            written.append(out_s)
        else:
            console.print("   [yellow]No proteins without an organism ID (standard pool empty)[/]")

        return written
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
