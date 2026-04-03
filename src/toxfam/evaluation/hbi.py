"""Homology-Based Inference (HBI) via MMseqs2 sequence similarity search.

Implements 1-NN label transfer: for each query protein, search against a
database of labeled training proteins, select the best hit by lowest E-value,
and transfer its family label to the query.

References
----------
Heinzinger et al. (2022) "Contrastive learning on protein embeddings
    enlightens midnight zone at the speed of light" NAR Genomics and
    Bioinformatics 4(1):lqac043.

Hamp et al. (2013) "Homology-based inference sets the bar high for protein
    function prediction" BMC Bioinformatics 14(Suppl 3):S7.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pymmseqs.commands import createdb, search
from rich.console import Console

from toxfam.data._fasta import parse_fasta

console = Console()

DEFAULT_SENSITIVITY: float = 9.0
DEFAULT_EVALUE: float = float("inf")
DEFAULT_COV_MODE: int = 0
DEFAULT_MIN_SEQ_ID: float = 0.0
DEFAULT_MAX_SEQS: int = 100_000
NO_HIT_LABEL: str = "no hit"


@dataclass
class HBIResult:
    """Results from an HBI search."""

    predictions: pd.DataFrame
    n_queries: int
    n_with_hits: int

    @property
    def n_no_hits(self) -> int:
        return self.n_queries - self.n_with_hits

    @property
    def coverage(self) -> float:
        """Fraction of queries that received at least one hit."""
        return self.n_with_hits / self.n_queries if self.n_queries > 0 else 0.0


def write_fasta_from_df(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    id_column: str = "identifier",
    seq_column: str = "Sequence",
) -> None:
    """Write a FASTA file from a DataFrame."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row[id_column]}\n{row[seq_column]}\n")


def run_hbi_search(
    query_fasta: str | Path,
    target_fasta: str | Path,
    target_labels_df: pd.DataFrame,
    work_dir: str | Path,
    *,
    id_column: str = "identifier",
    label_column: str = "Protein families",
    sensitivity: float = DEFAULT_SENSITIVITY,
    evalue: float = DEFAULT_EVALUE,
    cov_mode: int = DEFAULT_COV_MODE,
    min_seq_id: float = DEFAULT_MIN_SEQ_ID,
    max_seqs: int = DEFAULT_MAX_SEQS,
) -> HBIResult:
    """Run MMseqs2 search and transfer labels from best hits.

    Parameters
    ----------
    query_fasta : Path
        FASTA file of query sequences to classify.
    target_fasta : Path
        FASTA file of labeled reference sequences (training set).
    target_labels_df : DataFrame
        Must contain columns ``id_column`` and ``label_column`` mapping each
        target sequence to its family label.
    work_dir : Path
        Directory for MMseqs2 intermediate files.

    Returns
    -------
    HBIResult
        Predictions DataFrame with columns:
        ``identifier``, ``hbi_prediction``, ``hbi_confidence``, ``evalue``.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate all query IDs for coverage tracking
    all_query_ids = [rec.id for rec in parse_fasta(query_fasta)]
    n_queries = len(all_query_ids)

    console.print(f"   Creating MMseqs2 databases ({n_queries} queries)...")
    query_db = createdb(str(query_fasta), str(work_dir / "queryDB"))
    target_db = createdb(str(target_fasta), str(work_dir / "targetDB"))

    with console.status("[bold]Running MMseqs2 sequence search..."):
        search_res = search(
            query_db.to_path(),
            target_db.to_path(),
            str(work_dir / "resultDB"),
            str(work_dir / "search_tmp"),
            s=sensitivity,
            e=evalue,
            cov_mode=cov_mode,
            min_seq_id=min_seq_id,
            max_seqs=max_seqs,
        )

    df_search = search_res.to_pandas()

    # Build target label map
    label_map = dict(zip(target_labels_df[id_column], target_labels_df[label_column]))

    if df_search.empty:
        console.print("   [yellow]No search hits found.[/]")
        predictions = pd.DataFrame(
            {
                "identifier": all_query_ids,
                "hbi_prediction": NO_HIT_LABEL,
                "hbi_confidence": 0.0,
                "evalue": np.nan,
            }
        )
        return HBIResult(predictions=predictions, n_queries=n_queries, n_with_hits=0)

    # Best hit per query: lowest E-value
    best_hits = df_search.loc[
        df_search.groupby("query")["evalue"].idxmin()
    ].reset_index(drop=True)

    best_hits["hbi_prediction"] = best_hits["target"].map(label_map)
    best_hits["hbi_confidence"] = best_hits["fident"]

    hit_df = best_hits[["query", "hbi_prediction", "hbi_confidence", "evalue"]].rename(
        columns={"query": "identifier"}
    )

    # Merge with all query IDs to identify no-hit cases
    all_ids_df = pd.DataFrame({"identifier": all_query_ids})
    predictions = all_ids_df.merge(hit_df, on="identifier", how="left")
    predictions["hbi_prediction"] = predictions["hbi_prediction"].fillna(NO_HIT_LABEL)
    predictions["hbi_confidence"] = predictions["hbi_confidence"].fillna(0.0)

    n_with_hits = int((predictions["hbi_prediction"] != NO_HIT_LABEL).sum())
    console.print(f"   Extracted best hits for {n_with_hits}/{n_queries} queries")

    return HBIResult(
        predictions=predictions, n_queries=n_queries, n_with_hits=n_with_hits
    )
