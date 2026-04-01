"""Pre-compute homology-based inference (HBI) features for all sequences.

For each sequence, derives a 4-dimensional feature vector from MMseqs2 search:
1. best_hit_fident: fractional identity of best hit (0 if no hit)
2. best_hit_is_toxic: binary label of best hit's family (0 if no hit)
3. top5_frac_toxic: fraction of top-5 hits that are toxic (0 if no hits)
4. neg_log_evalue: normalized -log10(best hit evalue) in [0,1] (0 if no hit)

For training data: leave-one-out (exclude self-hits) to avoid data leakage.
For val/test data: standard search against full training set.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pymmseqs.commands import createdb, search
from rich.console import Console

from toxfam._paths import get_project_root
from toxfam.evaluation.metrics import to_binary_class

console = Console()


def _write_fasta(df: pd.DataFrame, path: Path) -> None:
    """Write FASTA from DataFrame — delegates to shared helper."""
    from toxfam.data._fasta import write_fasta

    write_fasta(df, path)


def _run_mmseqs_search(
    query_fasta: Path,
    target_fasta: Path,
    work_dir: Path,
) -> pd.DataFrame:
    """Run MMseqs2 search with sensible parameters."""
    query_db = createdb(str(query_fasta), str(work_dir / "query_db"))
    target_db = createdb(str(target_fasta), str(work_dir / "target_db"))

    search_res = search(
        query_db.to_path(),
        target_db.to_path(),
        str(work_dir / "search_res"),
        str(work_dir / "tmp"),
        s=7,
        e=10,
        min_seq_id=0.0,
        max_seqs=10,
    )
    return search_res.to_pandas()


def _compute_features_from_hits(
    hits: pd.DataFrame,
    all_ids: list[str],
    label_map: dict[str, int],
    *,
    exclude_self: bool = False,
) -> np.ndarray:
    """Derive 4-dim feature vectors from MMseqs2 hits.

    Returns array of shape (len(all_ids), 4).
    """
    if exclude_self:
        hits = hits[hits["query"] != hits["target"]].copy()

    features = np.zeros((len(all_ids), 4), dtype=np.float32)
    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}

    if hits.empty:
        return features

    # Add target labels
    hits["target_is_toxic"] = hits["target"].map(label_map).fillna(0).astype(int)

    # Best hit per query (lowest evalue, excluding self)
    best_hits = hits.loc[hits.groupby("query")["evalue"].idxmin()]

    # Top-5 hits per query
    sorted_hits = hits.sort_values(["query", "evalue"])
    top5 = sorted_hits.groupby("query").head(5)
    top5_stats = (
        top5.groupby("query")
        .agg(
            n_hits=("target_is_toxic", "count"),
            n_toxic=("target_is_toxic", "sum"),
        )
        .reset_index()
    )
    top5_stats["frac_toxic"] = top5_stats["n_toxic"] / top5_stats["n_hits"]

    # Compute normalized neg_log_evalue
    evalues = best_hits["evalue"].to_numpy().astype(float)
    neg_log_ev = -np.log10(np.clip(evalues, 1e-300, None))
    max_score = neg_log_ev.max() if len(neg_log_ev) > 0 else 1.0
    if max_score > 0:
        norm_neg_log_ev = neg_log_ev / max_score
    else:
        norm_neg_log_ev = np.zeros_like(neg_log_ev)

    # Fill feature vectors
    for i, (_, row) in enumerate(best_hits.iterrows()):
        qid = row["query"]
        if qid not in id_to_idx:
            continue
        idx = id_to_idx[qid]
        features[idx, 0] = row["fident"]
        features[idx, 1] = row["target_is_toxic"]
        features[idx, 3] = norm_neg_log_ev[i]

    for _, row in top5_stats.iterrows():
        qid = row["query"]
        if qid in id_to_idx:
            features[id_to_idx[qid], 2] = row["frac_toxic"]

    return features


def compute_hbi_features(
    training_csv: Path | None = None,
    output_h5: Path | None = None,
) -> Path:
    """Compute HBI features for all sequences (train/val/test).

    Train sequences use leave-one-out (exclude self-hits).
    Val/test sequences search against full training set.

    Returns path to output H5 file.
    """
    root = get_project_root()
    if training_csv is None:
        training_csv = root / "data" / "processed" / "training_data.csv"
    if output_h5 is None:
        output_h5 = root / "data" / "intermediate" / "hbi" / "hbi_features.h5"

    output_h5.parent.mkdir(parents=True, exist_ok=True)

    console.print("Loading data...")
    df = pd.read_csv(training_csv)
    train_df = df[df["Split"] == "train"].copy()
    val_df = df[df["Split"] == "val"].copy()
    test_df = df[df["Split"] == "test"].copy()

    console.print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Build binary label map from family labels
    label_map = {}
    for _, row in df.iterrows():
        is_tox = 0 if to_binary_class(row["Protein families"]) == "nontoxin" else 1
        label_map[row["identifier"]] = is_tox

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write FASTA files
        train_fasta = tmpdir / "train.fasta"
        _write_fasta(train_df, train_fasta)

        # --- Train: search train vs train (leave-one-out) ---
        console.print("\nComputing train HBI features (leave-one-out)...")
        train_work = tmpdir / "train_work"
        train_work.mkdir()
        train_hits = _run_mmseqs_search(train_fasta, train_fasta, train_work)
        train_features = _compute_features_from_hits(
            train_hits,
            train_df["identifier"].tolist(),
            label_map,
            exclude_self=True,
        )
        console.print(f"  Train queries with hits: {(train_features[:, 0] > 0).sum()}/{len(train_df)}")

        # --- Val: search val vs train ---
        console.print("Computing val HBI features...")
        if len(val_df) > 0:
            val_fasta = tmpdir / "val.fasta"
            _write_fasta(val_df, val_fasta)
            val_work = tmpdir / "val_work"
            val_work.mkdir()
            val_hits = _run_mmseqs_search(val_fasta, train_fasta, val_work)
            val_features = _compute_features_from_hits(
                val_hits,
                val_df["identifier"].tolist(),
                label_map,
                exclude_self=False,
            )
            console.print(f"  Val queries with hits: {(val_features[:, 0] > 0).sum()}/{len(val_df)}")
        else:
            val_features = np.zeros((0, 4), dtype=np.float32)

        # --- Test: search test vs train ---
        console.print("Computing test HBI features...")
        test_fasta = tmpdir / "test.fasta"
        _write_fasta(test_df, test_fasta)
        test_work = tmpdir / "test_work"
        test_work.mkdir()
        test_hits = _run_mmseqs_search(test_fasta, train_fasta, test_work)
        test_features = _compute_features_from_hits(
            test_hits,
            test_df["identifier"].tolist(),
            label_map,
            exclude_self=False,
        )
        console.print(f"  Test queries with hits: {(test_features[:, 0] > 0).sum()}/{len(test_df)}")

    # Save to H5
    console.print(f"\nSaving HBI features to {output_h5}...")
    with h5py.File(str(output_h5), "w") as f:
        for split_name, split_df, features in [
            ("train", train_df, train_features),
            ("val", val_df, val_features),
            ("test", test_df, test_features),
        ]:
            for i, ident in enumerate(split_df["identifier"]):
                f.create_dataset(ident, data=features[i])

    total = len(train_df) + len(val_df) + len(test_df)
    console.print(f"Saved {total} feature vectors (shape: 4) to {output_h5}")

    return output_h5
