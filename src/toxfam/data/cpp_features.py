"""CPP (Comparative Physicochemical Profiling) feature generation.

Uses AAanalysis to generate global tox-vs-nontox physicochemical features
that complement ProtT5 embeddings. Produces a fixed-width (~100 dim) vector
per protein capturing discriminative physicochemical signatures.

Requires AAanalysis (``pip install aaanalysis`` or installed via uv).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rich.console import Console

from toxfam._paths import intermediate_dir

console = Console()


def run_cpp_pipeline(
    training_csv: str | Path,
    output_h5: str | Path | None = None,
    *,
    n_filter: int = 100,
    max_overlap: float = 0.5,
    max_cor: float = 0.5,
    subsample_ratio: float = 1.0,
) -> Path:
    """Generate global CPP features (tox vs nontox) and store in HDF5.

    Args:
        training_csv: Path to training_data.csv (must have
            'identifier', 'Sequence', 'is_toxic' columns).
        output_h5: Where to write the H5 file. Defaults to
            data/intermediate/cpp/cpp_features.h5.
        n_filter: Number of CPP features to select (default 100).
        max_overlap: Max positional overlap between features.
        max_cor: Max Pearson correlation between feature scales.
        subsample_ratio: Ratio of nontox-to-toxic for CPP feature selection
            (default 1.0 = match toxic count). Only affects feature selection,
            not the final feature matrix which covers all sequences.

    Returns:
        Path to the output H5 file.
    """
    import aaanalysis as aa

    if output_h5 is None:
        output_h5 = intermediate_dir() / "cpp" / "cpp_features.h5"
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    # -- Load data --
    console.print("[bold]CPP Feature Generation[/]")
    df = pd.read_csv(training_csv)
    console.print(f"   Loaded {len(df)} proteins from {training_csv}")

    # Binary labels: toxic=1, nontox=0
    if "is_toxic" in df.columns:
        labels = df["is_toxic"].astype(int).tolist()
    elif "Protein families" in df.columns:
        from toxfam.evaluation.metrics import to_binary_class
        labels = [0 if to_binary_class(f) == "nontoxin" else 1 for f in df["Protein families"]]
    else:
        raise ValueError("training_csv must have 'is_toxic' or 'Protein families' column")

    n_tox = sum(labels)
    n_nontox = len(labels) - n_tox
    console.print(f"   Labels: {n_tox} toxic (1), {n_nontox} non-toxic (0)")

    # -- Prepare sequences for CPP --
    # Entire sequence as TMD (no JMD)
    df_seq = pd.DataFrame({
        "entry": df["identifier"].values,
        "sequence": df["Sequence"].values,
        "tmd_start": 1,
        "tmd_stop": df["Sequence"].str.len().values,
        "label": labels,
    })

    # Clean sequences: replace non-standard amino acids
    df_seq["sequence"] = (
        df_seq["sequence"]
        .str.replace("U", "C", regex=False)
        .str.replace("Z", "E", regex=False)
        .str.replace("O", "K", regex=False)
        .str.replace("B", "D", regex=False)
        .str.replace("X", "A", regex=False)
    )
    # Recalculate tmd_stop after cleaning
    df_seq["tmd_stop"] = df_seq["sequence"].str.len()

    # Drop entries with empty sequences
    df_seq = df_seq[df_seq["sequence"].str.len() > 0].reset_index(drop=True)

    # AAanalysis CPP splits TMD into N/C halves; default split_kws needs
    # sub-parts >= 15 AA, so full sequences must be >= 32 AA.
    # Short sequences get zero vectors (model falls back on ProtT5+HBI).
    MIN_SEQ_LEN = 32
    short_mask = df_seq["sequence"].str.len() < MIN_SEQ_LEN
    short_entries = df_seq.loc[short_mask, "entry"].tolist()
    df_long = df_seq[~short_mask].reset_index(drop=True)
    if short_entries:
        console.print(
            f"   [yellow]{len(short_entries)} sequences < {MIN_SEQ_LEN} AA "
            f"excluded from CPP (will get zero vectors)[/]"
        )

    median_len = int(df_long["sequence"].str.len().median())
    console.print(f"   Median sequence length: {median_len} ({len(df_long)} sequences)")

    # -- Subsample for feature selection --
    # CPP creates ~580K candidate features and builds a (n_samples × n_features)
    # matrix internally. With 63K sequences this exceeds 100+ GB RAM.
    # Subsampling both groups preserves discriminative signal while keeping
    # memory manageable. 1K per class is statistically sufficient for
    # identifying discriminative physicochemical features.
    MAX_PER_CLASS = 1000
    tox_mask = df_long["label"] == 1
    n_tox_long = int(tox_mask.sum())
    n_nontox_available = int((~tox_mask).sum())
    rng = np.random.default_rng(seed=42)

    # Subsample toxic if needed
    tox_idx = df_long.index[tox_mask].to_numpy()
    if n_tox_long > MAX_PER_CLASS:
        tox_idx = rng.choice(tox_idx, size=MAX_PER_CLASS, replace=False)

    # Subsample nontox: match toxic count (capped at MAX_PER_CLASS)
    n_nontox_target = min(len(tox_idx), int(len(tox_idx) * subsample_ratio))
    nontox_idx = df_long.index[~tox_mask].to_numpy()
    if n_nontox_target < n_nontox_available:
        nontox_idx = rng.choice(nontox_idx, size=n_nontox_target, replace=False)

    keep_idx = np.sort(np.concatenate([tox_idx, nontox_idx]))
    df_sub = df_long.loc[keep_idx].reset_index(drop=True)
    console.print(
        f"   [yellow]Subsampled to {int((df_sub['label'] == 1).sum())} tox + "
        f"{int((df_sub['label'] == 0).sum())} nontox = {len(df_sub)} total "
        f"for feature selection[/]"
    )

    # -- Generate parts for subsampled data (feature selection) --
    console.print("   Generating sequence parts (feature selection subset) ...")
    sf = aa.SequenceFeature()
    df_parts_sub = sf.get_df_parts(
        df_seq=df_sub,
        jmd_n_len=0,
        jmd_c_len=0,
    )

    # -- Run CPP on subsample --
    # AAanalysis CPP allocates large intermediates (scale × part × position × sample).
    # vectorized=False + n_batches processes scales in chunks to limit peak memory.
    console.print(f"   Running CPP (n_filter={n_filter}) ...")
    cpp = aa.CPP(df_parts=df_parts_sub, verbose=True)
    df_feat = cpp.run(
        labels=df_sub["label"].tolist(),
        label_test=1,
        label_ref=0,
        n_filter=n_filter,
        tmd_len=median_len,
        jmd_n_len=0,
        jmd_c_len=0,
        max_overlap=max_overlap,
        max_cor=max_cor,
        vectorized=False,
        n_jobs=1,
        n_batches=5,
    )
    console.print(f"   Selected {len(df_feat)} features")

    # -- Generate parts for ALL long sequences (feature matrix) --
    console.print("   Generating sequence parts (all sequences) ...")
    df_parts_all = sf.get_df_parts(
        df_seq=df_long,
        jmd_n_len=0,
        jmd_c_len=0,
    )

    # -- Compute feature matrix for ALL long sequences --
    console.print("   Computing feature matrix ...")
    features = df_feat["feature"].tolist()
    X = sf.feature_matrix(
        features=features,
        df_parts=df_parts_all,
        n_jobs=1,
    )
    console.print(f"   Feature matrix shape: {X.shape}")

    # -- Write to H5 (long sequences get real vectors, short get zeros) --
    n_features = X.shape[1]
    long_entries = df_long["entry"].tolist()
    with h5py.File(output_h5, "w") as h5:
        for i, entry in enumerate(long_entries):
            h5.create_dataset(entry, data=X[i].astype(np.float32))
        for entry in short_entries:
            h5.create_dataset(entry, data=np.zeros(n_features, dtype=np.float32))

    total_written = len(long_entries) + len(short_entries)
    console.print(f"   Wrote {total_written} entries to {output_h5}")

    # -- Save feature metadata --
    meta_path = output_h5.parent / "cpp_feature_names.json"
    feature_meta = []
    for _, row in df_feat.iterrows():
        feature_meta.append({
            "feature": row["feature"],
            "category": row.get("category", ""),
            "scale_name": row.get("scale_name", ""),
            "abs_auc": float(row.get("abs_auc", 0)),
            "abs_mean_dif": float(row.get("abs_mean_dif", 0)),
        })
    with open(meta_path, "w") as f:
        json.dump(feature_meta, f, indent=2)
    console.print(f"   Feature metadata saved to {meta_path}")

    console.print("[bold green]CPP feature generation complete.[/]")
    return output_h5
