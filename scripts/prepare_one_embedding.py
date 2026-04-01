"""Prepare One Embedding data for ToxFam experiments.

Generates codec-processed protein vectors from ProteEmbedExplorations'
per-residue ProtT5 embeddings and creates balanced training subsets.

Usage:
    uv run python scripts/prepare_one_embedding.py

Prerequisites:
    Per-residue embeddings at:
    ../ProteEmbedExplorations/data/external_validation/toxfam/residue_embeddings_prot_t5.h5
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OE_ROOT = PROJECT_ROOT.parent / "ProteEmbedExplorations"

# Source files from ProteEmbedExplorations
PERRES_H5 = OE_ROOT / "data/external_validation/toxfam/residue_embeddings_prot_t5.h5"
CODEC_H5 = OE_ROOT / "data/protspace_bundles/toxprot_codec_vecs.h5"
BASELINE_H5 = OE_ROOT / "data/external_validation/toxfam/embeddings_baseline_1024.h5"

# Output paths
OUT_DIR = PROJECT_ROOT / "data/processed/one_embedding"
SUBSET_CSV = OUT_DIR / "training_data_balanced.csv"
OE_CODEC_H5 = OUT_DIR / "codec_2048.h5"
OE_BASELINE_H5 = OUT_DIR / "baseline_1024.h5"


def create_balanced_subset() -> pd.DataFrame:
    """Create a balanced 1:1 toxin:nontoxin subset using only proteins
    that already have per-residue embeddings."""
    from toxfam.evaluation.metrics import to_binary_class

    df = pd.read_csv(PROJECT_ROOT / "data/processed/training_data.csv")

    with h5py.File(str(PERRES_H5), "r") as f:
        oe_ids = set(f.keys())

    subset = df[df["identifier"].isin(oe_ids)].copy()
    subset["binary_check"] = subset["Protein families"].apply(to_binary_class)

    n_tox = (subset["binary_check"] == "toxin").sum()
    n_nontox = (subset["binary_check"] == "nontoxin").sum()
    subset.drop(columns=["binary_check"], inplace=True)

    print(f"Balanced subset: {n_tox} toxin + {n_nontox} nontoxin = {len(subset)}")
    print(f"Splits: {subset['Split'].value_counts().to_dict()}")
    return subset


def copy_vectors(src_h5_path: Path, dst_h5_path: Path, protein_ids: set) -> int:
    """Copy vectors from source H5 to destination, keeping only specified IDs."""
    n = 0
    with h5py.File(str(src_h5_path), "r") as src:
        with h5py.File(str(dst_h5_path), "w") as dst:
            for pid in protein_ids:
                if pid in src:
                    dst.create_dataset(pid, data=src[pid][:].astype(np.float32))
                    n += 1
    return n


def apply_codec_to_per_residue(protein_ids: set, output_h5: Path, d_out: int = 768, dct_k: int = 4) -> int:
    """Apply One Embedding codec (centering + RP + DCT) to per-residue embeddings.

    This is the proper One Embedding pipeline:
    1. Load per-residue (L, 1024) embeddings
    2. Center (subtract corpus mean)
    3. Random projection 1024 → d_out
    4. DCT K coefficients → (dct_k * d_out,) protein vector

    Returns number of proteins processed.
    """
    from scipy.fft import dct

    # Load all per-residue embeddings
    print(f"Loading per-residue embeddings for {len(protein_ids)} proteins...")
    embeddings = {}
    with h5py.File(str(PERRES_H5), "r") as f:
        for pid in protein_ids:
            if pid in f:
                embeddings[pid] = f[pid][:].astype(np.float32)

    if not embeddings:
        print("No embeddings found!")
        return 0

    # Step 1: Compute corpus mean from a sample of residues
    print("Computing corpus mean...")
    all_residues = []
    rng = np.random.default_rng(42)
    for pid, emb in embeddings.items():
        # Sample up to 10 residues per protein to keep memory reasonable
        n = min(10, emb.shape[0])
        idx = rng.choice(emb.shape[0], n, replace=False)
        all_residues.append(emb[idx])
    all_residues = np.concatenate(all_residues, axis=0)
    corpus_mean = all_residues.mean(axis=0)
    print(f"  Corpus mean from {len(all_residues)} sampled residues")

    # Step 2: Random orthogonal projection matrix (Johnson-Lindenstrauss)
    print(f"Building random projection matrix: 1024 → {d_out}...")
    D_in = 1024
    R = rng.standard_normal((D_in, d_out)).astype(np.float32)
    Q, _ = np.linalg.qr(R, mode="reduced")
    R = Q * np.sqrt(D_in / d_out)  # Scale to preserve norms

    # Step 3: Apply codec to each protein
    print(f"Applying codec (center → RP → DCT K={dct_k})...")
    n = 0
    with h5py.File(str(output_h5), "w") as out:
        for pid, emb in embeddings.items():
            # Center
            centered = emb - corpus_mean
            # Random projection
            projected = centered @ R  # (L, d_out)
            # DCT protein-level vector
            L = projected.shape[0]
            k = min(dct_k, L)
            coeffs = dct(projected, type=2, axis=0, norm="ortho")  # (L, d_out)
            protein_vec = coeffs[:k].ravel().astype(np.float32)  # (k * d_out,)

            out.create_dataset(pid, data=protein_vec)
            n += 1

    vec_dim = dct_k * d_out
    print(f"  Wrote {n} protein vectors of dim {vec_dim}")
    return n


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check prerequisites
    for path, name in [(PERRES_H5, "per-residue"), (CODEC_H5, "codec"), (BASELINE_H5, "baseline")]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    # Create balanced subset CSV
    subset = create_balanced_subset()
    subset.to_csv(SUBSET_CSV, index=False)
    print(f"Saved subset to {SUBSET_CSV}")

    protein_ids = set(subset["identifier"])

    # Copy pre-computed codec vectors (2048d)
    n = copy_vectors(CODEC_H5, OE_CODEC_H5, protein_ids)
    print(f"Copied {n} codec 2048d vectors to {OE_CODEC_H5}")

    # Copy mean-pooled baseline (1024d)
    n = copy_vectors(BASELINE_H5, OE_BASELINE_H5, protein_ids)
    print(f"Copied {n} baseline 1024d vectors to {OE_BASELINE_H5}")

    # Apply proper One Embedding codec (center → RP → DCT)
    oe_proper_h5 = OUT_DIR / "one_embedding_3072.h5"
    n = apply_codec_to_per_residue(protein_ids, oe_proper_h5, d_out=768, dct_k=4)
    print(f"Generated {n} One Embedding 3072d vectors to {oe_proper_h5}")

    # Summary
    print("\n=== Files ready ===")
    for f in sorted(OUT_DIR.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name}: {size_mb:.1f} MB")

    print("\n=== Configs to use ===")
    print("  Baseline:      configs/oe_baseline.yaml   (embedding_dim: 1024)")
    print("  Pre-computed:   configs/oe_codec.yaml      (embedding_dim: 2048)")
    print("  One Embedding:  configs/oe_proper.yaml     (embedding_dim: 3072)")


if __name__ == "__main__":
    main()
