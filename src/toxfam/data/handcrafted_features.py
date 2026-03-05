"""Handcrafted sequence features: Atchley factors + cysteine patterns.

These complement ProtT5 embeddings by capturing explicit physicochemical
properties and disulfide-bond patterns relevant to venom toxins.

Atchley factors (10-dim): Each amino acid maps to 5 physicochemical values
(polarity, secondary structure propensity, molecular size, codon diversity,
electrostatic charge). We compute mean + std across the sequence = 10 features.

Cysteine patterns (5-dim): Venom toxins are characterized by conserved
disulfide frameworks. Features: count, fraction, potential disulfide bonds,
inter-cysteine spacing variability, framework indicator.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from toxfam._paths import get_project_root

# Atchley factor values for 20 standard amino acids.
# Columns: polarity, secondary structure, molecular size, codon diversity, charge
# Source: Atchley et al. (2005) PNAS 102:6395-6400
ATCHLEY_FACTORS: dict[str, list[float]] = {
    "A": [-0.591, -1.302, -0.733, 1.570, -0.146],
    "C": [-1.343, 0.465, -0.862, -1.020, -0.255],
    "D": [1.050, 0.302, -3.656, -0.259, -3.242],
    "E": [1.357, -1.453, 1.477, 0.113, -0.837],
    "F": [-1.006, -0.590, 1.891, -0.397, 0.412],
    "G": [-0.384, 1.652, 1.330, 1.045, 2.064],
    "H": [0.336, -0.417, -1.673, -1.474, -0.078],
    "I": [-1.239, -0.547, 2.131, 0.393, 0.816],
    "K": [1.831, -0.561, 0.533, -0.277, 1.648],
    "L": [-1.019, -0.987, -1.505, 1.266, -0.912],
    "M": [-0.663, -1.524, 2.219, -1.005, 1.212],
    "N": [0.945, 0.828, 1.299, -0.169, 0.933],
    "P": [0.189, 2.081, -1.628, 0.421, -1.392],
    "Q": [0.931, -0.179, -3.005, -0.503, -1.853],
    "R": [1.538, -0.055, 1.502, 0.440, 2.897],
    "S": [-0.228, 1.399, -4.760, 0.670, -2.647],
    "T": [-0.032, 0.326, 2.213, 0.908, 1.313],
    "V": [-1.337, -0.279, -0.544, 1.242, -1.262],
    "W": [-0.595, 0.009, 0.672, -2.128, -0.184],
    "Y": [0.260, 0.830, 3.097, -0.838, 1.512],
}


def compute_atchley_features(sequences: list[str]) -> np.ndarray:
    """Compute Atchley factor statistics (10-dim) for each sequence.

    For each of the 5 Atchley factors, computes mean and standard deviation
    across the sequence. Non-standard amino acids are skipped.

    Returns shape (N, 10).
    """
    n = len(sequences)
    features = np.zeros((n, 10), dtype=np.float32)

    for i, seq in enumerate(sequences):
        seq_upper = seq.upper()
        factors = []
        for aa in seq_upper:
            if aa in ATCHLEY_FACTORS:
                factors.append(ATCHLEY_FACTORS[aa])
        if not factors:
            continue
        arr = np.array(factors, dtype=np.float64)
        features[i, :5] = arr.mean(axis=0)
        features[i, 5:] = arr.std(axis=0) if len(factors) > 1 else 0.0

    return features


def compute_cysteine_features(sequences: list[str]) -> np.ndarray:
    """Compute cysteine pattern features (5-dim) for each sequence.

    Features:
        0: log2(cysteine count + 1)
        1: cysteine fraction (count / length)
        2: log2(potential disulfide bonds + 1) = log2(floor(n_cys/2) + 1)
        3: coefficient of variation of inter-cysteine spacing (0 if <3 cys)
        4: common venom framework indicator (6, 8, or 10 cysteines)

    Returns shape (N, 5).
    """
    n = len(sequences)
    features = np.zeros((n, 5), dtype=np.float32)

    # Common venom cysteine frameworks
    venom_cys_counts = {6, 8, 10}

    for i, seq in enumerate(sequences):
        seq_upper = seq.upper()
        seq_len = len(seq_upper)
        if seq_len == 0:
            continue

        cys_positions = [j for j, aa in enumerate(seq_upper) if aa == "C"]
        n_cys = len(cys_positions)

        features[i, 0] = np.log2(n_cys + 1)
        features[i, 1] = n_cys / seq_len
        features[i, 2] = np.log2(n_cys // 2 + 1)

        # Inter-cysteine spacing CV
        if n_cys >= 3:
            spacings = np.diff(cys_positions)
            mean_spacing = spacings.mean()
            if mean_spacing > 0:
                features[i, 3] = spacings.std() / mean_spacing

        # Venom framework indicator
        if n_cys in venom_cys_counts:
            features[i, 4] = 1.0

    return features


def compute_all_handcrafted(sequences: list[str]) -> np.ndarray:
    """Compute all handcrafted features (15-dim): Atchley (10) + Cysteine (5).

    Returns shape (N, 15).
    """
    atchley = compute_atchley_features(sequences)
    cysteine = compute_cysteine_features(sequences)
    return np.concatenate([atchley, cysteine], axis=1)


def compute_and_save_handcrafted_features(
    input_csv: Path | None = None,
    output_h5: Path | None = None,
) -> Path:
    """Compute handcrafted features for all sequences and save to HDF5.

    Parameters
    ----------
    input_csv : Path to training CSV with identifier and Sequence columns.
    output_h5 : Path for output HDF5 file.

    Returns the output path.
    """
    root = get_project_root()
    if input_csv is None:
        input_csv = root / "data" / "processed" / "training_data.csv"
    if output_h5 is None:
        output_h5 = root / "data" / "intermediate" / "handcrafted" / "handcrafted_features.h5"

    output_h5.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    identifiers = df["identifier"].tolist()
    sequences = df["Sequence"].tolist()

    print(f"Computing handcrafted features for {len(sequences)} sequences...")
    features = compute_all_handcrafted(sequences)
    print(f"  Features shape: {features.shape} (Atchley 10 + Cysteine 5)")

    with h5py.File(str(output_h5), "w") as f:
        for j, sid in enumerate(identifiers):
            f.create_dataset(sid, data=features[j])

    print(f"  Saved to: {output_h5}")
    return output_h5


def compute_and_save_counterpart_handcrafted(
    counterpart_csv: Path | None = None,
    output_h5: Path | None = None,
) -> Path | None:
    """Compute handcrafted features for counterpart sequences."""
    root = get_project_root()
    if counterpart_csv is None:
        counterpart_csv = root / "data" / "intermediate" / "counterparts" / "all_counterparts.csv"
    if not counterpart_csv.exists():
        print("No counterpart CSV found, skipping.")
        return None

    if output_h5 is None:
        output_h5 = root / "data" / "intermediate" / "handcrafted" / "counterpart_handcrafted_features.h5"

    output_h5.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(counterpart_csv)
    identifiers = df["identifier"].tolist()
    sequences = df["Sequence"].tolist()

    print(f"Computing handcrafted features for {len(sequences)} counterpart sequences...")
    features = compute_all_handcrafted(sequences)

    with h5py.File(str(output_h5), "w") as f:
        for j, sid in enumerate(identifiers):
            f.create_dataset(sid, data=features[j])

    print(f"  Saved to: {output_h5}")
    return output_h5


def run_handcrafted_pipeline(
    input_csv: Path | None = None,
    output_h5: Path | None = None,
) -> Path:
    """Compute handcrafted features for training data + counterparts."""
    main_h5 = compute_and_save_handcrafted_features(input_csv, output_h5)
    cp_h5 = compute_and_save_counterpart_handcrafted()
    if cp_h5:
        print(f"  Counterpart features: {cp_h5}")
    return main_h5
