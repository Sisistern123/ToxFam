#!/usr/bin/env python3
"""Standalone ToxinPred3 inference wrapper for isolated environment.

This script runs in an isolated Python environment with scikit-learn 1.0.2
to load ToxinPred3's pickled model (incompatible with modern sklearn).

Usage (from isolated venv):
    python toxinpred3_isolated.py --input sequences.csv --output predictions.csv

Input CSV must have columns: identifier, Sequence
Output CSV will have columns: identifier, p_toxic

Setup:
    uv venv .toxinpred3_env --python 3.10
    source .toxinpred3_env/bin/activate
    pip install scikit-learn==1.0.2 joblib numpy pandas toxinpred3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compute_aac(sequences: list[str]) -> np.ndarray:
    """Amino acid composition (20-dim)."""
    std = list("ACDEFGHIKLMNPQRSTVWY")
    n = len(sequences)
    aac = np.zeros((n, 20), dtype=np.float32)
    for i, seq in enumerate(sequences):
        s = seq.upper()
        L = len(s)
        if L == 0:
            continue
        for j, aa in enumerate(std):
            aac[i, j] = (s.count(aa) / L) * 100
    return aac


def compute_dpc(sequences: list[str]) -> np.ndarray:
    """Dipeptide composition (400-dim)."""
    std = list("ACDEFGHIKLMNPQRSTVWY")
    n = len(sequences)
    dpc = np.zeros((n, 400), dtype=np.float32)
    for i, seq in enumerate(sequences):
        s = seq.upper()
        L = len(s)
        if L < 2:
            continue
        idx = 0
        for a in std:
            for b in std:
                dp = a + b
                count = sum(1 for m in range(L - 1) if s[m:m + 2] == dp)
                dpc[i, idx] = (count / (L - 1)) * 100
                idx += 1
    return dpc


def main():
    parser = argparse.ArgumentParser(description="ToxinPred3 isolated inference")
    parser.add_argument("--input", required=True, help="Input CSV (identifier, Sequence)")
    parser.add_argument("--output", required=True, help="Output CSV (identifier, p_toxic)")
    parser.add_argument("--model-path", default=None, help="Path to toxinpred3 model .pkl")
    args = parser.parse_args()

    # Find model
    model_path = args.model_path
    if model_path is None:
        try:
            import toxinpred3.python_scripts.toxinpred3 as tp3
            model_path = str(
                Path(tp3.__file__).parent.parent / "model" / "toxinpred3.0_model.pkl"
            )
        except ImportError:
            print("ERROR: toxinpred3 not installed in this environment", file=sys.stderr)
            sys.exit(1)

    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    # Load data
    df = pd.read_csv(args.input)
    sequences = df["Sequence"].tolist()
    identifiers = df["identifier"].tolist()

    print(f"Computing features for {len(sequences)} sequences...", file=sys.stderr)
    aac = compute_aac(sequences)
    dpc = compute_dpc(sequences)
    features = np.concatenate([aac, dpc], axis=1)

    # Load model and predict
    import joblib
    clf = joblib.load(model_path)
    p_scores = clf.predict_proba(features)
    p_toxic = p_scores[:, -1].astype(np.float64)

    # Save
    out_df = pd.DataFrame({"identifier": identifiers, "p_toxic": p_toxic})
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(out_df)} predictions to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
