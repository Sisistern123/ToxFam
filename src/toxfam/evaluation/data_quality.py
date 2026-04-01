"""Data quality profiling for bias detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

console = Console()


def profile_training_data(
    input_csv: str | Path,
    *,
    h5_path: str | Path | None = None,
    output_dir: str | Path = Path("data/profile"),
    sample_size: int = 500,
) -> dict:
    """Profile training data for potential biases.

    Reports:
    - Class distribution (toxic vs nontoxic, per-family)
    - Organism distribution for toxic vs nontoxic
    - Sequence length distributions
    - Optional: embedding similarity analysis
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    report: dict = {}

    # 1. Class distribution
    if "is_toxic" not in df.columns:
        from toxfam.evaluation.metrics import to_binary_class

        df["is_toxic"] = df["Protein families"].apply(
            lambda x: to_binary_class(x) == "toxin"
        )

    tox_count = df["is_toxic"].sum()
    nontox_count = (~df["is_toxic"]).sum()
    report["class_distribution"] = {
        "toxic": int(tox_count),
        "nontoxic": int(nontox_count),
        "ratio": f"1:{nontox_count / max(tox_count, 1):.1f}",
    }

    # 2. Family distribution
    fam_counts = df["Protein families"].value_counts().to_dict()
    report["family_distribution"] = {
        k: int(v) for k, v in fam_counts.items()
    }

    # 3. Split distribution
    if "Split" in df.columns:
        split_summary = {}
        for split in ["train", "val", "test"]:
            split_df = df[df["Split"] == split]
            split_summary[split] = {
                "total": len(split_df),
                "toxic": int(split_df["is_toxic"].sum()),
                "nontoxic": int((~split_df["is_toxic"]).sum()),
            }
        report["split_distribution"] = split_summary

    # 4. Organism distribution (if column exists)
    if "Organism" in df.columns:
        tox_orgs = df[df["is_toxic"]]["Organism"].value_counts().head(20)
        nontox_orgs = df[~df["is_toxic"]]["Organism"].value_counts().head(20)
        report["top_organisms"] = {
            "toxic": {k: int(v) for k, v in tox_orgs.items()},
            "nontoxic": {k: int(v) for k, v in nontox_orgs.items()},
        }

    # 5. Sequence length distribution
    if "Sequence" in df.columns:
        df["_seq_len"] = df["Sequence"].str.len()
        tox_lens = df[df["is_toxic"]]["_seq_len"]
        nontox_lens = df[~df["is_toxic"]]["_seq_len"]
        report["sequence_lengths"] = {
            "toxic": {
                "mean": float(tox_lens.mean()),
                "median": float(tox_lens.median()),
                "std": float(tox_lens.std()),
                "min": int(tox_lens.min()),
                "max": int(tox_lens.max()),
            },
            "nontoxic": {
                "mean": float(nontox_lens.mean()),
                "median": float(nontox_lens.median()),
                "std": float(nontox_lens.std()),
                "min": int(nontox_lens.min()),
                "max": int(nontox_lens.max()),
            },
        }
        df.drop(columns=["_seq_len"], inplace=True)

    # 6. Embedding similarity (optional, sample-based)
    if h5_path is not None and Path(h5_path).exists():
        import h5py
        from sklearn.metrics.pairwise import cosine_similarity

        with h5py.File(h5_path, "r") as h5f:
            tox_ids = df[df["is_toxic"]]["identifier"].tolist()
            nontox_ids = df[~df["is_toxic"]]["identifier"].tolist()

            # Sample
            rng = np.random.default_rng(42)
            tox_sample = rng.choice(
                tox_ids, min(sample_size, len(tox_ids)), replace=False
            )
            nontox_sample = rng.choice(
                nontox_ids, min(sample_size, len(nontox_ids)), replace=False
            )

            tox_embs = np.array([h5f[pid][:] for pid in tox_sample if pid in h5f])
            nontox_embs = np.array([h5f[pid][:] for pid in nontox_sample if pid in h5f])

        if len(tox_embs) > 1 and len(nontox_embs) > 1:
            # Intra-class similarity
            tox_sim = cosine_similarity(tox_embs)
            nontox_sim = cosine_similarity(nontox_embs)
            cross_sim = cosine_similarity(tox_embs, nontox_embs)

            report["embedding_similarity"] = {
                "toxic_intra_mean": float(
                    tox_sim[np.triu_indices_from(tox_sim, k=1)].mean()
                ),
                "nontoxic_intra_mean": float(
                    nontox_sim[np.triu_indices_from(nontox_sim, k=1)].mean()
                ),
                "cross_class_mean": float(cross_sim.mean()),
                "sample_size": sample_size,
            }

    # 7. Potential bias indicators
    bias_warnings = []
    if nontox_count > 10 * tox_count:
        bias_warnings.append(
            f"Severe class imbalance: {nontox_count / tox_count:.0f}:1 nontox:toxic"
        )
    if "Organism" in df.columns:
        tox_org_unique = df[df["is_toxic"]]["Organism"].nunique()
        nontox_org_unique = df[~df["is_toxic"]]["Organism"].nunique()
        if nontox_org_unique < tox_org_unique * 0.3:
            bias_warnings.append(
                f"Organism diversity gap: {tox_org_unique} toxic vs "
                f"{nontox_org_unique} nontoxic organisms"
            )
    report["bias_warnings"] = bias_warnings

    # Save report
    report_path = output_dir / "data_profile.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4, default=str)

    console.print(f"Data profile saved to {report_path}")
    console.print(f"  Total samples: {len(df)}")
    console.print(f"  Toxic: {tox_count}, Nontoxic: {nontox_count}")
    if bias_warnings:
        for w in bias_warnings:
            console.print(f"  WARNING: {w}")

    return report
