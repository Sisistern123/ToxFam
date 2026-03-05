#!/usr/bin/env python3
"""Explore the new ToxProtFeb2026 dataset.

Usage:
    uv run python scripts/explore_new_data.py [--xml PATH] [--h5 PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore ToxProtFeb2026 dataset")
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path(
            "/Users/jcoludar/CascadeProjects/SpeciesEmbedding/data/ToxProtFeb2026/"
            "uniprotkb_taxonomy_id_33208_AND_cc_tiss_2026_02_20.xml"
        ),
        help="Path to UniProt XML file",
    )
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path(
            "/Users/jcoludar/CascadeProjects/SpeciesEmbedding/data/ToxProtFeb2026/"
            "nOFOJT38a1.h5"
        ),
        help="Path to ProtT5 embeddings H5 file",
    )
    args = parser.parse_args()

    from toxfam.data.xml_parser import parse_uniprot_xml

    print("=" * 70)
    print("ToxProtFeb2026 Dataset Exploration")
    print("=" * 70)

    # 1. Parse XML
    print(f"\n1. Parsing XML: {args.xml}")
    df = parse_uniprot_xml(args.xml)
    print(f"   Total entries: {len(df)}")
    print(f"   With family annotation: {(df['Protein families'] != '').sum()}")
    print(f"   Without family annotation: {(df['Protein families'] == '').sum()}")
    print(f"   Unique families (raw): {df['Protein families'].nunique()}")

    # 2. Check H5 alignment
    print(f"\n2. Checking H5 alignment: {args.h5}")
    with h5py.File(args.h5, "r") as h5:
        h5_keys = set(h5.keys())
        sample_key = list(h5_keys)[0]
        emb_shape = h5[sample_key].shape
        emb_dtype = h5[sample_key].dtype

    xml_ids = set(df["identifier"])
    print(f"   H5 keys: {len(h5_keys)}")
    print(f"   Overlap: {len(h5_keys & xml_ids)}")
    print(f"   Embedding shape: {emb_shape}, dtype: {emb_dtype}")

    # 3. Family distribution
    print("\n3. Family distribution (top 30):")
    fam_counts = df["Protein families"].value_counts()
    for i, (fam, count) in enumerate(fam_counts.head(30).items()):
        label = fam if fam else "(no family)"
        print(f"   {i + 1:3d}. {label[:70]:70s} {count:5d}")

    # 4. Family size histogram
    print("\n4. Family size distribution:")
    with_fam = df[df["Protein families"] != ""]
    fam_sizes = with_fam["Protein families"].value_counts()
    buckets = [
        (">=100 members", (fam_sizes >= 100).sum()),
        ("50-99 members", ((fam_sizes >= 50) & (fam_sizes < 100)).sum()),
        ("20-49 members", ((fam_sizes >= 20) & (fam_sizes < 50)).sum()),
        ("10-19 members", ((fam_sizes >= 10) & (fam_sizes < 20)).sum()),
        ("5-9 members", ((fam_sizes >= 5) & (fam_sizes < 10)).sum()),
        ("<5 members", (fam_sizes < 5).sum()),
    ]
    for label, count in buckets:
        print(f"   {label:20s}: {count:4d} families")

    print(f"\n   Families with >=10 members: {(fam_sizes >= 10).sum()}")
    print(f"   Families with >=20 members: {(fam_sizes >= 20).sum()}")
    print(f"   Proteins in families >=10: {fam_sizes[fam_sizes >= 10].sum()}")
    print(f"   Proteins in families >=20: {fam_sizes[fam_sizes >= 20].sum()}")

    # 5. Check overlap with existing ToxFam data
    print("\n5. Overlap with existing ToxFam data:")
    existing_csv = Path("data/processed/training_data.csv")
    if existing_csv.exists():
        existing_df = pd.read_csv(existing_csv)
        existing_ids = set(existing_df["identifier"])
        overlap = xml_ids & existing_ids
        print(f"   Existing training data: {len(existing_df)} proteins")
        print(f"   Overlap with new data: {len(overlap)} proteins")
        print(f"   New proteins not in existing: {len(xml_ids - existing_ids)}")
        print(f"   Existing not in new: {len(existing_ids - xml_ids)}")

        # Family comparison
        existing_families = set(existing_df["Protein families"].unique())
        new_families = set(df[df["Protein families"] != ""]["Protein families"].unique())
        print(f"   Existing families: {len(existing_families)}")
        print(f"   New families: {len(new_families)}")
        print(f"   Families in both: {len(existing_families & new_families)}")
    else:
        print("   (existing training_data.csv not found - skipping)")

    # 6. Sequence length stats
    print("\n6. Sequence length statistics:")
    df["seq_len"] = df["Sequence"].str.len()
    print(f"   Min: {df['seq_len'].min()}")
    print(f"   Max: {df['seq_len'].max()}")
    print(f"   Mean: {df['seq_len'].mean():.1f}")
    print(f"   Median: {df['seq_len'].median():.1f}")

    print("\n" + "=" * 70)
    print("Exploration complete.")


if __name__ == "__main__":
    main()
