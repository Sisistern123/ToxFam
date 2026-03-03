from __future__ import annotations

from pathlib import Path

import pandas as pd
from Bio import SeqIO


def write_fasta(df: pd.DataFrame, filename: str | Path) -> None:
    """Write dataframe with 'identifier' and 'Sequence' columns to FASTA."""
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['identifier']}\n{row['Sequence']}\n")


def create_filtered_fasta(
    sp6_dir: str | Path,
    original_tsv: str | Path,
    output_fasta: str | Path,
) -> None:
    """Apply SignalP6 signal peptide removal and write filtered FASTA.

    Args:
        sp6_dir: Directory containing SignalP6 output
            (processed_entries.fasta, output.gff3)
        original_tsv: Path to the original TSV with protein sequences
        output_fasta: Path to write the filtered FASTA
    """
    sp6_dir = Path(sp6_dir)
    original_tsv = Path(original_tsv)
    output_fasta = Path(output_fasta)

    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load original data
    df = pd.read_csv(original_tsv, sep="\t")
    tsv_count = len(df)
    print(f"Number of entries loaded from TSV: {tsv_count}")

    if "Entry" in df.columns:
        df = df.rename(columns={"Entry": "identifier"})

    # 2. Load processed sequences from SignalP6
    records = SeqIO.parse(str(sp6_dir / "processed_entries.fasta"), "fasta")
    df_proc = pd.DataFrame(
        [
            {"identifier": rec.id.split("|")[-1], "Sequence_cut": str(rec.seq)}
            for rec in records
        ]
    )

    # 3. Load GFF3 for scores
    gff_cols = [
        "identifier",
        "source",
        "feature_type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
        "attributes",
    ]
    df_gff = pd.read_csv(sp6_dir / "output.gff3", sep="\t", comment="#", names=gff_cols)
    df_gff["identifier"] = (
        df_gff["identifier"].str.split("|").str[-1].str.split().str[0]
    )

    # 4. Filter for high-confidence cleavage (Score > 0.8)
    df_filtered = pd.merge(df_gff, df_proc, on="identifier")
    df_high_conf = df_filtered[df_filtered["score"] > 0.8][
        ["identifier", "Sequence_cut"]
    ]

    # 5. Merge and Replace
    df_final = df.merge(df_high_conf, on="identifier", how="left")
    df_final["Sequence"] = df_final["Sequence_cut"].fillna(df_final["Sequence"])

    # 6. Report and Export
    final_count = len(df_final)
    print(f"Number of entries in final FASTA: {final_count}")
    print(f"Signal peptides replaced: {len(df_high_conf)} (Confidence > 0.8)")

    write_fasta(df_final, output_fasta)
    print(f"Successfully created: {output_fasta}")
