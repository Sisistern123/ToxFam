import pandas as pd
from Bio import SeqIO
from pathlib import Path

# --- Constants (Adjust these to your local paths) ---
SP6_DIR = Path("/Users/selin/PycharmProjects/ToxFam/benchmark/new/evaluation/unreviewed")  # Or your specific dir
ORIGINAL_CSV = Path("/Users/selin/Desktop/unreviewed.tsv")
OUTPUT_FASTA = SP6_DIR / "unreviewed.fasta"


def write_fasta(df, filename):
    """Writes dataframe to FASTA format."""
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['identifier']}\n{row['Sequence']}\n")


def create_filtered_fasta():
    # 1. Load original data and print TSV count
    df = pd.read_csv(ORIGINAL_CSV, sep="\t")
    tsv_count = len(df)
    print(f"📊 Number of entries loaded from TSV: {tsv_count}")

    # 2. Normalize columns to match the pipeline
    if "Entry" in df.columns:
        df = df.rename(columns={"Entry": "identifier"})

    # 3. Load processed sequences from SignalP6
    records = SeqIO.parse(str(SP6_DIR / "processed_entries.fasta"), "fasta")
    df_proc = pd.DataFrame([
        {"identifier": rec.id.split('|')[-1], "Sequence_cut": str(rec.seq)}
        for rec in records
    ])

    # 4. Load GFF3 for scores
    gff_cols = ["identifier", "source", "feature_type", "start", "end",
                "score", "strand", "phase", "attributes"]
    df_gff = pd.read_csv(SP6_DIR / "output.gff3", sep="\t", comment="#", names=gff_cols)

    # Extract clean ID from GFF
    df_gff["identifier"] = df_gff["identifier"].str.split("|").str[-1].str.split().str[0]

    # 5. Filter for high-confidence cleavage (Score > 0.8)
    df_filtered = pd.merge(df_gff, df_proc, on="identifier")
    df_high_conf = df_filtered[df_filtered["score"] > 0.8][["identifier", "Sequence_cut"]]

    # 6. Merge and Replace logic
    df_final = df.merge(df_high_conf, on="identifier", how="left")
    df_final["Sequence"] = df_final["Sequence_cut"].fillna(df_final["Sequence"])

    # 7. Final reporting and Export
    final_count = len(df_final)
    print(f"📊 Number of entries in final FASTA: {final_count}")
    print(f"✨ Signal peptides replaced: {len(df_high_conf)} (Confidence > 0.8)")

    write_fasta(df_final, OUTPUT_FASTA)
    print(f"✅ Successfully created: {OUTPUT_FASTA}")


if __name__ == "__main__":
    create_filtered_fasta()