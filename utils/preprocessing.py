#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toxin dataset pipeline with optional SignalP6 preprocessing, MMseqs2 clustering, and stratified splits.

Outputs:
  - ../data/raw/tox.fasta
  - ../data/raw/nontox.fasta
  - ../data/interm/tox_noSP.fasta
  - ../data/interm/nontox_noSP.fasta
  - ../data/protspace/tox.csv
  - ../data/protspace/all.csv
  - ../data/protspace/tox.fasta
  - ../data/protspace/all.fasta
  - ../data/interm/training_data.csv
  - ../benchmark/HBI/train_all_df.csv
  - ../benchmark/HBI/train_all_members.fasta
  - ../benchmark/test_data.csv
  - ../benchmark/test_data.fasta
  - ../data/sp6/{tox,nontox}/{output.gff3, processed_entries.fasta} (from SignalP6)
"""

from __future__ import annotations
import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Set

import numpy as np
import pandas as pd
from Bio import SeqIO
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from pymmseqs.commands import easy_cluster


# ---------- Constants ----------
BASE_DIR = Path(__file__).resolve().parent.parent  # one level above utils/
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERM_DIR = DATA_DIR / "interm"
PROTSPACE_DIR = DATA_DIR / "protspace"
SP6_TOX_DIR = DATA_DIR / "sp6/tox"
SP6_NONTox_DIR = DATA_DIR / "sp6/nontox"
FAMILIES_DIR = DATA_DIR / "families"
MMSEQS_DIR = DATA_DIR / "mmseqs"
BENCH_DIR = BASE_DIR / "benchmark"
BENCH_HBI_DIR = BENCH_DIR / "HBI"


# ---------- Utilities ----------
def ensure_dirs() -> None:
    for p in [
        RAW_DIR, INTERM_DIR, PROTSPACE_DIR,
        SP6_TOX_DIR, SP6_NONTox_DIR,
        FAMILIES_DIR, MMSEQS_DIR, BENCH_HBI_DIR, BENCH_DIR
    ]:
        p.mkdir(parents=True, exist_ok=True)


def write_fasta(df: pd.DataFrame, filename: os.PathLike | str) -> None:
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['identifier']}\n{row['Sequence']}\n")


def fasta_to_dataframe(fasta_file: os.PathLike | str) -> pd.DataFrame:
    records = SeqIO.parse(str(fasta_file), "fasta")
    return pd.DataFrame(
        [{"identifier": rec.id.split('|')[-1], "Sequence": str(rec.seq)} for rec in records]
    )


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


# ---------- Preprocessing ----------
def load_and_prepare_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # --- Toxic proteins ---
    tox_raw = pd.read_csv(RAW_DIR / "0800.tsv", sep="\t")
    print(f"Loaded tox table with {len(tox_raw)} rows.")
    tox = tox_raw.dropna(subset=["Protein families"]).copy()
    print(f"Tox after dropping rows without 'Protein families': {len(tox)} rows.")
    tox.rename(columns={"Entry": "identifier"}, inplace=True)

    # Keep a copy to track changes to family labels
    original_families = tox["Protein families"].copy()

    # --- basic normalization ---
    tox["Protein families"] = tox["Protein families"].str.split(";").str[0]
    tox["Protein families"] = tox["Protein families"].str.split(",").str[0]

    # Specific replacements (Conotoxin superfamilies)
    repl = {
        "I1 superfamily": "Conotoxin I1 superfamily",
        "O1 superfamily": "Conotoxin O1 superfamily",
        "O2 superfamily": "Conotoxin O2 superfamily",
        "E superfamily": "Conotoxin E superfamily",
        "F superfamily": "Conotoxin F superfamily",
    }
    tox["Protein families"] = tox["Protein families"].replace(repl)

    # --- regex-based collapsing into broad families ---
    mapping = {
        r"Conotoxin.*": "Conotoxin family",
        r"Neurotoxin.*": "Neurotoxin family",
        r"Scoloptoxin.*|Scolopendra.*": "Scoloptoxin family",
        r"Caterpillar.*": "Caterpillar family",
        r"Teretoxin.*": "Teretoxin family",
        r"Limacoditoxin.*": "Limacoditoxin family",
        r"Scutigerotoxin.*": "Scutigerotoxin family",
        r"Cationic peptide.*": "Cationic peptide family",
        r"Formicidae venom.*": "Formicidae venom family",
        r"Bradykinin-potentiating peptide family|Natriuretic peptide family|Natriuretic":
            "Natriuretic, Bradykinin potentiating peptide family",
        r".*phospholipase.*|.*Phospholipase.*": "Phospholipase family",
    }

    for pattern, replacement in mapping.items():
        tox["Protein families"] = tox["Protein families"].str.replace(pattern, replacement, regex=True)
    # Report how many family labels changed (approximate mapping effect)
    changed_families = (tox["Protein families"] != original_families).sum()
    print(f"Tox entries whose family label changed after normalization/mapping: {changed_families}")

    # Group rare families into "other"
    family_counts_before_other = tox["Protein families"].value_counts()
    tox["Protein families"] = tox["Protein families"].where(
        tox["Protein families"].map(tox["Protein families"].value_counts()) >= 10, "other"
    )
    n_other = (tox["Protein families"] == "other").sum()
    print("Tox family distribution before grouping rare families (top 30):")
    print(family_counts_before_other.head(30).to_string())
    print(f"Tox entries assigned to 'other' family: {n_other}")

    # --- Non-toxic proteins ---
    nontox_raw = pd.read_csv(RAW_DIR / "nontox.tsv", sep="\t").copy()
    print(f"Loaded nontox table with {len(nontox_raw)} rows.")
    nontox_raw.rename(columns={"Entry": "identifier"}, inplace=True)
    cutoff = nontox_raw["Sequence"].str.len().nlargest(int(np.ceil(len(nontox_raw) * 0.01))).min()
    nontox = nontox_raw[nontox_raw["Sequence"].str.len() <= cutoff].reset_index(drop=True)
    print(
        f"Nontox after length filtering: {len(nontox)} rows "
        f"(filtered out {len(nontox_raw) - len(nontox)} longest sequences)."
    )
    nontox["Protein families"] = "nontox"

    return tox, nontox



def apply_signalp_filtered_sequences(tox: pd.DataFrame, nontox: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def load_signalp_output(proc_path, gff_path):
        df_proc = fasta_to_dataframe(proc_path)
        gff_cols = ["identifier", "source", "feature_type", "start", "end",
                    "score", "strand", "phase", "attributes"]
        df_gff = pd.read_csv(gff_path, sep="\t", comment="#", names=gff_cols)
        df_gff["identifier"] = df_gff["identifier"].str.split("|").str[-1].str.split().str[0]
        df = pd.merge(df_gff, df_proc, on="identifier")
        return df[df["score"] > 0.8][["identifier", "Sequence"]].rename(columns={"Sequence": "Sequence_new"})

    tox_filt = load_signalp_output(SP6_TOX_DIR / "processed_entries.fasta", SP6_TOX_DIR / "output.gff3")
    nontox_filt = load_signalp_output(SP6_NONTox_DIR / "processed_entries.fasta", SP6_NONTox_DIR / "output.gff3")

    print(
        f"SignalP6 filtered sequences: {len(tox_filt)} tox and {len(nontox_filt)} nontox "
        f"entries with score > 0.8."
    )

    tox = tox.merge(tox_filt, on="identifier", how="left")
    nontox = nontox.merge(nontox_filt, on="identifier", how="left")

    tox_changed = tox["Sequence_new"].notna().sum()
    nontox_changed = nontox["Sequence_new"].notna().sum()
    print(
        "SignalP6 sequence updates (proxy for signal peptide removal): "
        f"{tox_changed} tox and {nontox_changed} nontox sequences updated."
    )

    tox["Sequence"] = tox["Sequence_new"].fillna(tox["Sequence"])
    nontox["Sequence"] = nontox["Sequence_new"].fillna(nontox["Sequence"])
    tox.drop(columns="Sequence_new", inplace=True, errors="ignore")
    nontox.drop(columns="Sequence_new", inplace=True, errors="ignore")
    return tox, nontox


# ---------- SignalP6 wrapper ----------
def maybe_run_signalp6(extra_args: str = "--organism euk") -> None:
    script_path = Path(__file__).resolve().parent / "run_signalp6.sh"
    if all([
        (SP6_TOX_DIR / "output.gff3").exists(),
        (SP6_NONTox_DIR / "output.gff3").exists(),
        (SP6_TOX_DIR / "processed_entries.fasta").exists(),
        (SP6_NONTox_DIR / "processed_entries.fasta").exists()
    ]):
        print("✅ SignalP6 outputs already exist — skipping SignalP6 run.")
        return

    print("⚙️ Running SignalP6 preprocessing via conda env 'signalp6'...")
    try:
        subprocess.run(
            ["bash", str(script_path), "--extra-args", extra_args],
            check=True
        )
        print("✅ SignalP6 completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ SignalP6 failed: {e}. Continuing with unprocessed sequences.")



# ---------- MMseqs2 & splitting ----------
def cluster_per_family_and_collect(data: pd.DataFrame, min_seq_id: float = 0.9) -> Tuple[pd.DataFrame, pd.DataFrame]:
    FAMILIES_DIR.mkdir(parents=True, exist_ok=True)
    MMSEQS_DIR.mkdir(parents=True, exist_ok=True)
    failures: List[Tuple[str, str, str]] = []

    print(
        f"Clustering {len(data)} sequences across {data['Protein families'].nunique()} "
        f"protein families with MMseqs2 (min_seq_id={min_seq_id})."
    )

    for family, group in data.groupby("Protein families"):
        safe = sanitize_filename(family)
        family_fa = FAMILIES_DIR / f"{safe}.fasta"
        write_fasta(group, family_fa)

        fam_mm_dir = MMSEQS_DIR / safe
        fam_mm_dir.mkdir(parents=True, exist_ok=True)
        cluster_prefix = fam_mm_dir / "cluster"
        tmp_dir = fam_mm_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            easy_cluster(
                fasta_files=str(family_fa),
                cluster_prefix=str(cluster_prefix),
                tmp_dir=str(tmp_dir),
                min_seq_id=min_seq_id,
            )
        except Exception as e:
            print(f"⚠️ MMseqs easy-cluster failed for {safe}: {e}")
            failures.append((str(family_fa), str(cluster_prefix), str(tmp_dir)))

    if failures:
        print("\nManual mmseqs2 commands for failed entries:")
        for fasta, out, tmp in failures:
            print(f"mmseqs easy-cluster {fasta} {out} {tmp} --min-seq-id {min_seq_id}")

    rep_seqs_all, rep_seqs_tox = [], []
    for family_dir in os.listdir(MMSEQS_DIR):
        full_path = MMSEQS_DIR / family_dir
        rep_fasta = full_path / "cluster_rep_seq.fasta"
        if not rep_fasta.exists():
            continue
        seqs = [{"identifier": rec.id, "Sequence": str(rec.seq)} for rec in SeqIO.parse(str(rep_fasta), "fasta")]
        rep_seqs_all.extend(seqs)
        if family_dir != "nontox":
            rep_seqs_tox.extend(seqs)

    rep_df_all = pd.DataFrame(rep_seqs_all).merge(
        data[["identifier", "Protein families"]], on="identifier", how="left"
    )
    rep_df_tox = pd.DataFrame(rep_seqs_tox).merge(
        data[["identifier", "Protein families"]], on="identifier", how="left"
    )

    for df in (rep_df_all, rep_df_tox):
        df["Protein families"] = df["Protein families"].where(
            df["Protein families"].map(df["Protein families"].value_counts()) >= 10, "other"
        )

    print(
        f"Cluster representatives: {len(rep_df_all)} total "
        f"({len(rep_df_tox)} toxic, {len(rep_df_all) - len(rep_df_tox)} nontox)."
    )
    return rep_df_all, rep_df_tox


def multilabel_stratified_splits(rep_df_all: pd.DataFrame):
    df = rep_df_all.copy()
    df["fam_list"] = df["Protein families"].apply(lambda x: x.split(",") if isinstance(x, str) else [])
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["fam_list"])

    print(
        f"Preparing stratified splits on {len(df)} cluster representatives "
        f"covering {len(mlb.classes_)} unique protein families."
    )

    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    trainval_idx, test_idx = next(msss1.split(df, Y))
    df_trainval, test_df = df.iloc[trainval_idx], df.iloc[test_idx]
    Y_trainval = Y[trainval_idx]

    val_frac = 0.15 / 0.85
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
    train_idx, val_idx = next(msss2.split(df_trainval, Y_trainval))
    train_df = df_trainval.iloc[train_idx].reset_index(drop=True)
    val_df = df_trainval.iloc[val_idx].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    def _split_stats(name: str, subset: pd.DataFrame) -> None:
        fam_lists = subset["fam_list"]
        n_clusters = len(subset)
        families: Set[str] = set()
        for fams in fam_lists:
            families.update(fams)
        print(f"{name} split: {n_clusters} cluster representatives, {len(families)} unique protein families.")

    _split_stats("Train", train_df)
    _split_stats("Validation", val_df)
    _split_stats("Test", test_df)

    for subset in (train_df, val_df, test_df):
        subset["Protein families"] = subset["fam_list"].apply(",".join)
        subset.drop(columns="fam_list", inplace=True)
    return train_df, val_df, test_df


def build_train_all_members(data: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    train_reps = set(train_df["identifier"])
    rep2members: Dict[str, Set[str]] = {}
    for family in os.listdir(MMSEQS_DIR):
        fam_dir = MMSEQS_DIR / family
        tsv_path = fam_dir / "cluster_cluster.tsv"
        if not tsv_path.exists():
            continue
        with open(tsv_path) as tsv:
            for line in tsv:
                rep, member = line.rstrip().split("\t")
                if rep in train_reps:
                    rep2members.setdefault(rep, set()).add(member)
    for rep in train_reps:
        rep2members.setdefault(rep, set()).add(rep)
    all_members = set().union(*rep2members.values()) if rep2members else set()
    train_all_df = (
        data.loc[data["identifier"].isin(all_members)]
        .drop_duplicates(subset="identifier")
        .reset_index(drop=True)
    )
    return train_all_df


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Run toxin dataset pipeline")
    parser.add_argument("--run-signalp6", action="store_true", help="Run SignalP6 preprocessing (optional)")
    parser.add_argument("--signalp6-extra", default="--organism euk", help="Extra args for SignalP6")
    parser.add_argument("--min-seq-id", type=float, default=0.9, help="MMseqs2 clustering identity threshold")
    args = parser.parse_args()

    ensure_dirs()
    tox, nontox = load_and_prepare_raw()

    # Save initial FASTAs
    write_fasta(tox, RAW_DIR / "tox.fasta")
    write_fasta(nontox, RAW_DIR / "nontox.fasta")

    # Run SignalP6 optionally
    if args.run_signalp6:
        maybe_run_signalp6(args.signalp6_extra)
    else:
        print("⚠️ Skipping SignalP6 call (use --run-signalp6 to enable).")

    # Apply processed sequences if SignalP6 outputs exist
    if (SP6_TOX_DIR / "output.gff3").exists() and (SP6_NONTox_DIR / "output.gff3").exists():
        tox, nontox = apply_signalp_filtered_sequences(tox, nontox)
    else:
        print("⚠️ No SignalP6 output found — continuing with unprocessed sequences.")

    # Combine datasets
    nontox["Protein families"] = "nontox"
    data = pd.concat([tox, nontox], ignore_index=True)

    # Save intermediary FASTAs
    write_fasta(tox, INTERM_DIR / "tox_noSP.fasta")
    write_fasta(nontox, INTERM_DIR / "nontox_noSP.fasta")

    # Cluster and collect representatives
    rep_df_all, rep_df_tox = cluster_per_family_and_collect(data, min_seq_id=args.min_seq_id)

    # Save ProtSpace outputs
    rep_df_tox[["identifier", "Protein families"]].to_csv(PROTSPACE_DIR / "tox.csv", index=False)
    rep_df_all[["identifier", "Protein families"]].to_csv(PROTSPACE_DIR / "all.csv", index=False)
    write_fasta(rep_df_tox, PROTSPACE_DIR / "tox.fasta")
    write_fasta(rep_df_all, PROTSPACE_DIR / "all.fasta")

    # Splits
    train_df, val_df, test_df = multilabel_stratified_splits(rep_df_all)
    train_df["Split"], val_df["Split"], test_df["Split"] = "train", "val", "test"
    training_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    training_data.to_csv(INTERM_DIR / "training_data.csv", index=False)

    # Build benchmark data
    train_all_df = build_train_all_members(data, train_df)
    BENCH_HBI_DIR.mkdir(parents=True, exist_ok=True)
    train_all_df.to_csv(BENCH_HBI_DIR / "train_all_df.csv", index=False)
    write_fasta(train_all_df, BENCH_HBI_DIR / "train_all_members.fasta")
    test_df.to_csv(BENCH_DIR / "test_data.csv", index=False)
    write_fasta(test_df, BENCH_DIR / "test_data.fasta")
    val_df.to_csv(BENCH_DIR / "val_data.csv", index=False)
    write_fasta(val_df, BENCH_DIR / "val_data.fasta")

    print("✅ Pipeline finished successfully.")


if __name__ == "__main__":
    main()
