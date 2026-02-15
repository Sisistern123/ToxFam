"""Toxin dataset pipeline: filtering, clustering, stratified splits."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from pymmseqs.commands import easy_cluster
from sklearn.preprocessing import MultiLabelBinarizer

from toxfam._paths import get_project_root


def _base_dir() -> Path:
    return get_project_root()


def _data_dir() -> Path:
    return _base_dir() / "data"


# ---------- Utilities ----------


def ensure_dirs() -> None:
    dirs = [
        _data_dir() / "raw",
        _data_dir() / "interm",
        _data_dir() / "protspace",
        _data_dir() / "sp6" / "tox",
        _data_dir() / "sp6" / "nontox",
        _data_dir() / "families",
        _data_dir() / "mmseqs",
        _base_dir() / "benchmark" / "HBI",
        _base_dir() / "benchmark",
    ]
    for p in dirs:
        p.mkdir(parents=True, exist_ok=True)


def write_fasta(df: pd.DataFrame, filename: os.PathLike | str) -> None:
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['identifier']}\n{row['Sequence']}\n")


def fasta_to_dataframe(fasta_file: os.PathLike | str) -> pd.DataFrame:
    records = SeqIO.parse(str(fasta_file), "fasta")
    return pd.DataFrame(
        [
            {"identifier": rec.id.split("|")[-1], "Sequence": str(rec.seq)}
            for rec in records
        ]
    )


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


# ---------- Preprocessing ----------


def load_and_prepare_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = _data_dir() / "raw"

    tox = (
        pd.read_csv(raw_dir / "0800.tsv", sep="\t")
        .dropna(subset=["Protein families"])
        .copy()
    )
    tox.rename(columns={"Entry": "identifier"}, inplace=True)

    tox["Protein families"] = tox["Protein families"].str.split(";").str[0]
    tox["Protein families"] = tox["Protein families"].str.split(",").str[0]

    repl = {
        "I1 superfamily": "Conotoxin I1 superfamily",
        "O1 superfamily": "Conotoxin O1 superfamily",
        "O2 superfamily": "Conotoxin O2 superfamily",
        "E superfamily": "Conotoxin E superfamily",
        "F superfamily": "Conotoxin F superfamily",
    }
    tox["Protein families"] = tox["Protein families"].replace(repl)

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
        r"Bradykinin-potentiating peptide family|Natriuretic peptide family|Natriuretic": "Natriuretic, Bradykinin potentiating peptide family",
        r".*phospholipase.*|.*Phospholipase.*": "Phospholipase family",
    }

    for pattern, replacement in mapping.items():
        tox["Protein families"] = tox["Protein families"].str.replace(
            pattern, replacement, regex=True
        )

    tox["Protein families"] = tox["Protein families"].where(
        tox["Protein families"].map(tox["Protein families"].value_counts()) >= 10,
        "other",
    )

    nontox = pd.read_csv(raw_dir / "nontox.tsv", sep="\t").copy()
    nontox.rename(columns={"Entry": "identifier"}, inplace=True)
    cutoff = (
        nontox["Sequence"]
        .str.len()
        .nlargest(int(np.ceil(len(nontox) * 0.01)))
        .min()
    )
    nontox = nontox[nontox["Sequence"].str.len() <= cutoff].reset_index(drop=True)
    nontox["Protein families"] = "nontox"

    return tox, nontox


def apply_signalp_filtered_sequences(
    tox: pd.DataFrame, nontox: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sp6_tox_dir = _data_dir() / "sp6" / "tox"
    sp6_nontox_dir = _data_dir() / "sp6" / "nontox"

    def load_signalp_output(proc_path, gff_path):
        df_proc = fasta_to_dataframe(proc_path)
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
        df_gff = pd.read_csv(gff_path, sep="\t", comment="#", names=gff_cols)
        df_gff["identifier"] = (
            df_gff["identifier"].str.split("|").str[-1].str.split().str[0]
        )
        df = pd.merge(df_gff, df_proc, on="identifier")
        return df[df["score"] > 0.8][["identifier", "Sequence"]].rename(
            columns={"Sequence": "Sequence_new"}
        )

    tox_filt = load_signalp_output(
        sp6_tox_dir / "processed_entries.fasta", sp6_tox_dir / "output.gff3"
    )
    nontox_filt = load_signalp_output(
        sp6_nontox_dir / "processed_entries.fasta", sp6_nontox_dir / "output.gff3"
    )

    tox = tox.merge(tox_filt, on="identifier", how="left")
    nontox = nontox.merge(nontox_filt, on="identifier", how="left")

    tox["Sequence"] = tox["Sequence_new"].fillna(tox["Sequence"])
    nontox["Sequence"] = nontox["Sequence_new"].fillna(nontox["Sequence"])
    tox.drop(columns="Sequence_new", inplace=True, errors="ignore")
    nontox.drop(columns="Sequence_new", inplace=True, errors="ignore")
    return tox, nontox


# ---------- SignalP6 wrapper ----------


def maybe_run_signalp6(extra_args: str = "--organism euk") -> None:
    sp6_tox_dir = _data_dir() / "sp6" / "tox"
    sp6_nontox_dir = _data_dir() / "sp6" / "nontox"

    script_path = _base_dir() / "scripts" / "run_signalp6.sh"

    if all(
        [
            (sp6_tox_dir / "output.gff3").exists(),
            (sp6_nontox_dir / "output.gff3").exists(),
            (sp6_tox_dir / "processed_entries.fasta").exists(),
            (sp6_nontox_dir / "processed_entries.fasta").exists(),
        ]
    ):
        print("SignalP6 outputs already exist -- skipping SignalP6 run.")
        return

    print("Running SignalP6 preprocessing via conda env 'signalp6'...")
    try:
        subprocess.run(
            ["bash", str(script_path), "--extra-args", extra_args], check=True
        )
        print("SignalP6 completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"SignalP6 failed: {e}. Continuing with unprocessed sequences.")


# ---------- MMseqs2 & splitting ----------


def cluster_per_family_and_collect(
    data: pd.DataFrame, min_seq_id: float = 0.9
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    families_dir = _data_dir() / "families"
    mmseqs_dir = _data_dir() / "mmseqs"
    families_dir.mkdir(parents=True, exist_ok=True)
    mmseqs_dir.mkdir(parents=True, exist_ok=True)
    failures: List[Tuple[str, str, str]] = []

    for family, group in data.groupby("Protein families"):
        safe = sanitize_filename(family)
        family_fa = families_dir / f"{safe}.fasta"
        write_fasta(group, family_fa)

        fam_mm_dir = mmseqs_dir / safe
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
            print(f"MMseqs easy-cluster failed for {safe}: {e}")
            failures.append((str(family_fa), str(cluster_prefix), str(tmp_dir)))

    if failures:
        print("\nManual mmseqs2 commands for failed entries:")
        for fasta, out, tmp in failures:
            print(
                f"mmseqs easy-cluster {fasta} {out} {tmp} --min-seq-id {min_seq_id}"
            )

    rep_seqs_all, rep_seqs_tox = [], []
    for family_dir in os.listdir(mmseqs_dir):
        full_path = mmseqs_dir / family_dir
        rep_fasta = full_path / "cluster_rep_seq.fasta"
        if not rep_fasta.exists():
            continue
        seqs = [
            {"identifier": rec.id, "Sequence": str(rec.seq)}
            for rec in SeqIO.parse(str(rep_fasta), "fasta")
        ]
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
            df["Protein families"].map(df["Protein families"].value_counts()) >= 10,
            "other",
        )
    return rep_df_all, rep_df_tox


def multilabel_stratified_splits(rep_df_all: pd.DataFrame):
    df = rep_df_all.copy()
    df["fam_list"] = df["Protein families"].apply(
        lambda x: x.split(",") if isinstance(x, str) else []
    )
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["fam_list"])

    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.15, random_state=42
    )
    trainval_idx, test_idx = next(msss1.split(df, Y))
    df_trainval, test_df = df.iloc[trainval_idx], df.iloc[test_idx]
    Y_trainval = Y[trainval_idx]

    val_frac = 0.15 / 0.85
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=42
    )
    train_idx, val_idx = next(msss2.split(df_trainval, Y_trainval))
    train_df = df_trainval.iloc[train_idx].reset_index(drop=True)
    val_df = df_trainval.iloc[val_idx].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    for subset in (train_df, val_df, test_df):
        subset["Protein families"] = subset["fam_list"].apply(",".join)
        subset.drop(columns="fam_list", inplace=True)
    return train_df, val_df, test_df


def build_train_all_members(
    data: pd.DataFrame, train_df: pd.DataFrame
) -> pd.DataFrame:
    mmseqs_dir = _data_dir() / "mmseqs"
    train_reps = set(train_df["identifier"])
    rep2members: Dict[str, Set[str]] = {}
    for family in os.listdir(mmseqs_dir):
        fam_dir = mmseqs_dir / family
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


# ---------- Main pipeline ----------


def run_preprocessing_pipeline(
    *,
    run_signalp6: bool = False,
    signalp6_extra: str = "--organism euk",
    min_seq_id: float = 0.9,
) -> None:
    """Run the full preprocessing pipeline."""
    raw_dir = _data_dir() / "raw"
    interm_dir = _data_dir() / "interm"
    protspace_dir = _data_dir() / "protspace"
    sp6_tox_dir = _data_dir() / "sp6" / "tox"
    sp6_nontox_dir = _data_dir() / "sp6" / "nontox"
    bench_dir = _base_dir() / "benchmark"
    bench_hbi_dir = bench_dir / "HBI"

    ensure_dirs()
    tox, nontox = load_and_prepare_raw()

    write_fasta(tox, raw_dir / "tox.fasta")
    write_fasta(nontox, raw_dir / "nontox.fasta")

    if run_signalp6:
        maybe_run_signalp6(signalp6_extra)
    else:
        print("Skipping SignalP6 call (use --run-signalp6 to enable).")

    if (sp6_tox_dir / "output.gff3").exists() and (
        sp6_nontox_dir / "output.gff3"
    ).exists():
        tox, nontox = apply_signalp_filtered_sequences(tox, nontox)
    else:
        print("No SignalP6 output found -- continuing with unprocessed sequences.")

    nontox["Protein families"] = "nontox"
    data = pd.concat([tox, nontox], ignore_index=True)

    write_fasta(tox, interm_dir / "tox_noSP.fasta")
    write_fasta(nontox, interm_dir / "nontox_noSP.fasta")

    rep_df_all, rep_df_tox = cluster_per_family_and_collect(
        data, min_seq_id=min_seq_id
    )

    rep_df_tox[["identifier", "Protein families"]].to_csv(
        protspace_dir / "tox.csv", index=False
    )
    rep_df_all[["identifier", "Protein families"]].to_csv(
        protspace_dir / "all.csv", index=False
    )
    write_fasta(rep_df_tox, protspace_dir / "tox.fasta")
    write_fasta(rep_df_all, protspace_dir / "all.fasta")

    train_df, val_df, test_df = multilabel_stratified_splits(rep_df_all)
    train_df["Split"], val_df["Split"], test_df["Split"] = "train", "val", "test"
    training_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    training_data.to_csv(interm_dir / "training_data.csv", index=False)

    train_all_df = build_train_all_members(data, train_df)
    bench_hbi_dir.mkdir(parents=True, exist_ok=True)
    train_all_df.to_csv(bench_hbi_dir / "train_all_df.csv", index=False)
    write_fasta(train_all_df, bench_hbi_dir / "train_all_members.fasta")
    test_df.to_csv(bench_dir / "test_data.csv", index=False)
    write_fasta(test_df, bench_dir / "test_data.fasta")
    val_df.to_csv(bench_dir / "val_data.csv", index=False)
    write_fasta(val_df, bench_dir / "val_data.fasta")

    print("Pipeline finished successfully.")
