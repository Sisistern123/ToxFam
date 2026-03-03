"""Toxin dataset pipeline: filtering, clustering, stratified splits."""

from __future__ import annotations

import io
import os
import re
import subprocess
from contextlib import redirect_stdout
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from pymmseqs.commands import easy_cluster
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from sklearn.preprocessing import MultiLabelBinarizer

from toxfam._paths import get_project_root, raw_dir, intermediate_dir, processed_dir

console = Console()


# ---------- Utilities ----------


def ensure_dirs() -> None:
    interm = intermediate_dir()
    dirs = [
        raw_dir(),
        processed_dir(),
        interm / "fasta",
        interm / "families",
        interm / "mmseqs",
        interm / "sp6" / "tox",
        interm / "sp6" / "nontox",
        interm / "representatives",
        interm / "taxonomy",
        get_project_root() / "benchmark" / "HBI",
        get_project_root() / "benchmark",
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
    raw = raw_dir()

    tox = (
        pd.read_csv(raw / "0800.tsv", sep="\t")
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

    nontox = pd.read_csv(raw / "nontox.tsv", sep="\t").copy()
    nontox.rename(columns={"Entry": "identifier"}, inplace=True)
    cutoff = (
        nontox["Sequence"].str.len().nlargest(int(np.ceil(len(nontox) * 0.01))).min()
    )
    nontox = nontox[nontox["Sequence"].str.len() <= cutoff].reset_index(drop=True)
    nontox["Protein families"] = "nontox"

    return tox, nontox


def apply_signalp_filtered_sequences(
    tox: pd.DataFrame, nontox: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sp6_tox_dir = intermediate_dir() / "sp6" / "tox"
    sp6_nontox_dir = intermediate_dir() / "sp6" / "nontox"

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
    sp6_tox_dir = intermediate_dir() / "sp6" / "tox"
    sp6_nontox_dir = intermediate_dir() / "sp6" / "nontox"

    script_path = get_project_root() / "scripts" / "run_signalp6.sh"

    if all(
        [
            (sp6_tox_dir / "output.gff3").exists(),
            (sp6_nontox_dir / "output.gff3").exists(),
            (sp6_tox_dir / "processed_entries.fasta").exists(),
            (sp6_nontox_dir / "processed_entries.fasta").exists(),
        ]
    ):
        console.print("  Using cached SignalP6 output")
        return

    console.print("  Running SignalP6 via conda env 'signalp6'...")
    try:
        subprocess.run(
            ["bash", str(script_path), "--extra-args", extra_args], check=True
        )
        console.print("  SignalP6 completed")
    except subprocess.CalledProcessError as e:
        console.print(f"  [yellow]SignalP6 failed: {e}[/]")


# ---------- MMseqs2 & splitting ----------


def cluster_per_family_and_collect(
    data: pd.DataFrame, min_seq_id: float = 0.9
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    families_dir = intermediate_dir() / "families"
    mmseqs_dir = intermediate_dir() / "mmseqs"
    families_dir.mkdir(parents=True, exist_ok=True)
    mmseqs_dir.mkdir(parents=True, exist_ok=True)
    failures: List[Tuple[str, str, str]] = []

    grouped = list(data.groupby("Protein families"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
        refresh_per_second=30,
    ) as progress:
        task = progress.add_task("Clustering families", total=len(grouped))
        for family, group in grouped:
            safe = sanitize_filename(family)
            progress.update(
                task, description=f"Clustering [cyan]{safe}[/]", refresh=True
            )
            family_fa = families_dir / f"{safe}.fasta"
            write_fasta(group, family_fa)

            fam_mm_dir = mmseqs_dir / safe
            fam_mm_dir.mkdir(parents=True, exist_ok=True)
            cluster_prefix = fam_mm_dir / "cluster"
            tmp_dir = fam_mm_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Suppress pymmseqs' hardcoded print output
                with redirect_stdout(io.StringIO()):
                    easy_cluster(
                        fasta_files=str(family_fa),
                        cluster_prefix=str(cluster_prefix),
                        tmp_dir=str(tmp_dir),
                        min_seq_id=min_seq_id,
                    )
            except Exception as e:
                console.print(f"[red]MMseqs easy-cluster failed for {safe}: {e}[/]")
                failures.append((str(family_fa), str(cluster_prefix), str(tmp_dir)))

            progress.advance(task)

    if failures:
        console.print(f"\n[red]Failed:[/] {len(failures)} families")
        for fasta, out, tmp in failures:
            console.print(
                f"  mmseqs easy-cluster {fasta} {out} {tmp} --min-seq-id {min_seq_id}"
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
        n_splits=1, test_size=0.30, random_state=42
    )
    train_idx, valtest_idx = next(msss1.split(df, Y))
    train_df, df_valtest = df.iloc[train_idx], df.iloc[valtest_idx]
    Y_valtest = Y[valtest_idx]

    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=42
    )
    val_idx, test_idx = next(msss2.split(df_valtest, Y_valtest))
    train_df = train_df.reset_index(drop=True)
    val_df = df_valtest.iloc[val_idx].reset_index(drop=True)
    test_df = df_valtest.iloc[test_idx].reset_index(drop=True)

    for subset in (train_df, val_df, test_df):
        subset["Protein families"] = subset["fam_list"].apply(",".join)
        subset.drop(columns="fam_list", inplace=True)
    return train_df, val_df, test_df


def build_train_all_members(data: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    mmseqs_dir = intermediate_dir() / "mmseqs"
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
    run_signalp6: bool = True,
    signalp6_extra: str = "--organism euk",
    min_seq_id: float = 0.9,
) -> None:
    """Run the full preprocessing pipeline."""
    interm = intermediate_dir()
    fasta_dir = interm / "fasta"
    rep_dir = interm / "representatives"
    proc = processed_dir()
    sp6_tox_dir = interm / "sp6" / "tox"
    sp6_nontox_dir = interm / "sp6" / "nontox"
    bench_dir = get_project_root() / "benchmark"
    bench_hbi_dir = bench_dir / "HBI"

    ensure_dirs()

    # -- Step 1: Load raw data --
    console.print("\n[bold]1.[/] Loading raw data")
    tox, nontox = load_and_prepare_raw()
    n_families = tox["Protein families"].nunique()
    console.print(
        f"   {len(tox)} toxin sequences ({n_families} families), "
        f"{len(nontox)} non-toxin sequences"
    )

    write_fasta(tox, fasta_dir / "tox.fasta")
    write_fasta(nontox, fasta_dir / "nontox.fasta")

    # -- Step 2: SignalP6 --
    console.print("\n[bold]2.[/] SignalP6 signal peptide removal")
    has_sp6 = (sp6_tox_dir / "output.gff3").exists() and (
        sp6_nontox_dir / "output.gff3"
    ).exists()
    if run_signalp6 and not has_sp6:
        maybe_run_signalp6(signalp6_extra)
        has_sp6 = (sp6_tox_dir / "output.gff3").exists() and (
            sp6_nontox_dir / "output.gff3"
        ).exists()
    elif not run_signalp6 and not has_sp6:
        console.print("   Skipped (use --run-signalp6 to enable)")

    if has_sp6:
        tox, nontox = apply_signalp_filtered_sequences(tox, nontox)
        console.print("   Applied signal peptide removal")
    else:
        console.print("   No SignalP6 output found, using raw sequences")

    nontox["Protein families"] = "nontox"
    data = pd.concat([tox, nontox], ignore_index=True)

    write_fasta(tox, fasta_dir / "tox_noSP.fasta")
    write_fasta(nontox, fasta_dir / "nontox_noSP.fasta")

    # -- Step 3: MMseqs2 clustering --
    n_families_total = data["Protein families"].nunique()
    console.print(
        f"\n[bold]3.[/] MMseqs2 clustering "
        f"({n_families_total} families, min_seq_id={min_seq_id})"
    )
    rep_df_all, rep_df_tox = cluster_per_family_and_collect(data, min_seq_id=min_seq_id)
    console.print(
        f"   {len(rep_df_all)} representative sequences "
        f"({len(rep_df_tox)} toxin, {len(rep_df_all) - len(rep_df_tox)} non-toxin)"
    )

    rep_df_tox[["identifier", "Protein families"]].to_csv(
        rep_dir / "tox.csv", index=False
    )
    rep_df_all[["identifier", "Protein families"]].to_csv(
        rep_dir / "all.csv", index=False
    )
    write_fasta(rep_df_tox, rep_dir / "tox.fasta")
    write_fasta(rep_df_all, rep_dir / "all.fasta")

    # -- Step 4: Stratified splits --
    console.print("\n[bold]4.[/] Stratified train/val/test splits")
    train_df, val_df, test_df = multilabel_stratified_splits(rep_df_all)
    train_df["Split"], val_df["Split"], test_df["Split"] = "train", "val", "test"
    training_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    training_data.to_csv(proc / "training_data.csv", index=False)

    train_all_df = build_train_all_members(data, train_df)
    bench_hbi_dir.mkdir(parents=True, exist_ok=True)
    train_all_df.to_csv(bench_hbi_dir / "train_all_df.csv", index=False)
    write_fasta(train_all_df, bench_hbi_dir / "train_all_members.fasta")
    test_df.to_csv(bench_dir / "test_data.csv", index=False)
    write_fasta(test_df, bench_dir / "test_data.fasta")
    val_df.to_csv(bench_dir / "val_data.csv", index=False)
    write_fasta(val_df, bench_dir / "val_data.fasta")

    # -- Summary table --
    console.print()
    table = Table(show_header=True, header_style="bold", padding=(0, 1))
    table.add_column("Split", style="cyan")
    table.add_column("Sequences", justify="right")
    table.add_column("Families", justify="right")
    for name, df in [("train (reps)", train_df), ("val", val_df), ("test", test_df)]:
        table.add_row(name, str(len(df)), str(df["Protein families"].nunique()))
    table.add_row(
        "train (all members)",
        str(len(train_all_df)),
        str(train_all_df["Protein families"].nunique()),
        style="dim",
    )
    console.print(table)
    console.print("\n[bold green]Done.[/]")
