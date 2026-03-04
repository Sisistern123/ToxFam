"""Toxin dataset pipeline: filtering, clustering, stratified splits."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from toxfam.data._fasta import parse_fasta, write_fasta
from toxfam.data.normalization import normalize_protein_families
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


def fasta_to_dataframe(fasta_file: os.PathLike | str) -> pd.DataFrame:
    records = parse_fasta(fasta_file)
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

    tox = normalize_protein_families(tox)

    nontox = pd.read_csv(raw / "nontox.tsv", sep="\t").copy()
    nontox.rename(columns={"Entry": "identifier"}, inplace=True)
    cutoff = (
        nontox["Sequence"].str.len().nlargest(int(np.ceil(len(nontox) * 0.01))).min()
    )
    nontox = nontox[nontox["Sequence"].str.len() <= cutoff].reset_index(drop=True)
    nontox["Protein families"] = "nontox"

    return tox, nontox


# ---------- SignalP6 per-sequence caching ----------


def _seq_hash(seq: str) -> str:
    """MD5 hash of a protein sequence."""
    return hashlib.md5(seq.encode()).hexdigest()


def _sp6_cache_path() -> Path:
    return intermediate_dir() / "sp6" / "sp6_cache.json"


def _load_sp6_cache() -> Dict[str, str | None]:
    """Load per-sequence SP6 cache. Returns {seq_hash: mature_seq or None}."""
    path = _sp6_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_sp6_cache(cache: Dict[str, str | None]) -> None:
    path = _sp6_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f)
    tmp.rename(path)


def _parse_sp6_output(sp6_dir: Path) -> Dict[str, str]:
    """Parse SP6 output → {identifier: mature_sequence} for high-confidence hits."""
    proc_fasta = sp6_dir / "processed_entries.fasta"
    gff_path = sp6_dir / "output.gff3"
    if not proc_fasta.exists() or not gff_path.exists():
        return {}

    df_proc = fasta_to_dataframe(proc_fasta)
    gff_cols = [
        "identifier", "source", "feature_type", "start", "end",
        "score", "strand", "phase", "attributes",
    ]
    df_gff = pd.read_csv(gff_path, sep="\t", comment="#", names=gff_cols)
    df_gff["identifier"] = (
        df_gff["identifier"].str.split("|").str[-1].str.split().str[0]
    )
    merged = pd.merge(df_gff, df_proc, on="identifier")
    hits = merged[merged["score"] > 0.8]
    return dict(zip(hits["identifier"], hits["Sequence"]))


def _bootstrap_sp6_cache(
    tox: pd.DataFrame, nontox: pd.DataFrame,
) -> Dict[str, str | None]:
    """Build cache from existing monolithic SP6 output files."""
    cache: Dict[str, str | None] = {}
    sp6_base = intermediate_dir() / "sp6"

    for label, df in [("tox", tox), ("nontox", nontox)]:
        sp_hits = _parse_sp6_output(sp6_base / label)
        for _, row in df.iterrows():
            h = _seq_hash(row["Sequence"])
            cache[h] = sp_hits.get(row["identifier"])
    return cache


def _run_signalp6_batch(
    df: pd.DataFrame, extra_args: str,
) -> Dict[str, str | None]:
    """Run SP6 on a batch of sequences → {seq_hash: mature_seq or None}."""
    if df.empty:
        return {}

    sp6_project = get_project_root() / "tools" / "signalp6"
    model_dir = sp6_project / "bin" / "signalp-6-package" / "models"

    mode = (
        "slow-sequential"
        if (model_dir / "sequential_models_signalp6").is_dir()
        else "fast"
    )

    batch_dir = intermediate_dir() / "sp6" / "_batch"
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    batch_dir.mkdir(parents=True)

    tmp_fasta = batch_dir / "input.fasta"
    write_fasta(df, tmp_fasta)

    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    cmd = [
        "uv", "run", "--quiet", "--project", str(sp6_project),
        "signalp6",
        "--fastafile", str(tmp_fasta),
        "--output_dir", str(batch_dir),
        "--model_dir", str(model_dir),
        *extra_args.split(),
        "--mode", mode,
        "--bsize", "10",
        "--format", "none",
    ]

    try:
        subprocess.run(cmd, check=True, env=env, capture_output=True)
    except subprocess.CalledProcessError as e:
        console.print(f"   [yellow]SP6 batch failed (exit {e.returncode})[/]")
        shutil.rmtree(batch_dir, ignore_errors=True)
        return {}

    sp_hits = _parse_sp6_output(batch_dir)
    result: Dict[str, str | None] = {}
    for _, row in df.iterrows():
        h = _seq_hash(row["Sequence"])
        result[h] = sp_hits.get(row["identifier"])

    shutil.rmtree(batch_dir, ignore_errors=True)
    return result


def run_signalp6_step(
    tox: pd.DataFrame,
    nontox: pd.DataFrame,
    extra_args: str = "--organism euk",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply SignalP6 signal peptide removal with per-sequence caching.

    Each sequence is cached by its MD5 hash. Only sequences not in the cache
    are sent to SignalP6. On first run, the cache is bootstrapped from any
    existing monolithic SP6 output files.
    """
    sp6_project = get_project_root() / "tools" / "signalp6"
    if not (sp6_project / "bin" / "signalp-6-package").exists():
        raise RuntimeError(
            "SignalP6 not installed. See docs/signalp6_setup.md for setup instructions."
        )

    cache = _load_sp6_cache()
    if not cache:
        bootstrapped = _bootstrap_sp6_cache(tox, nontox)
        if bootstrapped:
            cache = bootstrapped
            _save_sp6_cache(cache)
            console.print(
                f"   Bootstrapped cache from existing SP6 output ({len(cache)} seqs)"
            )

    # Find uncached sequences
    all_df = pd.concat([tox, nontox], ignore_index=True)
    hashes = all_df["Sequence"].apply(_seq_hash)
    uncached_mask = ~hashes.isin(cache)
    n_uncached = int(uncached_mask.sum())

    if n_uncached == 0:
        console.print(f"   All {len(all_df)} sequences cached")
    else:
        console.print(f"   Running SP6 on {n_uncached} uncached sequences ...")
        uncached_df = all_df[uncached_mask].drop_duplicates(subset="Sequence")
        new_results = _run_signalp6_batch(uncached_df, extra_args)
        cache.update(new_results)
        _save_sp6_cache(cache)

    # Apply cached results
    def apply_cache(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        new_seqs = []
        for seq in df["Sequence"]:
            mature = cache.get(_seq_hash(seq))
            new_seqs.append(mature if mature is not None else seq)
        df["Sequence"] = new_seqs
        return df

    return apply_cache(tox), apply_cache(nontox)


# ---------- MMseqs2 & splitting ----------


def cluster_per_family_and_collect(
    data: pd.DataFrame, min_seq_id: float = 0.9
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mmseqs_dir = intermediate_dir() / "mmseqs"
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
            fam_mm_dir = mmseqs_dir / safe
            fam_mm_dir.mkdir(parents=True, exist_ok=True)
            family_fa = fam_mm_dir / "input.fasta"
            write_fasta(group, family_fa)
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
            for rec in parse_fasta(rep_fasta)
        ]
        rep_seqs_all.extend(seqs)
        if family_dir != "nontox":
            rep_seqs_tox.extend(seqs)

    merge_cols = ["identifier", "Protein families"]
    if "Organism (ID)" in data.columns:
        merge_cols.append("Organism (ID)")
    rep_df_all = pd.DataFrame(rep_seqs_all).merge(
        data[merge_cols], on="identifier", how="left"
    )
    rep_df_tox = pd.DataFrame(rep_seqs_tox).merge(
        data[merge_cols], on="identifier", how="left"
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
    signalp6_extra: str = "--organism euk",
    min_seq_id: float = 0.9,
) -> None:
    """Run the full preprocessing pipeline."""
    interm = intermediate_dir()
    fasta_dir = interm / "fasta"
    rep_dir = interm / "mmseqs" / "representatives"
    proc = processed_dir()
    bench_dir = get_project_root() / "benchmark"
    bench_hbi_dir = bench_dir / "HBI"

    # -- Step 0: Download raw data if missing --
    raw = raw_dir()
    if not (raw / "0800.tsv").exists() or not (raw / "nontox.tsv").exists():
        console.print("\n[bold]0.[/] Downloading raw data from UniProt")
        script = get_project_root() / "scripts" / "download_raw_data.sh"
        subprocess.run(["bash", str(script)], check=True)

    # -- Step 1: Load raw data --
    console.print("\n[bold]1.[/] Loading raw data")
    tox, nontox = load_and_prepare_raw()
    n_families = tox["Protein families"].nunique()
    console.print(
        f"   {len(tox)} toxin sequences ({n_families} families), "
        f"{len(nontox)} non-toxin sequences"
    )

    fasta_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(tox, fasta_dir / "tox.fasta")
    write_fasta(nontox, fasta_dir / "nontox.fasta")

    # -- Step 2: SignalP6 --
    console.print("\n[bold]2.[/] SignalP6 signal peptide removal")
    tox, nontox = run_signalp6_step(tox, nontox, signalp6_extra)

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

    rep_dir.mkdir(parents=True, exist_ok=True)
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
    proc.mkdir(parents=True, exist_ok=True)
    training_data.to_csv(proc / "training_data.csv", index=False)

    train_all_df = build_train_all_members(data, train_df)
    bench_hbi_dir.mkdir(parents=True, exist_ok=True)
    train_all_df.to_csv(bench_hbi_dir / "train_all_df.csv", index=False)
    write_fasta(train_all_df, bench_hbi_dir / "train_all_members.fasta")
    bench_dir.mkdir(parents=True, exist_ok=True)
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
