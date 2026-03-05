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


def _remove_nontox_contamination(nontox: pd.DataFrame) -> pd.DataFrame:
    """Remove venom/toxin protein contamination from the nontox dataset.

    The nontox.tsv contains ~375 entries with "venom" or "toxin" in their
    UniProt family name — these are actual venom proteins that were incorrectly
    included. Removing them prevents the model from learning that venom
    proteins are non-toxic.
    """
    fam_col = "Protein families"
    if fam_col not in nontox.columns:
        return nontox

    original_n = len(nontox)
    venom_mask = nontox[fam_col].str.contains(
        r"venom|toxin|sarafotoxin", case=False, na=False
    )
    n_removed = venom_mask.sum()
    nontox = nontox[~venom_mask].reset_index(drop=True)
    if n_removed > 0:
        console.print(
            f"   Removed {n_removed} contaminated entries from nontox "
            f"(venom/toxin family names), {original_n} → {len(nontox)}"
        )
    return nontox


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
    nontox = _remove_nontox_contamination(nontox)
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
        sp6_project = get_project_root() / "tools" / "signalp6"
        if not (sp6_project / "bin" / "signalp-6-package").exists():
            raise RuntimeError(
                "SignalP6 not installed. See docs/signalp6_setup.md for setup instructions."
            )
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
            rep_fasta = fam_mm_dir / "cluster_rep_seq.fasta"
            old_hash = hashlib.md5(family_fa.read_bytes()).hexdigest() if family_fa.exists() else None
            write_fasta(group, family_fa)
            new_hash = hashlib.md5(family_fa.read_bytes()).hexdigest()

            if old_hash == new_hash and rep_fasta.exists():
                progress.advance(task)
                continue

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


def _rebalance_splits(
    df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    train_cids: set,
    val_cids: set,
    test_cids: set,
    *,
    min_train_frac: float = 0.50,
) -> Tuple[set, set, set]:
    """Move clusters between splits to ensure minimum family representation in train.

    For each family where train fraction < min_train_frac:
    1. Identify clusters in val/test that contain that family
    2. Move the smallest such cluster to train
    3. Repeat until train_frac >= min_train_frac or no more moveable clusters

    Constraint: never leave val or test empty.
    """
    train_cids = set(train_cids)
    val_cids = set(val_cids)
    test_cids = set(test_cids)

    # Build cluster membership lookup: cid -> set of families
    cid_to_families: Dict[int, set] = {}
    cid_to_size: Dict[int, int] = {}
    for _, row in cluster_df.iterrows():
        cid = row["_cluster_id"]
        cid_to_families[cid] = row["families"]
        cid_to_size[cid] = row["size"]

    all_families = set(df["Protein families"].unique())
    moved = 0

    for fam in sorted(all_families):
        fam_mask = df["Protein families"] == fam
        total_fam = fam_mask.sum()
        if total_fam < 10:
            continue

        for _ in range(20):  # safety limit
            train_fam = (fam_mask & df["_cluster_id"].isin(train_cids)).sum()
            if total_fam == 0 or train_fam / total_fam >= min_train_frac:
                break

            # Find smallest moveable cluster in val or test containing this family
            candidates = []
            for source, source_cids in [("val", val_cids), ("test", test_cids)]:
                if len(source_cids) <= 1:
                    continue  # never empty a split
                for cid in source_cids:
                    if fam in cid_to_families.get(cid, set()):
                        candidates.append((cid_to_size.get(cid, 0), cid, source))

            if not candidates:
                break

            candidates.sort()
            _, move_cid, source = candidates[0]

            train_cids.add(move_cid)
            if source == "val":
                val_cids.discard(move_cid)
            else:
                test_cids.discard(move_cid)

            # Update _cluster_id -> split in df for subsequent checks
            df.loc[df["_cluster_id"] == move_cid, "_split"] = "train"
            moved += 1

    if moved > 0:
        console.print(f"   Rebalancing: moved {moved} cluster(s) to train")

    return train_cids, val_cids, test_cids


def _log_split_quality(df: pd.DataFrame) -> None:
    """Log split quality metrics for families."""
    all_families = sorted(df["Protein families"].unique())
    low_train = []
    missing_splits = []

    for fam in all_families:
        fam_df = df[df["Protein families"] == fam]
        split_counts = fam_df["_split"].value_counts()
        total = len(fam_df)
        train_count = split_counts.get("train", 0)
        val_count = split_counts.get("val", 0)
        test_count = split_counts.get("test", 0)

        if train_count < 5 and total >= 10:
            low_train.append((fam, train_count, total))
        if val_count == 0 or test_count == 0:
            missing_splits.append((fam, val_count, test_count, total))

    if low_train:
        console.print(f"   [yellow]Families with <5 train samples ({len(low_train)}):[/]")
        for fam, tc, tot in low_train[:10]:
            console.print(f"     {fam}: {tc}/{tot} in train")
    if missing_splits:
        console.print(
            f"   [yellow]Families missing val or test ({len(missing_splits)}):[/]"
        )
        for fam, vc, tc, tot in missing_splits[:10]:
            console.print(f"     {fam}: val={vc}, test={tc}, total={tot}")
    if not low_train and not missing_splits:
        console.print("   [green]Split quality: all families well-represented[/]")


def identity_aware_splits(
    rep_df_all: pd.DataFrame,
    *,
    base_seq_id: float = 0.3,
    relaxed_thresholds: list[float] | None = None,
    train_ratio: float = 0.70,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split representatives with adaptive identity thresholds.

    1. Cluster all reps at base_seq_id (default 30%).
    2. Assign whole clusters to splits via multilabel stratified splitting.
    3. For families in only one split: relax threshold until splittable.
    """
    import tempfile

    if relaxed_thresholds is None:
        relaxed_thresholds = [0.4, 0.5, 0.6, 0.7]

    df = rep_df_all.copy()
    fasta_dir = intermediate_dir() / "identity_splits"
    fasta_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Write all reps to a single FASTA and cluster at base_seq_id
    all_fasta = fasta_dir / "all_reps.fasta"
    write_fasta(df, all_fasta)

    console.print(f"   Clustering all reps at {base_seq_id*100:.0f}% identity ...")
    cluster_prefix = fasta_dir / "global_cluster"
    tmp_dir = fasta_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with redirect_stdout(io.StringIO()):
            easy_cluster(
                fasta_files=str(all_fasta),
                cluster_prefix=str(cluster_prefix),
                tmp_dir=str(tmp_dir),
                min_seq_id=base_seq_id,
            )
    except Exception as e:
        console.print(f"[red]Global clustering failed: {e}[/]")
        console.print("[yellow]Falling back to random stratified splits[/]")
        return multilabel_stratified_splits(rep_df_all)

    # Parse cluster assignments: {representative: [members]}
    cluster_tsv = fasta_dir / "global_cluster_cluster.tsv"
    rep_to_cluster: Dict[str, int] = {}
    cluster_id = 0
    cluster_map: Dict[str, int] = {}  # representative -> cluster_id
    with open(cluster_tsv) as f:
        for line in f:
            rep, member = line.strip().split("\t")
            if rep not in cluster_map:
                cluster_map[rep] = cluster_id
                cluster_id += 1
            rep_to_cluster[member] = cluster_map[rep]

    df["_cluster_id"] = df["identifier"].map(rep_to_cluster)
    # Proteins that didn't appear in TSV get their own cluster
    max_cid = df["_cluster_id"].max() if df["_cluster_id"].notna().any() else -1
    missing_mask = df["_cluster_id"].isna()
    if missing_mask.any():
        df.loc[missing_mask, "_cluster_id"] = range(
            int(max_cid) + 1, int(max_cid) + 1 + int(missing_mask.sum())
        )
    df["_cluster_id"] = df["_cluster_id"].astype(int)

    # Step 2: Build a cluster-level DataFrame for stratified splitting
    cluster_groups = df.groupby("_cluster_id")
    cluster_df = pd.DataFrame(
        {
            "_cluster_id": list(cluster_groups.groups.keys()),
            "families": [
                set(g["Protein families"].unique())
                for _, g in cluster_groups
            ],
            "size": [len(g) for _, g in cluster_groups],
        }
    )
    cluster_df["fam_list"] = cluster_df["families"].apply(
        lambda s: list(s)
    )

    mlb = MultiLabelBinarizer()
    Y_clusters = mlb.fit_transform(cluster_df["fam_list"])

    # Guard: if too few clusters, fall back to random stratified splits
    if len(cluster_df) < 4:
        console.print(
            f"   [yellow]Only {len(cluster_df)} clusters — falling back to "
            f"random stratified splits[/]"
        )
        return multilabel_stratified_splits(rep_df_all)

    # Stratified split at cluster level
    val_test_ratio = 1.0 - train_ratio
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_test_ratio, random_state=42
    )
    train_cidx, valtest_cidx = next(msss1.split(cluster_df, Y_clusters))
    Y_valtest = Y_clusters[valtest_cidx]

    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=42
    )
    val_cidx, test_cidx = next(msss2.split(
        cluster_df.iloc[valtest_cidx], Y_valtest
    ))

    train_cluster_ids = set(cluster_df.iloc[train_cidx]["_cluster_id"])
    val_cluster_ids = set(
        cluster_df.iloc[valtest_cidx].iloc[val_cidx]["_cluster_id"]
    )

    # Assign split labels
    def _assign_split(cid: int) -> str:
        if cid in train_cluster_ids:
            return "train"
        if cid in val_cluster_ids:
            return "val"
        return "test"

    df["_split"] = df["_cluster_id"].apply(_assign_split)

    # Step 2b: Rebalance splits to ensure minimum family representation in train
    test_cluster_ids = set(cluster_df["_cluster_id"]) - train_cluster_ids - val_cluster_ids
    train_cluster_ids, val_cluster_ids, test_cluster_ids = _rebalance_splits(
        df, cluster_df, train_cluster_ids, val_cluster_ids, test_cluster_ids,
    )

    # Re-assign split labels after rebalancing
    def _assign_split_rebalanced(cid: int) -> str:
        if cid in train_cluster_ids:
            return "train"
        if cid in val_cluster_ids:
            return "val"
        return "test"

    df["_split"] = df["_cluster_id"].apply(_assign_split_rebalanced)

    # Step 3: Adaptive relaxation for under-represented families
    threshold_log: Dict[str, float] = {}
    all_families = set(df["Protein families"].unique())

    for fam in all_families:
        fam_splits = set(df.loc[df["Protein families"] == fam, "_split"])
        if len(fam_splits) >= 2:
            threshold_log[fam] = base_seq_id
            continue

        # This family is stuck in one split — try relaxing
        fam_members = df[df["Protein families"] == fam]
        if len(fam_members) < 2:
            threshold_log[fam] = base_seq_id
            continue

        resolved = False
        for threshold in relaxed_thresholds:
            with tempfile.TemporaryDirectory(prefix="toxfam_relax_") as tmpd:
                tmpd = Path(tmpd)
                fam_fasta = tmpd / "fam.fasta"
                write_fasta(fam_members, fam_fasta)

                relax_prefix = tmpd / "relax_cluster"
                relax_tmp = tmpd / "tmp"
                relax_tmp.mkdir()

                try:
                    with redirect_stdout(io.StringIO()):
                        easy_cluster(
                            fasta_files=str(fam_fasta),
                            cluster_prefix=str(relax_prefix),
                            tmp_dir=str(relax_tmp),
                            min_seq_id=threshold,
                        )
                except Exception:
                    continue

                relax_tsv = tmpd / "relax_cluster_cluster.tsv"
                if not relax_tsv.exists():
                    continue

                sub_clusters: Dict[str, List[str]] = {}
                with open(relax_tsv) as f:
                    for line in f:
                        rep, member = line.strip().split("\t")
                        sub_clusters.setdefault(rep, []).append(member)

                if len(sub_clusters) >= 2:
                    # Assign sub-clusters to different splits
                    sub_reps = list(sub_clusters.keys())
                    current_split = next(iter(fam_splits))  # don't mutate
                    other_splits = [s for s in ("train", "val", "test") if s != current_split]

                    # Move some sub-clusters to other splits
                    for i, srep in enumerate(sub_reps[1:], 1):
                        target_split = other_splits[(i - 1) % len(other_splits)]
                        member_ids = set(sub_clusters[srep])
                        df.loc[
                            (df["Protein families"] == fam) &
                            (df["identifier"].isin(member_ids)),
                            "_split",
                        ] = target_split

                    threshold_log[fam] = threshold
                    resolved = True
                    break

        if not resolved:
            threshold_log[fam] = base_seq_id

    # Summary
    from collections import Counter
    thresh_counts = Counter(threshold_log.values())
    console.print("   Split threshold summary:")
    for t in sorted(thresh_counts.keys()):
        console.print(f"     {t*100:.0f}%: {thresh_counts[t]} families")

    flagged = {f: t for f, t in threshold_log.items() if t > base_seq_id}
    if flagged:
        console.print(f"   {len(flagged)} families required relaxed thresholds")

    # Log split quality
    _log_split_quality(df)

    # Build output DataFrames
    train_df = df[df["_split"] == "train"].drop(
        columns=["_cluster_id", "_split"]
    ).reset_index(drop=True)
    val_df = df[df["_split"] == "val"].drop(
        columns=["_cluster_id", "_split"]
    ).reset_index(drop=True)
    test_df = df[df["_split"] == "test"].drop(
        columns=["_cluster_id", "_split"]
    ).reset_index(drop=True)

    # Store fam_list column like multilabel_stratified_splits does (for compat)
    for subset in (train_df, val_df, test_df):
        if "fam_list" in subset.columns:
            subset["Protein families"] = subset["fam_list"].apply(",".join)
            subset.drop(columns="fam_list", inplace=True)

    return train_df, val_df, test_df


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
    include_counterparts: bool = False,
) -> None:
    """Run the full preprocessing pipeline."""
    interm = intermediate_dir()
    fasta_dir = interm / "fasta"
    rep_dir = interm / "mmseqs" / "representatives"
    proc = processed_dir()
    bench_dir = get_project_root() / "benchmark"
    bench_hbi_dir = bench_dir / "HBI"

    # -- Step 0: Check raw data --
    raw = raw_dir()
    if not (raw / "0800.tsv").exists() or not (raw / "nontox.tsv").exists():
        console.print(
            "[red]Raw data missing.[/] Run [bold]toxfam download-data[/] first."
        )
        raise SystemExit(1)

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

    # -- Step 2b: Inject counterparts (optional) --
    if include_counterparts:
        counterpart_csv = raw / "nontox_counterparts" / "counterparts.csv"
        if counterpart_csv.exists():
            console.print("\n[bold]2b.[/] Injecting non-toxic counterparts")
            cp_df = pd.read_csv(counterpart_csv)
            cp_df = cp_df[["identifier", "Sequence", "Protein families", "organism_id"]].copy()
            cp_df.rename(columns={"organism_id": "Organism (ID)"}, inplace=True)
            existing_ids = set(data["identifier"])
            new_cp = cp_df[~cp_df["identifier"].isin(existing_ids)]
            console.print(
                f"   {len(cp_df)} counterparts total, {len(cp_df) - len(new_cp)} already present, "
                f"adding {len(new_cp)} new sequences to nontox pool"
            )
            data = pd.concat([data, new_cp], ignore_index=True)
        else:
            console.print(
                "\n[yellow]   Counterpart CSV not found at "
                f"{counterpart_csv}, skipping[/]"
            )

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

    # -- Step 4: Identity-aware stratified splits --
    console.print("\n[bold]4.[/] Identity-aware stratified train/val/test splits")
    train_df, val_df, test_df = identity_aware_splits(rep_df_all, base_seq_id=0.3)
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
