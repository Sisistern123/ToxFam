"""Hierarchical training data assembly pipeline.

Merges toxin sources (new XML + old TSV), finds non-toxic homologs
(name matching + MMseqs2 search), re-clusters combined dataset, and
produces hierarchical_training_data.csv.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Tuple

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from pymmseqs.commands import easy_cluster
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table
from sklearn.preprocessing import MultiLabelBinarizer

from toxfam._paths import intermediate_dir, processed_dir, raw_dir
from toxfam.data._fasta import parse_fasta, write_fasta
from toxfam.data.normalization import normalize_protein_families
from toxfam.data.preprocessing import (
    identity_aware_splits,
    run_signalp6_step,
    sanitize_filename,
)
from toxfam.data.xml_parser import parse_uniprot_xml

console = Console()


# ---------- Step 1: Gather all toxin proteins ----------


def _load_old_tox() -> pd.DataFrame:
    """Load old toxin TSV (0800.tsv), rename columns."""
    tox = (
        pd.read_csv(raw_dir() / "0800.tsv", sep="\t")
        .dropna(subset=["Protein families"])
        .copy()
    )
    tox.rename(columns={"Entry": "identifier"}, inplace=True)
    tox["is_toxic"] = True
    return tox


def _load_new_tox(xml_path: Path) -> pd.DataFrame:
    """Parse new UniProt XML and mark as toxic."""
    df = parse_uniprot_xml(xml_path)
    # Keep only proteins with a family annotation
    df = df[df["Protein families"] != ""].copy()
    df["is_toxic"] = True
    return df


def merge_toxin_sources(xml_path: Path) -> pd.DataFrame:
    """Merge old TSV toxins and new XML toxins, deduplicating by identifier.

    Prefers new XML records for overlapping identifiers (richer annotations).
    """
    old_tox = _load_old_tox()
    new_tox = _load_new_tox(xml_path)

    # Keep required columns from both
    keep_cols = ["identifier", "Sequence", "Protein families", "Organism (ID)", "is_toxic"]
    old_cols = [c for c in keep_cols if c in old_tox.columns]
    new_cols = [c for c in keep_cols if c in new_tox.columns]
    old_tox = old_tox[old_cols]
    new_tox = new_tox[new_cols]

    # Prefer new XML records for overlapping IDs
    old_only = old_tox[~old_tox["identifier"].isin(new_tox["identifier"])]
    combined = pd.concat([new_tox, old_only], ignore_index=True)
    combined = combined.drop_duplicates(subset="identifier").reset_index(drop=True)

    return combined


# ---------- Step 2 & 3: Find non-toxic homologs ----------


def _load_nontox_with_families() -> pd.DataFrame:
    """Load nontox.tsv with ORIGINAL family annotations (not overwritten to 'nontox')."""
    nontox = pd.read_csv(raw_dir() / "nontox.tsv", sep="\t").copy()
    nontox.rename(columns={"Entry": "identifier"}, inplace=True)
    nontox["is_toxic"] = False
    return nontox


def find_nontox_by_name_matching(
    tox_df: pd.DataFrame,
    *,
    max_nontox_per_family: int = 200,
) -> pd.DataFrame:
    """Find non-toxic homologs by direct family name matching.

    After normalizing both tox and nontox family labels, proteins from nontox.tsv
    whose normalized family name matches a toxin family are paired as non-toxic members.
    """
    nontox_raw = _load_nontox_with_families()

    # Normalize family names on both sides
    tox_normalized = normalize_protein_families(tox_df, min_count=1)
    nontox_normalized = normalize_protein_families(nontox_raw, min_count=1)

    tox_families = set(tox_normalized["Protein families"].unique()) - {"other", ""}

    # Find nontox proteins with matching family names
    matched = nontox_normalized[
        nontox_normalized["Protein families"].isin(tox_families)
    ].copy()

    # Use normalized family names
    matched["Protein families"] = nontox_normalized.loc[
        matched.index, "Protein families"
    ]

    # Cap per family to prevent dominance
    if max_nontox_per_family > 0:
        matched = (
            matched.groupby("Protein families")
            .apply(
                lambda g: g.sample(
                    n=min(len(g), max_nontox_per_family), random_state=42
                ),
                include_groups=False,
            )
            .reset_index(level=0, drop=True)
            .reset_index(drop=True)
        )

    keep_cols = ["identifier", "Sequence", "Protein families", "Organism (ID)", "is_toxic"]
    available = [c for c in keep_cols if c in matched.columns]
    return matched[available]


def find_nontox_by_mmseqs2_search(
    tox_df: pd.DataFrame,
    already_matched_ids: set[str],
    *,
    max_nontox_per_family: int = 200,
    min_seq_id: float = 0.3,
    evalue: float = 1e-5,
) -> pd.DataFrame:
    """Find non-toxic homologs via MMseqs2 sequence search.

    For toxin families with insufficient name-matched nontox members,
    searches all remaining nontox proteins by sequence similarity.
    """
    nontox_raw = _load_nontox_with_families()

    # Exclude already-matched nontox proteins
    remaining = nontox_raw[~nontox_raw["identifier"].isin(already_matched_ids)]
    if remaining.empty:
        return pd.DataFrame()

    with tempfile.TemporaryDirectory(prefix="toxfam_search_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Write toxin query FASTA (one representative per family — first occurrence)
        tox_reps = tox_df.drop_duplicates(subset="Protein families")
        query_fasta = tmpdir / "query.fasta"
        write_fasta(tox_reps, query_fasta)

        # Write nontox target FASTA
        target_fasta = tmpdir / "target.fasta"
        write_fasta(remaining, target_fasta)

        # Run MMseqs2 easy-search
        result_tsv = tmpdir / "results.tsv"
        cmd = [
            "mmseqs", "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(result_tsv),
            str(tmpdir / "tmp"),
            "--min-seq-id", str(min_seq_id),
            "-e", str(evalue),
            "--format-output", "query,target,evalue,bits",
            "--threads", "4",
            "-v", "2",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            console.print(f"[yellow]MMseqs2 easy-search warning: {proc.stderr[:200]}[/]")
            return pd.DataFrame()

        if not result_tsv.exists() or result_tsv.stat().st_size == 0:
            return pd.DataFrame()

        # Parse search results
        hits = pd.read_csv(
            result_tsv, sep="\t", header=None,
            names=["query", "target", "evalue", "bits"],
        )

    if hits.empty:
        return pd.DataFrame()

    # Map query IDs back to family names
    id_to_family = dict(zip(tox_reps["identifier"], tox_reps["Protein families"]))
    hits["family"] = hits["query"].map(id_to_family)

    # For each nontox hit, assign to the best (lowest evalue) matching family
    best_hits = hits.sort_values("evalue").drop_duplicates(subset="target", keep="first")

    # Build result DataFrame
    nontox_lookup = remaining.set_index("identifier")
    records = []
    for _, hit in best_hits.iterrows():
        target_id = hit["target"]
        if target_id not in nontox_lookup.index:
            continue
        nontox_row = nontox_lookup.loc[target_id]
        records.append({
            "identifier": target_id,
            "Sequence": nontox_row.get("Sequence", ""),
            "Protein families": hit["family"],
            "Organism (ID)": nontox_row.get("Organism (ID)", ""),
            "is_toxic": False,
        })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)

    # Cap per family
    if max_nontox_per_family > 0:
        result = (
            result.groupby("Protein families")
            .apply(
                lambda g: g.sample(
                    n=min(len(g), max_nontox_per_family), random_state=42
                ),
                include_groups=False,
            )
            .reset_index(level=0, drop=True)
            .reset_index(drop=True)
        )

    return result


# ---------- Step 5: Re-cluster combined dataset ----------


def cluster_combined_dataset(
    data: pd.DataFrame,
    *,
    min_seq_id: float = 0.9,
    family_min_count: int = 10,
) -> pd.DataFrame:
    """Cluster per family (tox + nontox together) at min_seq_id identity.

    Returns a DataFrame of cluster representatives with preserved is_toxic flags.
    Families with <family_min_count representatives are collapsed.
    """
    mmseqs_dir = intermediate_dir() / "hierarchical" / "mmseqs"
    mmseqs_dir.mkdir(parents=True, exist_ok=True)

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
            progress.update(task, description=f"Clustering [cyan]{safe}[/]", refresh=True)

            fam_dir = mmseqs_dir / safe
            fam_dir.mkdir(parents=True, exist_ok=True)
            family_fa = fam_dir / "input.fasta"
            rep_fasta = fam_dir / "cluster_rep_seq.fasta"

            write_fasta(group, family_fa)

            cluster_prefix = fam_dir / "cluster"
            tmp_dir = fam_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            try:
                with redirect_stdout(io.StringIO()):
                    easy_cluster(
                        fasta_files=str(family_fa),
                        cluster_prefix=str(cluster_prefix),
                        tmp_dir=str(tmp_dir),
                        min_seq_id=min_seq_id,
                    )
            except Exception as e:
                console.print(f"[red]MMseqs easy-cluster failed for {safe}: {e}[/]")

            progress.advance(task)

    # Collect representatives
    rep_seqs = []
    for family_dir in sorted(mmseqs_dir.iterdir()):
        rep_fasta = family_dir / "cluster_rep_seq.fasta"
        if not rep_fasta.exists():
            continue
        for rec in parse_fasta(rep_fasta):
            rep_seqs.append({"identifier": rec.id, "Sequence": str(rec.seq)})

    if not rep_seqs:
        raise RuntimeError("No representative sequences collected after clustering")

    rep_df = pd.DataFrame(rep_seqs)

    # Merge back metadata (family, is_toxic, organism)
    merge_cols = ["identifier", "Protein families", "is_toxic"]
    if "Organism (ID)" in data.columns:
        merge_cols.append("Organism (ID)")

    rep_df = rep_df.merge(
        data[merge_cols].drop_duplicates(subset="identifier"),
        on="identifier",
        how="left",
    )

    # Collapse small families
    fam_counts = rep_df["Protein families"].value_counts()
    small = set(fam_counts[fam_counts < family_min_count].index) - {"other"}
    if small:
        # Toxic members of small families → "other_toxin", nontox → drop
        mask_small = rep_df["Protein families"].isin(small)
        rep_df.loc[mask_small & rep_df["is_toxic"], "Protein families"] = "other_toxin"
        rep_df = rep_df[~(mask_small & ~rep_df["is_toxic"])].reset_index(drop=True)

    return rep_df


# ---------- Step 7: Stratified splits ----------


def hierarchical_stratified_splits(
    rep_df: pd.DataFrame,
    *,
    base_seq_id: float = 0.3,
    use_identity_clustering: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified splits, optionally using identity-aware clustering.

    70% train / 15% val / 15% test.

    Args:
        rep_df: Representative DataFrame with identifier, Sequence, Protein families.
        base_seq_id: Identity threshold for cluster-then-split.
        use_identity_clustering: If True, use identity_aware_splits with MMseqs2.
            If False, fall back to random stratified splitting.
    """
    if use_identity_clustering:
        return identity_aware_splits(rep_df, base_seq_id=base_seq_id)

    # Fallback: random multilabel stratified split (original behavior)
    df = rep_df.copy()
    df["_strat_label"] = (
        df["Protein families"] + "__" + df.get("is_toxic", pd.Series(True, index=df.index)).astype(str)
    )
    df["_label_list"] = df["_strat_label"].apply(lambda x: x.split(",") if "," in x else [x])

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["_label_list"])

    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=42
    )
    train_idx, valtest_idx = next(msss1.split(df, Y))
    train_df = df.iloc[train_idx]
    df_valtest = df.iloc[valtest_idx]
    Y_valtest = Y[valtest_idx]

    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=42
    )
    val_idx, test_idx = next(msss2.split(df_valtest, Y_valtest))

    train_df = train_df.reset_index(drop=True)
    val_df = df_valtest.iloc[val_idx].reset_index(drop=True)
    test_df = df_valtest.iloc[test_idx].reset_index(drop=True)

    for subset in (train_df, val_df, test_df):
        subset.drop(columns=["_strat_label", "_label_list"], inplace=True)

    return train_df, val_df, test_df


# ---------- Main pipeline ----------


def run_hierarchical_preprocessing(
    *,
    xml_path: Path | None = None,
    signalp6_extra: str = "--organism euk",
    min_seq_id: float = 0.9,
    family_min_count: int = 10,
    max_nontox_per_family: int = 200,
    skip_signalp6: bool = False,
    skip_mmseqs2_search: bool = False,
) -> Path:
    """Run the full hierarchical data assembly pipeline.

    Returns the path to the output CSV.
    """
    if xml_path is None:
        xml_path = Path(
            "/Users/jcoludar/CascadeProjects/SpeciesEmbedding/data/ToxProtFeb2026/"
            "uniprotkb_taxonomy_id_33208_AND_cc_tiss_2026_02_20.xml"
        )

    hier_dir = intermediate_dir() / "hierarchical"
    hier_dir.mkdir(parents=True, exist_ok=True)

    # -- Step 1: Merge toxin sources --
    console.print("\n[bold]1.[/] Merging toxin sources (XML + old TSV)")
    combined_tox = merge_toxin_sources(xml_path)
    console.print(f"   {len(combined_tox)} unique toxin proteins")

    # -- Step 2: Normalize family labels --
    console.print("\n[bold]2.[/] Normalizing family labels")
    combined_tox = normalize_protein_families(combined_tox, min_count=1)
    n_families = combined_tox["Protein families"].nunique()
    console.print(f"   {n_families} normalized families")

    # -- Step 3: Find non-toxic homologs --
    console.print("\n[bold]3.[/] Finding non-toxic homologs")

    # Prong A: Name matching
    console.print("   [cyan]3a.[/] Name matching against nontox.tsv")
    nontox_named = find_nontox_by_name_matching(
        combined_tox, max_nontox_per_family=max_nontox_per_family
    )
    matched_families = nontox_named["Protein families"].nunique() if len(nontox_named) > 0 else 0
    console.print(
        f"   Found {len(nontox_named)} nontox proteins in {matched_families} matching families"
    )

    # Prong B: MMseqs2 search (optional)
    nontox_searched = pd.DataFrame()
    if not skip_mmseqs2_search:
        console.print("   [cyan]3b.[/] MMseqs2 sequence search for additional homologs")
        already_matched = set(nontox_named["identifier"]) if len(nontox_named) > 0 else set()
        nontox_searched = find_nontox_by_mmseqs2_search(
            combined_tox,
            already_matched,
            max_nontox_per_family=max_nontox_per_family,
        )
        if len(nontox_searched) > 0:
            console.print(
                f"   Found {len(nontox_searched)} additional nontox via search "
                f"in {nontox_searched['Protein families'].nunique()} families"
            )
        else:
            console.print("   No additional nontox found via search")

    # Combine all
    all_nontox = pd.concat(
        [df for df in [nontox_named, nontox_searched] if len(df) > 0],
        ignore_index=True,
    )
    if len(all_nontox) > 0:
        all_nontox = all_nontox.drop_duplicates(subset="identifier").reset_index(drop=True)

    data = pd.concat([combined_tox, all_nontox], ignore_index=True)
    data = data.drop_duplicates(subset="identifier").reset_index(drop=True)
    n_tox = data["is_toxic"].sum()
    n_nontox = len(data) - n_tox
    console.print(f"   Combined: {len(data)} proteins ({n_tox} toxic, {n_nontox} non-toxic)")

    # -- Step 4: SignalP6 signal peptide removal --
    if not skip_signalp6:
        console.print("\n[bold]4.[/] SignalP6 signal peptide removal")
        tox_part = data[data["is_toxic"]].copy()
        nontox_part = data[~data["is_toxic"]].copy()
        tox_processed, nontox_processed = run_signalp6_step(
            tox_part, nontox_part, signalp6_extra
        )
        # Reassemble with processed sequences
        data = pd.concat([tox_processed, nontox_processed], ignore_index=True)
    else:
        console.print("\n[bold]4.[/] SignalP6 skipped")

    # -- Step 5: Re-cluster combined dataset --
    console.print(f"\n[bold]5.[/] Re-clustering at {min_seq_id*100:.0f}% identity")
    rep_df = cluster_combined_dataset(
        data, min_seq_id=min_seq_id, family_min_count=family_min_count
    )
    n_rep_tox = rep_df["is_toxic"].sum()
    n_rep_nontox = len(rep_df) - n_rep_tox
    console.print(
        f"   {len(rep_df)} representative sequences "
        f"({n_rep_tox} toxic, {n_rep_nontox} non-toxic)"
    )

    # -- Step 6: Label validation (optional — can be slow) --
    console.print("\n[bold]6.[/] Label validation skipped (run manually if needed)")

    # -- Step 7: Stratified splits --
    console.print("\n[bold]7.[/] Stratified train/val/test splits")
    train_df, val_df, test_df = hierarchical_stratified_splits(rep_df)
    train_df["Split"] = "train"
    val_df["Split"] = "val"
    test_df["Split"] = "test"
    training_data = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # -- Step 8: Write output --
    output_path = processed_dir() / "hierarchical_training_data.csv"
    processed_dir().mkdir(parents=True, exist_ok=True)
    output_cols = ["identifier", "Sequence", "Protein families", "is_toxic", "Split"]
    if "Organism (ID)" in training_data.columns:
        output_cols.append("Organism (ID)")
    training_data[output_cols].to_csv(output_path, index=False)

    # -- Summary --
    console.print()
    table = Table(show_header=True, header_style="bold", padding=(0, 1))
    table.add_column("Split", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Toxic", justify="right")
    table.add_column("Non-toxic", justify="right")
    table.add_column("Families", justify="right")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        table.add_row(
            name,
            str(len(df)),
            str(int(df["is_toxic"].sum())),
            str(int((~df["is_toxic"]).sum())),
            str(df["Protein families"].nunique()),
        )
    console.print(table)
    console.print(f"\n[bold green]Done.[/] Output: {output_path}")

    return output_path
