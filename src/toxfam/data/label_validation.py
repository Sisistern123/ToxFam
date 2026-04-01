"""MMseqs2-based family label validation and resolution."""

from __future__ import annotations

import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ClusterInfo:
    """Summary of a single MMseqs2 cluster."""

    representative: str
    members: list[str]
    family_counts: dict[str, int] = field(default_factory=dict)
    dominant_family: str = ""
    is_consistent: bool = True


@dataclass
class ValidationReport:
    """Report from label validation."""

    clusters: list[ClusterInfo]
    inconsistent_clusters: list[ClusterInfo]
    split_families: dict[str, list[int]]  # family -> list of cluster indices
    total_proteins: int = 0
    total_clusters: int = 0


def _run_mmseqs2_cluster(
    fasta_path: str | Path,
    workdir: str | Path,
    min_seq_id: float = 0.4,
    coverage: float = 0.8,
    threads: int = 4,
) -> dict[str, list[str]]:
    """Run MMseqs2 easy-cluster and return {representative: [members]} dict."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tmp_dir = workdir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    output_prefix = str(workdir / "clusters")

    cmd = [
        "mmseqs",
        "easy-cluster",
        str(fasta_path),
        output_prefix,
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        "0",
        "--cluster-mode",
        "0",
        "--threads",
        str(threads),
        "-v",
        "2",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MMseqs2 clustering failed: {result.stderr}")

    cluster_tsv = f"{output_prefix}_cluster.tsv"
    clusters: dict[str, list[str]] = defaultdict(list)
    with open(cluster_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                clusters[parts[0]].append(parts[1])

    return dict(clusters)


def validate_families(
    fasta_path: str | Path,
    family_df: pd.DataFrame,
    *,
    min_seq_id: float = 0.4,
    coverage: float = 0.8,
    threads: int = 4,
    inconsistency_threshold: float = 0.2,
) -> ValidationReport:
    """Validate family labels by clustering with MMseqs2.

    Args:
        fasta_path: Path to FASTA file with all proteins.
        family_df: DataFrame with 'identifier' and 'Protein families' columns.
        min_seq_id: MMseqs2 minimum sequence identity.
        coverage: MMseqs2 minimum coverage.
        threads: Number of threads for MMseqs2.
        inconsistency_threshold: Fraction threshold to flag a cluster as inconsistent.

    Returns:
        ValidationReport with cluster analysis.
    """
    id_to_family = dict(
        zip(family_df["identifier"], family_df["Protein families"], strict=False)
    )

    with tempfile.TemporaryDirectory(prefix="toxfam_validate_") as tmpdir:
        raw_clusters = _run_mmseqs2_cluster(
            fasta_path, tmpdir, min_seq_id=min_seq_id, coverage=coverage, threads=threads
        )

    clusters: list[ClusterInfo] = []
    inconsistent: list[ClusterInfo] = []
    family_to_cluster_indices: dict[str, list[int]] = defaultdict(list)

    for rep, members in raw_clusters.items():
        family_counts: dict[str, int] = Counter()
        for m in members:
            fam = id_to_family.get(m, "")
            if fam:
                family_counts[fam] += 1

        dominant = max(family_counts, key=family_counts.get) if family_counts else ""
        total = sum(family_counts.values())

        # Check consistency: is there a minority label with >threshold representation?
        is_consistent = True
        if total > 0 and len(family_counts) > 1:
            for fam, count in family_counts.items():
                if fam != dominant and count / total >= inconsistency_threshold:
                    is_consistent = False
                    break

        ci = ClusterInfo(
            representative=rep,
            members=members,
            family_counts=dict(family_counts),
            dominant_family=dominant,
            is_consistent=is_consistent,
        )
        idx = len(clusters)
        clusters.append(ci)
        if not is_consistent:
            inconsistent.append(ci)

        for fam in family_counts:
            family_to_cluster_indices[fam].append(idx)

    # Find split families (same family across multiple clusters)
    split_families = {
        fam: indices
        for fam, indices in family_to_cluster_indices.items()
        if len(indices) > 1
    }

    return ValidationReport(
        clusters=clusters,
        inconsistent_clusters=inconsistent,
        split_families=split_families,
        total_proteins=sum(len(c.members) for c in clusters),
        total_clusters=len(clusters),
    )


def resolve_labels(
    report: ValidationReport,
    family_df: pd.DataFrame,
    *,
    embeddings: dict[str, np.ndarray] | None = None,
    min_family_size: int = 10,
) -> pd.DataFrame:
    """Resolve label conflicts using validation report.

    For inconsistent clusters, reassign minority members to the dominant label.
    For families with <min_family_size members after resolution, mark as 'other_toxin'.

    Args:
        report: Output from validate_families().
        family_df: Original DataFrame with 'identifier' and 'Protein families'.
        embeddings: Optional dict of {protein_id: embedding} for centroid merging.
        min_family_size: Minimum members to keep a family as a standalone class.

    Returns:
        New DataFrame with cleaned 'Protein families' column.
    """
    df = family_df.copy()
    id_to_idx = {row["identifier"]: i for i, row in df.iterrows()}

    # Step 1: For inconsistent clusters, reassign minority to dominant
    reassigned = 0
    for cluster in report.inconsistent_clusters:
        dominant = cluster.dominant_family
        if not dominant:
            continue
        for member in cluster.members:
            if member in id_to_idx:
                idx = id_to_idx[member]
                current = df.at[idx, "Protein families"]
                if current != dominant and current:
                    df.at[idx, "Protein families"] = dominant
                    reassigned += 1

    print(f"  Reassigned {reassigned} proteins to dominant cluster labels")

    # Step 2: Merge small families
    counts = df["Protein families"].value_counts()
    small_families = set(counts[counts < min_family_size].index) - {"", "other_toxin"}

    if small_families and embeddings is not None:
        # Try to merge into nearest large family by centroid similarity
        large_families = set(counts[counts >= min_family_size].index) - {"", "other_toxin"}
        large_centroids: dict[str, np.ndarray] = {}
        for fam in large_families:
            fam_ids = df[df["Protein families"] == fam]["identifier"]
            embs = [embeddings[pid] for pid in fam_ids if pid in embeddings]
            if embs:
                centroid = np.mean(embs, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    large_centroids[fam] = centroid / norm

        merged = 0
        for fam in small_families:
            fam_ids = df[df["Protein families"] == fam]["identifier"]
            embs = [embeddings[pid] for pid in fam_ids if pid in embeddings]
            if not embs or not large_centroids:
                # Fall back to other_toxin
                df.loc[df["Protein families"] == fam, "Protein families"] = "other_toxin"
                continue

            small_centroid = np.mean(embs, axis=0)
            norm = np.linalg.norm(small_centroid)
            if norm > 0:
                small_centroid = small_centroid / norm

            best_sim = -1.0
            best_fam = "other_toxin"
            for lfam, lcentroid in large_centroids.items():
                sim = float(np.dot(small_centroid, lcentroid))
                if sim > best_sim:
                    best_sim = sim
                    best_fam = lfam

            if best_sim >= 0.85:
                df.loc[df["Protein families"] == fam, "Protein families"] = best_fam
                merged += 1
            else:
                df.loc[df["Protein families"] == fam, "Protein families"] = "other_toxin"

        print(f"  Merged {merged} small families into nearest large family")
    elif small_families:
        df.loc[df["Protein families"].isin(small_families), "Protein families"] = "other_toxin"
        print(f"  Collapsed {len(small_families)} small families into 'other_toxin'")

    remaining = df["Protein families"].value_counts()
    print(f"  Final: {len(remaining)} unique families, {len(df)} total proteins")

    return df
