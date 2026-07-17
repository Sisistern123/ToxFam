"""Rebuild data/processed/hbi_train_all.{csv,fasta} from the pinned manifest.

Why this exists: ``hbi_train_all`` is the HBI search reference — all cluster
members of the *training* representatives. It is a split-derived artifact
distributed via release, and a stale copy (built against an earlier split)
contains val/test proteins, so homology self-matches inflate the baseline.

This rebuilds it **from the current manifest and the existing clustering
intermediates**, and NEVER re-runs the stratified splitter — so Ivan's manual
curation, which is keyed to specific identifiers in specific splits, is untouched.

Reconstruction:
  data  = training_data.csv reps (all splits: id, Seq, family, Organism)
        + non-representative cluster members (id from cluster_cluster.tsv,
          Sequence from the post-SP noSP fastas, family inherited from the
          member's representative — clustering is per-family, so a member shares
          its rep's family).
  train_all = build_train_all_members(data, manifest[train])   # train reps only

Then: assert the result shares no identifier with val/test, stamp it with the
manifest hash, and write CSV + FASTA.

Usage:  uv run python scripts/rebuild_hbi_reference.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from toxfam._paths import intermediate_dir, processed_dir  # noqa: E402
from toxfam.data._fasta import read_fasta_as_dict, write_fasta  # noqa: E402
from toxfam.data.preprocessing import build_train_all_members  # noqa: E402
from toxfam.data.split_manifest import (  # noqa: E402
    load_manifest,
    manifest_sha256,
    write_provenance,
)


def reconstruct_data() -> pd.DataFrame:
    """All post-SP proteins (reps + non-rep cluster members) with id/Seq/family."""
    proc = processed_dir()
    interm = intermediate_dir()

    reps = pd.read_csv(proc / "training_data.csv")
    keep = [c for c in ("identifier", "Sequence", "Protein families", "Organism (ID)")
            if c in reps.columns]
    reps = reps[keep].copy()
    rep2fam = dict(zip(reps["identifier"], reps["Protein families"]))

    seqs = {
        **read_fasta_as_dict(interm / "fasta" / "tox_noSP.fasta"),
        **read_fasta_as_dict(interm / "fasta" / "nontox_noSP.fasta"),
    }

    # Non-rep members: parse every family's cluster map, inherit the rep's family.
    mmseqs_dir = interm / "mmseqs"
    rows: list[dict] = []
    seen = set(reps["identifier"])
    for fam_dir in sorted(mmseqs_dir.iterdir()):
        tsv = fam_dir / "cluster_cluster.tsv"
        if not tsv.exists():
            continue
        for line in tsv.read_text().splitlines():
            rep, member = line.split("\t")
            if member in seen or member not in seqs:
                continue
            fam = rep2fam.get(rep)
            if fam is None:  # rep not in the labelled rep set — skip
                continue
            rows.append(
                {"identifier": member, "Sequence": seqs[member], "Protein families": fam}
            )
            seen.add(member)

    data = pd.concat([reps, pd.DataFrame(rows)], ignore_index=True)
    # Fill Sequence for reps from the noSP fastas if the CSV lacked it (defensive).
    if data["Sequence"].isna().any():
        data["Sequence"] = data.apply(
            lambda r: seqs.get(r["identifier"], r["Sequence"]), axis=1
        )
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Report, do not write.")
    args = ap.parse_args()

    proc = processed_dir()
    man = load_manifest()
    train_df = man[man["Split"] == "train"][["identifier"]].copy()
    holdout = set(man[man["Split"].isin(("val", "test"))]["identifier"])

    print(f"manifest sha:      {manifest_sha256()[:12]}")
    print(f"train reps:        {len(train_df):,}")
    print(f"val+test (holdout): {len(holdout):,}")

    data = reconstruct_data()
    print(f"reconstructed data: {len(data):,} proteins "
          f"({data['Protein families'].nunique()} families)")

    train_all = build_train_all_members(data, train_df)
    leaked = set(train_all["identifier"]) & holdout
    missing_reps = set(train_df["identifier"]) - set(train_all["identifier"])

    print(f"\nrebuilt hbi_train_all: {len(train_all):,} members")
    print(f"  val/test leakage:    {len(leaked)}   (must be 0)")
    print(f"  train reps present:  {len(train_df) - len(missing_reps):,}/{len(train_df):,}")

    if leaked:
        print(f"  ✗ ABORT: {len(leaked)} leaked, e.g. {sorted(leaked)[:3]}")
        return 1
    if missing_reps:
        print(f"  ✗ ABORT: {len(missing_reps)} train reps missing from rebuild, "
              f"e.g. {sorted(missing_reps)[:3]}")
        return 1
    print("  ✓ clean: 0 leakage, all train reps covered")

    if args.dry_run:
        print("\n[dry-run] not writing.")
        return 0

    out_csv = proc / "hbi_train_all.csv"
    train_all.to_csv(out_csv, index=False)
    write_fasta(train_all, proc / "hbi_train_all.fasta")
    write_provenance(out_csv)
    print(f"\nwrote {out_csv} + .fasta + provenance stamp ({manifest_sha256()[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
