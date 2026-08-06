#!/usr/bin/env python
"""Assemble a completed curation sheet from a prefilled round + the curator's return.

``build_confident_error_curation.py --prefill-from`` emits two files for a re-curation
round: ``confident_errors_to_curate.tsv``, in which rows whose verdict transferred from
the previous round are already filled and the rest are blank, and
``confident_errors_new_to_curate.tsv``, holding only the blank ones. The curator answers
the second file. This script merges the answers back, producing the round's
``confident_errors_curated.tsv`` -- the single fully-answered sheet that
``paper.stats.load_curated_verdicts`` consumes.

``fp_category`` needs two sources, because the prefilled sheet has no such column: the
returned file carries it for the rows the curator just answered, and ``prior_verdicts.csv``
(written when the previous round was archived) carries it for the transferred ones.

Refuses to write unless every row ends up answered, the returned rows correspond exactly
to the blanks, and the sheet matches its key one-to-one -- the same invariants
``load_curated_verdicts`` enforces at read time, checked here so a bad merge fails at
assembly instead of surfacing as a figure that silently drops rows.

Usage:
    uv run python scripts/assemble_curated_round.py \
        --round-dir paper/data/curation/recuration_2026-07-19_combined_run \
        --prior-verdicts paper/data/curation/archive/2026-07-10_combined_run_e236807/prior_verdicts.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console

console = Console()

# The schema load_curated_verdicts expects, in the order the previous round used.
COLUMNS = [
    "curation_id",
    "identifier",
    "swissprot_side",
    "swissprot_family",
    "model_predicted_family",
    "organism_id",
    "sequence",
    "verdict",
    "assessment",
    "assessment_note",
    "fp_category",
]

# Columns the curator must not have altered: the sheet was cut blind from the checkpoint,
# so a change here means the returned file is not the sheet that was handed out.
BLIND_COLUMNS = [
    "identifier",
    "swissprot_side",
    "swissprot_family",
    "model_predicted_family",
    "organism_id",
    "sequence",
]


def _fail(msg: str) -> None:
    console.print(f"[bold red]refusing to write:[/] {msg}")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round-dir", required=True, type=Path)
    ap.add_argument(
        "--prior-verdicts",
        type=Path,
        default=None,
        help="prior_verdicts.csv of the archived round, source of fp_category for "
        "transferred rows",
    )
    ap.add_argument(
        "--returned",
        default="confident_errors_new_curated.tsv",
        help="the curator's answered file, inside --round-dir",
    )
    ap.add_argument("--dry-run", action="store_true", help="report, do not write")
    args = ap.parse_args()

    rd: Path = args.round_dir
    sheet = pd.read_csv(rd / "confident_errors_to_curate.tsv", sep="\t", dtype=str)
    returned = pd.read_csv(rd / args.returned, sep="\t", dtype=str)
    key = pd.read_csv(rd / "confident_errors_key.tsv", sep="\t", dtype=str)

    blank = sheet["verdict"].isna() | sheet["verdict"].fillna("").str.strip().eq("")
    console.print(
        f"sheet {len(sheet)} rows: {(~blank).sum()} prefilled, {blank.sum()} blank; "
        f"returned file has {len(returned)} rows"
    )

    # The returned rows must be exactly the blanks -- no extras, none missing.
    blank_ids = set(sheet.loc[blank, "curation_id"])
    ret_ids = set(returned["curation_id"])
    if blank_ids != ret_ids:
        _fail(
            f"returned rows do not match the blanks: {len(ret_ids - blank_ids)} extra, "
            f"{len(blank_ids - ret_ids)} missing"
        )

    # The blind columns must be untouched, or the return is not the sheet handed out.
    s_idx = sheet.set_index("curation_id")
    r_idx = returned.set_index("curation_id")
    for col in BLIND_COLUMNS:
        mismatched = (s_idx.loc[sorted(ret_ids), col] != r_idx.loc[sorted(ret_ids), col]).sum()
        if mismatched:
            _fail(f"{mismatched} returned row(s) have an altered blind column {col!r}")
    console.print(f"  blind columns intact across all {len(ret_ids)} returned rows")

    out = sheet.copy()
    if "fp_category" not in out.columns:
        out["fp_category"] = pd.NA

    # Fill the answers the curator just gave.
    for col in ("verdict", "assessment", "assessment_note", "fp_category"):
        if col in returned.columns:
            out[col] = out["curation_id"].map(r_idx[col]).fillna(out[col])

    # fp_category for the transferred rows comes from the archived round.
    if args.prior_verdicts:
        prior = pd.read_csv(args.prior_verdicts, dtype=str).set_index("identifier")
        transferred = out.loc[~out["curation_id"].isin(ret_ids), "identifier"]
        mapped = transferred.map(prior["fp_category"])
        out.loc[mapped.index, "fp_category"] = out.loc[mapped.index, "fp_category"].fillna(mapped)
        missing = set(transferred) - set(prior.index)
        if missing:
            _fail(f"{len(missing)} transferred row(s) absent from prior verdicts, e.g. {sorted(missing)[:5]}")

    # Every row answered, and every false positive categorised.
    unanswered = out["verdict"].fillna("").str.strip().eq("").sum()
    if unanswered:
        _fail(f"{unanswered} row(s) still have no verdict")
    nontox = out["verdict"].str.strip().str.lower().eq("nontox")
    uncategorised = (nontox & out["fp_category"].fillna("").str.strip().eq("")).sum()
    if uncategorised:
        _fail(f"{uncategorised} nontox row(s) have no fp_category")

    # Sheet and key must agree one-to-one, as load_curated_verdicts requires.
    if set(out["identifier"]) != set(key["identifier"]):
        _fail("sheet and key describe different protein sets")
    if out["identifier"].duplicated().any() or key["identifier"].duplicated().any():
        _fail("duplicate identifiers in sheet or key")

    out = out[COLUMNS]
    console.print(
        f"  [green]all {len(out)} rows answered[/]; "
        f"verdict {out['verdict'].str.strip().str.lower().value_counts().to_dict()}; "
        f"assessment {out['assessment'].str.strip().str.lower().value_counts().to_dict()}"
    )

    dest = rd / "confident_errors_curated.tsv"
    if args.dry_run:
        console.print(f"[yellow]--dry-run:[/] would write {dest}")
        return 0
    if dest.exists():
        _fail(f"{dest} already exists; delete it first if you mean to rebuild")
    out.to_csv(dest, sep="\t", index=False)
    console.print(f"wrote [cyan]{dest}[/] ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
