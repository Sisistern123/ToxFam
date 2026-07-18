#!/usr/bin/env python
"""Reproducibly build the confident-error curation sheet for BLIND expert review.

Runs the calibrated combined model on the requested splits, finds predictions that
DISAGREE with the Swiss-Prot family label at calibrated confidence >= threshold, and
writes two files to ``paper/data/curation/``:

  * ``confident_errors_key.tsv``       — INTERNAL. identifier, split, actual_label,
    predicted_label, confidence. Keeps the confidence + split so the curator's verdicts
    can be merged back for the analysis/figure.
  * ``confident_errors_to_curate.tsv`` — BLIND sheet handed to the curator. Contains the
    identifier, the Swiss-Prot family, the model-predicted family, organism and sequence,
    plus empty ``verdict`` / ``assessment`` / ``assessment_note`` columns. It deliberately
    OMITS the calibrated confidence and the split, and the row order is SHUFFLED (fixed
    seed) so that nothing biases the curator toward the model.
  * ``curation_provenance.json``       — the split manifest hash, the checkpoint's stamp,
    and the commit, so a sheet can always answer which split its ``split`` column names.

The ``split`` column is load-bearing: a confident error on a *training* protein is
evidence that Swiss-Prot under-annotates it (the model saw the label, was optimised
toward it, and still refuses it), whereas a confident error on a *held-out* protein has
two explanations — bad label, or failed generalisation. Get the column wrong and the
sheet answers neither question. So the split is read from the git-tracked manifest, and
the run aborts unless the checkpoint was trained against that same manifest.

This supersedes an ad-hoc notebook cell that produced the old test-only
``model_test_wrong_conf.csv`` (which used a fragile ``.iloc[series.index]``
pattern). Run:

    uv run scripts/build_confident_error_curation.py --splits train,val,test --threshold 0.9

Always prints the full count matrix (splits x {0.8, 0.9}) before writing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.console import Console

from toxfam._git import git_commit_short
from toxfam._paths import get_project_root, processed_dir
from toxfam.data.split_manifest import (
    apply_manifest,
    manifest_sha256,
    provenance_path,
    verify_split_provenance,
)
from toxfam.model.inference import run_inference

console = Console()


def _write_provenance(
    out_dir: Path,
    model_dir: Path,
    splits: list[str],
    threshold: float,
    prefill_from: str | None = None,
) -> None:
    """Record which split and which checkpoint produced this sheet.

    The previous sheet could not answer "which split is this row's `split` column from?".
    It had been labelled from a training_data.csv that a later `download-data --force`
    had replaced, so 81 rows read "test" when 18 of them were.
    """
    (out_dir / "curation_provenance.json").write_text(
        json.dumps(
            {
                "split_manifest_sha256": manifest_sha256(),
                "checkpoint_stamp": json.loads(provenance_path(model_dir).read_text()),
                "model_dir": str(model_dir),
                "splits": splits,
                "confidence_threshold": threshold,
                "prefilled_from": prefill_from,
                "git_commit": git_commit_short(),
            },
            indent=2,
        )
        + "\n"
    )


def predictions_for_split(td: pd.DataFrame, split: str, model_dir: Path, h5: Path) -> pd.DataFrame:
    """Calibrated combined-model predictions for one split, with ground truth + metadata."""
    # training_data.csv is the *post-preprocessing* split CSV: its "Protein families"
    # column is already the normalized 38-class label space (incl. "nontox"), identical
    # to the benchmark actual_label. Do NOT re-normalize — normalize_protein_families
    # expects raw UniProt strings and would remap "nontox" away, mangling ground truth.
    df = td[td["Split"] == split].dropna(subset=["Protein families"]).copy()
    inf = run_inference(df, h5, model_dir)
    out = (
        df[["identifier", "Protein families", "Organism (ID)", "Sequence"]]
        .reset_index(drop=True)
        .rename(columns={"Protein families": "actual_label",
                         "Organism (ID)": "organism_id", "Sequence": "sequence"})
    )
    out["predicted_label"] = inf["predicted_label"].values
    out["confidence"] = inf["confidence"].values
    out["split"] = split
    return out


def _prefill_verdicts(blind: pd.DataFrame, ce: pd.DataFrame, prior_csv: Path) -> pd.DataFrame:
    """Carry verdicts over from an earlier curation round, by identifier.

    A verdict answers one specific question — "Swiss-Prot says X, the model says Y; who
    is right?" — so it only transfers while *both* labels are unchanged. A different
    checkpoint can predict a different family for the same protein, and the old answer
    would then be attached to a question nobody asked. Such rows are left blank for the
    curator rather than carried over, and the count is printed.
    """
    prior = pd.read_csv(prior_csv)
    ask = ce.set_index("identifier")[["actual_label", "predicted_label"]]

    same = prior.join(ask, on="identifier", how="inner", rsuffix="_now")
    unchanged = (
        same["actual_label"].eq(same["actual_label_now"])
        & same["predicted_label"].eq(same["predicted_label_now"])
    )
    carried = same[unchanged].set_index("identifier")
    stale = int((~unchanged).sum())

    for col in ("verdict", "assessment", "assessment_note"):
        filled = blind["identifier"].map(carried[col])
        blind[col] = filled.fillna("")

    console.print(
        f"[yellow]pre-filled[/] {len(carried)} verdict(s) from {prior_csv.name} "
        f"({len(blind) - len(carried)} left for the curator)"
        + (f"; [red]{stale} refused[/] — the label or the prediction changed since"
           if stale else "")
    )
    return blind


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="train,val,test",
                    help="comma-separated subset of train,val,test (default: all)")
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="calibrated-confidence threshold for 'confident' (default 0.9)")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed for the blind sheet")
    ap.add_argument("--train-subsample-frac", type=float, default=None,
                    help="if set, keep only this fraction of the TRAIN confident errors, "
                         "sampled equally across confidence bins in [threshold, 1.0] "
                         "(test+val kept in full); e.g. 0.3333 for one third")
    ap.add_argument("--train-subsample-bins", type=int, default=10,
                    help="number of equal-width confidence bins for stratified train subsampling")
    ap.add_argument("--prefill-from", default=None,
                    help="CSV of verdicts from an earlier curation round; rows whose "
                         "identifier reappears here are pre-filled so the curator does "
                         "not re-answer them (see _prefill_verdicts for when a verdict "
                         "is refused)")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root = get_project_root()
    model_dir = Path(args.model_dir) if args.model_dir else root / "model" / "model_output" / "combined_run"
    h5 = processed_dir() / "embeddings.h5"

    # The sheet's `split` column is its whole point: it separates "the model was trained
    # on this protein" (evidence of under-annotation) from "the model never saw it"
    # (evidence of generalisation). Two things have to hold for that column to mean
    # anything, and neither used to:
    #   1. the split must come from the manifest, not from training_data.csv's own Split
    #      column, which `download-data --force` can replace;
    #   2. the checkpoint must have been trained against that same manifest.
    # Check before creating anything, so a refused run leaves no directory behind.
    verify_split_provenance(model_dir)

    out_dir = Path(args.out_dir) if args.out_dir else root / "paper" / "data" / "curation"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    td = apply_manifest(pd.read_csv(processed_dir() / "training_data.csv"))

    frames = []
    for s in splits:
        console.print(f"[bold]running inference on split '{s}'[/] ...")
        frames.append(predictions_for_split(td, s, model_dir, h5))
    allp = pd.concat(frames, ignore_index=True)
    allp["wrong"] = allp["actual_label"] != allp["predicted_label"]

    console.print("\n[bold]Confident-error counts (wrong predictions):[/]")
    for thr in (0.8, 0.9):
        row = {s: int(((allp["split"] == s) & allp["wrong"] & (allp["confidence"] >= thr)).sum())
               for s in splits}
        row["TOTAL"] = sum(row.values())
        console.print(f"  conf >= {thr}: " + "   ".join(f"{k}={v}" for k, v in row.items()))

    ce = allp[allp["wrong"] & (allp["confidence"] >= args.threshold)].copy()

    if args.train_subsample_frac is not None and "train" in splits:
        import numpy as np
        tr = ce[ce["split"] == "train"]
        rest = ce[ce["split"] != "train"]
        n_target = round(len(tr) * args.train_subsample_frac)

        # Retain train confident errors that already carry a transferable verdict (same
        # identifier + Swiss-Prot label + prediction as the prefill round) and subsample
        # only the NEW ones to top the sheet up to n_target. Subsampling blind to curation
        # status would resample the retained set on every re-run and discard completed
        # verdicts; here a re-run keeps all prior curation and asks only for the delta.
        retained = tr.iloc[:0]
        if args.prefill_from:
            prior = pd.read_csv(args.prefill_from)
            ask = tr.set_index("identifier")[["actual_label", "predicted_label"]]
            j = prior.join(ask, on="identifier", how="inner", rsuffix="_now")
            keep_ids = set(j.loc[
                j["actual_label"].eq(j["actual_label_now"])
                & j["predicted_label"].eq(j["predicted_label_now"]),
                "identifier"])
            retained = tr[tr["identifier"].isin(keep_ids)]
            tr = tr[~tr["identifier"].isin(keep_ids)]

        n_sample = max(0, n_target - len(retained))
        if n_sample and len(tr):
            edges = np.linspace(args.threshold, 1.0, args.train_subsample_bins + 1)
            bin_idx = np.clip(np.digitize(tr["confidence"], edges) - 1, 0, args.train_subsample_bins - 1)
            per_bin = max(1, round(n_sample / args.train_subsample_bins))
            picked = [g.sample(min(len(g), per_bin), random_state=args.seed + int(b))
                      for b, g in tr.groupby(bin_idx)]
            sampled = pd.concat(picked)
        else:
            sampled = tr.iloc[:0]
        tr = pd.concat([retained, sampled])
        ce = pd.concat([rest, tr], ignore_index=True)
        console.print(f"[yellow]train subsampled[/] to {len(tr)} "
                      f"({len(retained)} retained already-curated + {len(sampled)} new; "
                      f"target ~{n_target}, {args.train_subsample_bins} confidence bins)")

    ce["swissprot_side"] = ce["actual_label"].eq("nontox").map(
        {True: "nontoxin", False: "toxin(KW-0800)"})

    key = ce[["identifier", "split", "actual_label", "predicted_label", "confidence"]]
    key.to_csv(out_dir / "confident_errors_key.tsv", sep="\t", index=False)

    blind = (ce[["identifier", "swissprot_side", "actual_label", "predicted_label",
                 "organism_id", "sequence"]]
             .rename(columns={"actual_label": "swissprot_family",
                              "predicted_label": "model_predicted_family"}))
    for c in ("verdict", "assessment", "assessment_note"):
        blind[c] = ""
    if args.prefill_from:
        blind = _prefill_verdicts(blind, ce, Path(args.prefill_from))
    blind = blind.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    blind.insert(0, "curation_id", [f"CE{i + 1:03d}" for i in range(len(blind))])
    blind.to_csv(out_dir / "confident_errors_to_curate.tsv", sep="\t", index=False)

    # When re-curating against a new checkpoint, hand the curator ONLY the rows that
    # still need a verdict (the prefill filled the rest). Same blind format and shuffle.
    if args.prefill_from:
        new_only = blind[blind["verdict"].astype(str).str.strip() == ""]
        new_only.to_csv(out_dir / "confident_errors_new_to_curate.tsv", sep="\t", index=False)
        console.print(f"[green]wrote[/] {len(new_only)} NEW (uncurated) rows -> "
                      f"{out_dir}/confident_errors_new_to_curate.tsv")

    _write_provenance(out_dir, model_dir, splits, args.threshold, args.prefill_from)

    console.print(
        f"\n[green]wrote[/] {len(ce)} confident errors (conf >= {args.threshold}, "
        f"splits={'+'.join(splits)}) -> {out_dir}/confident_errors_to_curate.tsv (blind) "
        f"+ confident_errors_key.tsv (internal) "
        f"+ curation_provenance.json (split manifest {manifest_sha256()[:12]})")


if __name__ == "__main__":
    main()
