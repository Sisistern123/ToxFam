r"""Emit the data-pipeline counts the manuscript cites, from the real artifacts.

Why this exists: these counts used to live in THREE hand-maintained places --
``manuscript/dataset_numbers.tex`` (23 ``\newcommand`` macros), the frozen ``C`` dict in
:mod:`paper.figures.figure_pipeline`, and the prose of Methods/Results/Supplementary,
which mostly retyped the literals instead of using the macros. ``figure_pipeline`` even
carried a "keep the two in sync when the snapshot changes" comment -- the signature of a
missing single source. Nothing checked that the three agreed, and nothing ever would.

Now one module computes them from tracked artifacts and writes both consumers:

* ``paper/figures/output/dataset_numbers.json`` -- read by ``figure_pipeline``
* ``dataset_numbers.tex`` -- written into ``paper/figures/output/`` and straight into the
  manuscript checkout (when present), exactly as ``numbers_manifest`` does for
  ``results_numbers.tex``.

Provenance, per value:

* ``preprocessing_numbers.json`` (tracked; ``make preprocessing-audit``) supplies the
  retrieval/curation funnel and the non-toxin cluster count.
* ``data/splits/split_manifest.json`` -- the pinned split -- supplies the representative
  total and the train/val/test sizes. It is the same manifest every checkpoint is bound
  to, so the manuscript's split numbers cannot drift from the split that was trained on.
* ``data/raw/0800.tsv`` supplies the raw toxin row count and the number of distinct
  UniProt family strings before normalisation.
* The combined run's ``class_indices.json`` supplies the label-space size.
* Everything else is *derived arithmetic* over those (see ``_DERIVED``), so a subtraction
  can never disagree with its operands.

Three values are NOT reproducible from tracked artifacts and are carried explicitly in
``_CARRIED`` with their provenance. They are the signal-peptide counts (which need the
SignalP6 pass) and the normalised-label count before the min-count collapse. They are
still emitted from this one place, so the three-way duplication is gone either way --
but they are frozen, and ``_CARRIED`` says so rather than pretending otherwise.

Run: ``uv run python -m paper.figures.dataset_numbers`` (or ``make dataset-numbers``).
"""

from __future__ import annotations

import json

import pandas as pd
from rich.console import Console

from paper._paths import figures_output_dir, manuscript_tex_target, model_run_dir
from toxfam._paths import get_project_root, raw_dir

console = Console()

# Not reproducible from tracked artifacts. Each needs a pipeline stage whose inputs are
# gitignored, so they are pinned here with the step that produces them. If preprocessing
# is re-run on changed data, refresh these from that run's log.
_CARRIED = {
    # Distinct labels after normalize_protein_families() but BEFORE the min-count
    # collapse that folds families with <10 members into "other". Produced by
    # toxfam.data.preprocessing; a naive normalize(min_count=1) over the raw TSV does
    # NOT reproduce it (that yields 142), so it is not computed here.
    "NormLabels": 45,
    # Sequences whose signal peptide SignalP6 removed, per lane.
    "SpTox": 3785,
    "SpNontox": 17241,
}

# name -> (fn(v) -> int, human-readable identity) for values that are pure arithmetic
# over the sourced ones. Keeping them derived means a subtraction cannot drift from the
# numbers it subtracts.
_DERIVED = {
    "RawTotal": (lambda v: v["RawTox"] + v["RawNontox"], "RawTox + RawNontox"),
    "DropNoFam": (lambda v: v["RawTox"] - v["FamTox"], "RawTox - FamTox"),
    "FamTotal": (lambda v: v["FamTox"] + v["RawNontox"], "FamTox + RawNontox"),
    "LenTotal": (lambda v: v["FamTox"] + v["LenNontox"], "FamTox + LenNontox"),
    "DropCluster": (lambda v: v["LenTotal"] - v["RepTotal"], "LenTotal - RepTotal"),
    # The label space is the toxin families plus the single non-toxin class.
    "RepLabels": (lambda v: v["NumClasses"] - 1, "NumClasses - 1"),
    # Toxin families proper: the label space minus the non-toxin class and minus the
    # catch-all "other". Cited as "36 families" throughout; it had no macro at all.
    "NumFamilies": (lambda v: v["NumClasses"] - 2, "NumClasses - 2"),
    # The paper's motivating statistic, quoted in the Abstract, the Introduction and the
    # Results. Derived, not read from the JSON's own ``nearest_is_nontoxin_frac``, so the
    # percentage and the count it is computed from can never disagree in print.
    "BestHitNonToxPct": (
        lambda v: 100 * v["BestHitNonTox"] / v["RepTox"],
        "100 * BestHitNonTox / RepTox",
    ),
}


def compute() -> dict[str, int | float]:
    """The pipeline counts, sourced and derived. Raises if the identities disagree."""
    pre = json.loads((figures_output_dir() / "preprocessing_numbers.json").read_text())
    funnel, lanes = pre["funnel"], pre["lanes"]

    manifest = json.loads(
        (get_project_root() / "data" / "splits" / "split_manifest.json").read_text()
    )
    counts = manifest["counts"]

    raw_tox = pd.read_csv(raw_dir() / "0800.tsv", sep="\t")
    classes = json.loads((model_run_dir() / "class_indices.json").read_text())

    v: dict[str, int | float] = {
        "RawTox": int(funnel["tox_raw"]),
        "RawNontox": int(funnel["nt_raw"]),
        # Distinct UniProt "Protein families" strings before normalisation -- the
        # numerator of the "collapsed by a factor of four" claim.
        "RawFamStrings": int(raw_tox["Protein families"].dropna().nunique()),
        "FamTox": int(funnel["tox_with_family"]),
        "DropLen": int(funnel["length_n_removed"]),
        "LenThresh": int(funnel["length_cutoff"]),
        "LenNontox": int(funnel["length_n_after"]),
        "RepTox": int(pre["nearest_neighbour"]["n_toxin_reps"]),
        "BestHitNonTox": int(pre["nearest_neighbour"]["nearest_is_nontoxin"]),
        "RepNontox": int(lanes["nontox"]["clusters"]),
        "RepTotal": int(manifest["n_proteins"]),
        "SplitTrain": int(counts["train"]),
        "SplitVal": int(counts["val"]),
        "SplitTest": int(counts["test"]),
        "NumClasses": len(classes),
        **_CARRIED,
    }
    for name, (fn, _) in _DERIVED.items():
        computed = fn(v)
        v[name] = computed if isinstance(computed, float) else int(computed)

    # The two independent routes to the representative total must agree: the split
    # manifest's own row count, and the per-lane cluster counts. They come from
    # different files, so a mismatch means one of them is stale.
    lanes_total = v["RepTox"] + v["RepNontox"]
    if lanes_total != v["RepTotal"]:
        raise SystemExit(
            f"representative total disagrees: split_manifest.json says {v['RepTotal']:,} "
            f"but RepTox + RepNontox = {lanes_total:,}. One of split_manifest.json or "
            "preprocessing_numbers.json is stale -- re-run 'make preprocessing-audit' "
            "after 'toxfam preprocess'."
        )
    if v["SplitTrain"] + v["SplitVal"] + v["SplitTest"] != v["RepTotal"]:
        raise SystemExit("split sizes do not sum to the representative total")
    return v


def _thousands(value: int) -> str:
    r"""Oxford SCIMED thousands separator: a thin space, and only from 10 000 up.

    The checklist reads "Thousand separator is a thin space for 10 000 and above". It
    states what the separator IS; it does not say four-digit numbers go without one, and
    the author prefers the separator throughout for consistency -- 9\,779, not 9779. The
    macros are used inside math mode, where ``\,`` is the thin space.
    """
    return f"{value:,}".replace(",", r"\,") if value >= 1000 else str(value)


def _tex(name: str, value: int | float) -> str:
    r"""One \newcommand, formatted to the journal's number style.

    Counts take the thousands separator; a percentage is a float and takes one decimal
    place instead (the macro is used bare, so the ``%`` sign stays in the prose).
    """
    body = f"{value:.1f}" if isinstance(value, float) else _thousands(value)
    return f"\\newcommand{{\\{name}}}{{{body}}}"


def emit(values: dict[str, int | float], path) -> None:
    """Write the macro file. Ordered to follow the pipeline, not the alphabet."""
    order = [
        ("retrieval", ["RawTox", "RawNontox", "RawTotal", "RawFamStrings"]),
        ("family + length curation", ["DropNoFam", "FamTox", "FamTotal", "NormLabels"]),
        ("length filter", ["DropLen", "LenThresh", "LenNontox", "LenTotal"]),
        ("signal peptides", ["SpTox", "SpNontox"]),
        ("redundancy reduction", ["DropCluster", "RepTox", "RepNontox", "RepTotal"]),
        ("best-hit ceiling", ["BestHitNonTox", "BestHitNonToxPct"]),
        ("label space", ["NumFamilies", "RepLabels", "NumClasses"]),
        ("splits", ["SplitTrain", "SplitVal", "SplitTest"]),
    ]
    lines = [
        "% Auto-generated by paper/figures/dataset_numbers.py -- DO NOT EDIT BY HAND.",
        "% Single source of truth for the data-processing pipeline counts. Computed from",
        "% preprocessing_numbers.json, data/splits/split_manifest.json, data/raw/0800.tsv",
        "% and the combined run's class_indices.json; the subtractions are derived, so they",
        "% cannot drift from their operands. dataset_numbers.py writes this file straight",
        r"% into the manuscript repo too (when checked out), so \input{dataset_numbers}",
        "% never drifts from the pipeline it describes.",
        "",
    ]
    for heading, names in order:
        lines.append(f"% -- {heading} --")
        lines += [_tex(n, values[n]) for n in names]
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    values = compute()

    out_dir = figures_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_numbers.json").write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n"
    )
    emit(values, out_dir / "dataset_numbers.tex")
    console.print(
        f"wrote {out_dir / 'dataset_numbers.json'} and .tex ({len(values)} macros)"
    )

    manuscript = manuscript_tex_target("dataset_numbers.tex")
    if manuscript is not None:
        emit(values, manuscript)
        console.print(f"synced {manuscript}")
    else:
        console.print("[dim]no manuscript checkout found -- skipped the sync[/]")


if __name__ == "__main__":
    main()
