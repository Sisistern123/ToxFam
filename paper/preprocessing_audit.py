"""Empirical audit of the ToxFam preprocessing decisions (reusable, unit-tested).

Companion to :mod:`paper.stats`: that module measures the *model*, this one measures
the *data pipeline* that feeds it. Both live in the repo-only ``paper`` tree and are
imported by manuscript-facing drivers, never by ``src/toxfam``.

The Methods section states six preprocessing choices and quantifies almost none of
them. Each function here re-derives one of those claims from the frozen raw TSVs, the
git-tracked split manifest, and the pipeline's own MMseqs2 / SignalP6 caches.

Layout mirrors :mod:`paper.stats`: **pure transforms first** (they take DataFrames and
return DataFrames/dicts, and are what ``paper/tests/test_preprocessing_audit.py``
covers), then the **I/O + MMseqs drivers**, which need the gitignored pipeline caches
and are therefore exercised by ``make preprocessing-audit`` rather than by CI.

Three measurement subtleties are load-bearing and are enforced here rather than left
to call sites:

* ``other`` is **not** a protein family. ``normalize_protein_families`` collapses
  sub-10-member families into it *before* clustering, so ``data/intermediate/mmseqs/``
  contains a literal ``other/`` directory. Counting it as a family inflates every
  "N toxin families" statistic and puts a pseudo-class inside the five largest.
  :func:`is_real_family` is the single gate.
* A **nearest neighbour must be ranked the way the baseline being bounded ranks it**.
  :func:`nearest_neighbour_labels` defaults to ``toxfam.evaluation.hbi``'s own
  parameters (min E-value, no coverage floor), because a coverage-filtered neighbour
  is a different protein and understates the transfer-error floor by ~1.7x.
* Comparing alignments before/after signal-peptide trimming must be **paired per
  query**. Medians-of-per-family-medians are stable while ~30% of individual
  alignments move; see :func:`paired_alignment_shift`.
"""

from __future__ import annotations

import contextlib
import io
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from toxfam._paths import get_project_root, intermediate_dir, raw_dir
from toxfam.data._fasta import parse_fasta, write_fasta
from toxfam.data.normalization import (
    ensure_identifier_column,
    normalize_protein_families,
)
from toxfam.data.split_manifest import load_manifest

# --- Labels that are not protein families -----------------------------------
# "nontox" is the single bucket every non-toxin is relabelled into
# (preprocessing.load_and_prepare_raw); "other" is the sub-min_count collapse target
# (normalization.normalize_protein_families). Neither is a family, and both appear as
# directories under data/intermediate/mmseqs/.
NONTOX_LABEL = "nontox"
OTHER_LABEL = "other"
NON_FAMILY_LABELS = frozenset({NONTOX_LABEL, OTHER_LABEL})

# --- Pipeline parameters this audit is auditing ------------------------------
# Kept as named constants so a claim about "the 10-member floor" or "the 90% cutoff"
# cannot silently diverge from the number the pipeline actually used.
PIPELINE_MIN_SEQ_ID = 0.9
MIN_MEMBERS = 10  # normalize_protein_families(min_count=...) default

# A pair counts as a near-duplicate only if it is BOTH highly identical and mutually
# near-full-length: fident alone is over the aligned region, so 99% identity across 20
# of 300 residues is not a duplicate.
DUP_IDENT = 0.90
DUP_COV = 0.80

# MMseqs2 output columns requested from every search in this module.
SEARCH_COLS = "query,target,fident,alnlen,qlen,tlen,evalue,bits"


# ---------------------------------------------------------------------------
# Pure transforms (unit-tested; no I/O)
# ---------------------------------------------------------------------------


def is_real_family(label: object) -> bool:
    """True if ``label`` names an actual protein family.

    Excludes both pipeline pseudo-classes. See the module docstring: counting
    ``other`` as a family is the single easiest way to get every family-count claim
    in this audit wrong, because it is the 4th-largest bucket by membership and so
    lands inside any "five largest families" selection.
    """
    return str(label) not in NON_FAMILY_LABELS


def real_family_mask(labels: Iterable[object]) -> np.ndarray:
    """Boolean mask selecting entries whose label is a real family."""
    return np.array([is_real_family(x) for x in labels], dtype=bool)


def first_family(s: pd.Series) -> pd.Series:
    """The pipeline's own tokenization: first listed family, before consolidation.

    Mirrors the leading statements of ``normalize_protein_families`` (split on ``;``
    then ``,``) but deliberately stops short of the regex consolidation, because the
    shared-vocabulary analysis needs the raw UniProt strings that consolidation
    destroys. The capitalization step is also omitted; it is a no-op on the current
    raw TSVs (no family string begins lowercase).
    """
    return s.str.strip().str.split(";").str[0].str.split(",").str[0].str.strip()


def add_alignment_coverage(hits: pd.DataFrame) -> pd.DataFrame:
    """Add ``alncov`` = alignment length over the *longer* of the two sequences.

    Deliberately stricter than MMseqs' own ``--cov-mode 0``: guards against a short
    local match being read as whole-protein similarity.
    """
    out = hits.copy()
    out["alncov"] = out["alnlen"] / out[["qlen", "tlen"]].max(axis=1)
    return out


def cluster_reduction_stats(records: Sequence[dict]) -> pd.DataFrame:
    """Per-family clustering outcome, with the pseudo-classes flagged, not dropped.

    ``records`` carry ``family``, ``members`` and ``clusters``. The returned frame adds
    ``removed``, ``reduction`` and ``is_family`` so callers can select real families
    explicitly rather than by an ad-hoc ``!= "nontox"`` test that silently keeps
    ``other``.
    """
    df = pd.DataFrame(list(records))
    if df.empty:
        return df.assign(removed=[], reduction=[], is_family=[])
    df["removed"] = df["members"] - df["clusters"]
    df["reduction"] = 1 - df["clusters"] / df["members"]
    df["is_family"] = df["family"].map(is_real_family)
    return df


def redundancy_concentration(stats: pd.DataFrame, *, top_n: int = 5) -> dict:
    """How concentrated redundancy removal is across the real toxin families.

    Answers F1: is redundancy a property of the toxin set, or of a handful of
    over-sequenced families? ``other`` and ``nontox`` are excluded, so ``top_n``
    selects genuine families.
    """
    fam = stats[stats["is_family"] & (stats["family"] != NONTOX_LABEL)]
    if fam.empty or fam["removed"].sum() == 0:
        return {
            "n_families": int(len(fam)),
            "top_n": top_n,
            "top_n_share": float("nan"),
            "total_removed": 0,
            "zero_redundancy": int((fam["removed"] == 0).sum()) if len(fam) else 0,
            "median_reduction": float("nan"),
        }
    top = fam.nlargest(top_n, "members")
    return {
        "n_families": int(len(fam)),
        "top_n": top_n,
        "top_n_share": float(top["removed"].sum() / fam["removed"].sum()),
        "total_removed": int(fam["removed"].sum()),
        "zero_redundancy": int((fam["removed"] == 0).sum()),
        "median_reduction": float(fam["reduction"].median()),
        "top_families": list(top["family"]),
    }


def lane_reduction(stats: pd.DataFrame) -> dict:
    """Sequences-to-representatives reduction, split by lane.

    The toxin lane here is *real families only*; ``other`` is reported separately so
    it is visible rather than silently folded into either lane.
    """

    def _agg(sub: pd.DataFrame) -> dict:
        members, clusters = int(sub["members"].sum()), int(sub["clusters"].sum())
        return {
            "families": int(len(sub)),
            "members": members,
            "clusters": clusters,
            "removed_frac": float(1 - clusters / members) if members else float("nan"),
        }

    is_nontox = stats["family"] == NONTOX_LABEL
    is_other = stats["family"] == OTHER_LABEL
    return {
        "toxin": _agg(stats[~is_nontox & ~is_other]),
        "other": _agg(stats[is_other]),
        "nontox": _agg(stats[is_nontox]),
    }


def ladder_summary(
    ladder: pd.DataFrame, *, min_members: int = MIN_MEMBERS
) -> pd.DataFrame:
    """Collapse a per-family threshold ladder into per-threshold label-space survival.

    This is F2's measurement: the quantity that matters is not how many
    representatives survive a stricter cutoff, but how many families stay above the
    ``min_members`` floor -- families below it are dissolved into ``other`` and stop
    being prediction targets. ``ladder`` must already exclude the pseudo-classes.
    """
    lad = ladder.assign(kept=ladder["n_rep"] >= min_members)
    lad["reps_below_floor"] = lad["n_rep"].where(~lad["kept"], 0)
    summary = (
        lad.groupby("threshold")
        .agg(
            toxin_reps=("n_rep", "sum"),
            families_kept=("kept", "sum"),
            seqs_into_other=("reps_below_floor", "sum"),
        )
        .reset_index()
    )
    n_families = int(ladder["family"].nunique())
    summary["families_dissolved"] = n_families - summary["families_kept"]
    return summary


def shared_family_table(tox_fam: pd.Series, nt_fam: pd.Series) -> pd.DataFrame:
    """Family strings occurring on *both* sides of the KW-0800 keyword.

    F3's scale measurement: a family here is a scaffold recruited into venom that
    still has non-venom members, so per-family clustering can never deduplicate it
    across the toxic/non-toxic boundary.
    """
    tox_counts, nt_counts = tox_fam.value_counts(), nt_fam.value_counts()
    shared = sorted(set(tox_fam.dropna()) & set(nt_fam.dropna()))
    return (
        pd.DataFrame({"family": shared})
        .assign(
            toxins=lambda d: d["family"].map(tox_counts).astype(int),
            non_toxins=lambda d: d["family"].map(nt_counts).astype(int),
            smaller_side=lambda d: d[["toxins", "non_toxins"]].min(axis=1),
        )
        .sort_values("toxins", ascending=False)
        .reset_index(drop=True)
    )


def near_duplicate_pairs(
    hits: pd.DataFrame,
    *,
    key: str = "target",
    min_ident: float = DUP_IDENT,
    min_cov: float = DUP_COV,
) -> pd.DataFrame:
    """Highest-identity near-duplicate hit per ``key``, above identity+coverage floors."""
    dup = hits[(hits["fident"] >= min_ident) & (hits["alncov"] >= min_cov)]
    return dup.sort_values("fident", ascending=False).drop_duplicates(key)


def best_hits(
    hits: pd.DataFrame, key: str, *, min_cov: float = 0.0, rank: str = "fident"
) -> pd.DataFrame:
    """Best hit per ``key``.

    ``rank`` selects the ordering: ``"fident"`` (most identical -- the right choice
    when the question is "how similar is the closest thing?"), ``"bits"``, or
    ``"evalue"`` (minimum -- what ``toxfam.evaluation.hbi`` transfers on).
    """
    d = hits[hits["alncov"] >= min_cov] if min_cov else hits
    if d.empty:
        return d
    if rank == "evalue":
        return d.loc[d.groupby(key)["evalue"].idxmin()]
    ascending = False  # fident and bits: higher is better
    return d.sort_values(rank, ascending=ascending).drop_duplicates(key)


def nearest_neighbour_labels(
    hits: pd.DataFrame,
    family_of: pd.Series | dict,
    *,
    rank: str = "evalue",
    min_cov: float = 0.0,
    drop_self: bool = True,
) -> pd.DataFrame:
    """Each query's nearest neighbour and whether that neighbour is a non-toxin.

    This bounds best-hit annotation transfer, so the defaults deliberately match
    ``toxfam.evaluation.hbi``: rank by **minimum E-value** and apply **no coverage
    floor**. Imposing this module's ``DUP_COV`` here would measure a different
    protein -- verified on the released representatives, a 0.80 coverage filter
    changes the answer for 145 of 3,416 toxins and understates the non-toxin-neighbour
    rate from 8.8% to 5.1%.
    """
    d = hits[hits["query"] != hits["target"]] if drop_self else hits
    best = best_hits(d, "query", min_cov=min_cov, rank=rank)
    if best.empty:
        return best.assign(nn_family=[], nn_is_nontox=[])
    fam = pd.Series(family_of) if not isinstance(family_of, pd.Series) else family_of
    nn_family = best["target"].map(fam)
    return best.assign(nn_family=nn_family, nn_is_nontox=nn_family == NONTOX_LABEL)


def identity_ladder(
    fident: pd.Series,
    *,
    n_total: int,
    label: str,
    thresholds: Sequence[float] = (0.9, 0.8, 0.7, 0.5, 0.3),
) -> pd.DataFrame:
    """Count entries at or above each identity threshold, as a share of ``n_total``.

    One helper for what the notebook otherwise hand-built per section with a different
    threshold tuple each time, which made the tables incomparable.
    """
    counts = [int((fident >= t).sum()) for t in thresholds]
    return pd.DataFrame(
        {
            "identity >=": [f"{t:.0%}" for t in thresholds],
            label: counts,
            f"% of {label}": [
                f"{c / n_total:.1%}" if n_total else "n/a" for c in counts
            ],
        }
    )


def homology_band_table(
    best: pd.DataFrame,
    *,
    subset_label: str,
    n_total: int,
    bands: Sequence[tuple[float, float]] = (
        (0.9, 1.01),
        (0.7, 0.9),
        (0.5, 0.7),
        (0.0, 0.5),
    ),
) -> dict:
    """Distribution of best-hit identity into the training set, in bands."""
    row = {"subset": subset_label, "n": n_total, "no hit": n_total - len(best)}
    for lo, hi in bands:
        label = f"{lo:.0%}-{min(hi, 1.0):.0%}"
        row[label] = int(((best["fident"] >= lo) & (best["fident"] < hi)).sum())
    return row


def paired_alignment_shift(
    untrimmed: pd.DataFrame, trimmed: pd.DataFrame, *, key: str = "query"
) -> dict:
    """Per-query paired comparison of alignments before vs after SP trimming.

    F9 hinges on whether the signal peptide changes the alignment. Comparing
    per-family *medians* answers a different and much weaker question: medians are
    stable while individual alignments move. This joins on ``key`` so each query is
    compared against itself, and reports the fraction that actually changed.
    """
    cols = ["alnlen", "bits", "fident", "alncov"]
    a = untrimmed.drop_duplicates(key).set_index(key)[cols]
    b = trimmed.drop_duplicates(key).set_index(key)[cols]
    joined = a.join(b, how="inner", lsuffix="_untrimmed", rsuffix="_trimmed")
    n = len(joined)
    if n == 0:
        return {"n_paired": 0}
    out = {"n_paired": int(n)}
    for c in cols:
        delta = joined[f"{c}_trimmed"] - joined[f"{c}_untrimmed"]
        out[f"{c}_changed"] = int((delta != 0).sum())
        out[f"{c}_changed_frac"] = float((delta != 0).mean())
        out[f"{c}_median_delta"] = float(delta.median())
    return out


def signal_peptide_shortcut(tox_trimmed: pd.Series, nt_trimmed: pd.Series) -> dict:
    """Score "has a signal peptide" alone as a binary toxic/non-toxic classifier.

    F9's second, independent argument: the tag is genuinely predictive, and genuinely
    misleading, because it tracks secretion rather than toxicity.
    """
    tox_yes, tox_no = int(tox_trimmed.sum()), int((~tox_trimmed).sum())
    nt_yes, nt_no = int(nt_trimmed.sum()), int((~nt_trimmed).sum())
    n_tox, n_nt = len(tox_trimmed), len(nt_trimmed)
    base = n_tox / (n_tox + n_nt)
    lift = tox_yes / (tox_yes + nt_yes) if (tox_yes + nt_yes) else float("nan")
    odds = (
        (tox_yes / tox_no) / (nt_yes / nt_no)
        if tox_no and nt_yes and nt_no
        else float("nan")
    )
    tp, fp, fn = tox_yes, nt_yes, tox_no
    return {
        "tox_rate": tox_yes / n_tox if n_tox else float("nan"),
        "nt_rate": nt_yes / n_nt if n_nt else float("nan"),
        "odds_ratio": float(odds),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "base_rate": float(base),
        "lift_rate": float(lift),
        "lift_factor": float(lift / base) if base else float("nan"),
    }


def mixed_frame_families(
    families: pd.Series,
    trimmed: pd.Series,
    *,
    min_members: int = MIN_MEMBERS,
    lo: float = 0.10,
    hi: float = 0.90,
) -> pd.DataFrame:
    """Per-family signal-peptide trim rate, flagging families carrying both frames.

    A family between ``lo`` and ``hi`` contains both precursor and mature sequences --
    the within-class heterogeneity that motivates trimming for an embedding model.
    """
    df = pd.DataFrame({"family": list(families), "trimmed": list(trimmed)})
    df = df[df["family"].map(is_real_family)]
    agg = (
        df.groupby("family")["trimmed"]
        .agg(n="size", trim_rate="mean")
        .query("n >= @min_members")
        .sort_values("trim_rate")
    )
    agg["mixed"] = agg["trim_rate"].between(lo, hi)
    return agg


def split_straddling_pairs(
    pairs: pd.DataFrame, *, left_split: str = "tox_split", right_split: str = "nt_split"
) -> dict:
    """Count near-duplicate pairs whose two members land in different splits.

    Direction-symmetric on purpose: a *test* toxin with a *training* non-toxin twin
    leaks exactly as much as the reverse, and reporting only one direction understates
    the count.
    """
    both = pairs.dropna(subset=[left_split, right_split])
    differ = both[left_split] != both[right_split]
    train_to_eval = (both[left_split] == "train") & (
        both[right_split].isin(["val", "test"])
    )
    eval_to_train = (both[right_split] == "train") & (
        both[left_split].isin(["val", "test"])
    )
    return {
        "n_pairs": int(len(both)),
        "different_split": int(differ.sum()),
        "same_split": int((~differ).sum()),
        "toxin_train_nontoxin_eval": int(train_to_eval.sum()),
        "toxin_eval_nontoxin_train": int(eval_to_train.sum()),
        "cross_train_eval": int((train_to_eval | eval_to_train).sum()),
    }


def clustering_delta_spread(clust: pd.DataFrame) -> dict:
    """Net *and* per-family spread of the trimmed-vs-untrimmed representative counts.

    The net is near zero, but it is a sum of sizeable cancelling per-family shifts;
    reporting only the net supports a stronger claim than the data does.
    """
    delta = clust["trimmed"] - clust["untrimmed"]
    rel = delta / clust["untrimmed"]
    return {
        "n_families": int(len(clust)),
        "net": int(delta.sum()),
        "net_frac": float(delta.sum() / clust["untrimmed"].sum()),
        "n_up": int((delta > 0).sum()),
        "n_down": int((delta < 0).sum()),
        "n_unchanged": int((delta == 0).sum()),
        "max_rel_increase": float(rel.max()),
        "max_rel_decrease": float(rel.min()),
        "mean_abs_rel": float(rel.abs().mean()),
    }


def length_cut_effect(lengths: pd.Series, *, pct: float = 0.01) -> dict:
    """The longest-``pct`` cut: its ceiling, and what share of residue mass it removes.

    Reproduces ``preprocessing.load_and_prepare_raw``'s rule exactly (drop the longest
    ``pct``, keeping ties at the cutoff) so the audit and the pipeline cannot diverge.
    """
    cutoff = int(lengths.nlargest(int(np.ceil(len(lengths) * pct))).min())
    kept = lengths[lengths <= cutoff]
    return {
        "cutoff": cutoff,
        "n_before": int(len(lengths)),
        "n_after": int(len(kept)),
        "n_removed": int(len(lengths) - len(kept)),
        "median_before": float(lengths.median()),
        "median_after": float(kept.median()),
        "residue_mass_removed": float(1 - kept.sum() / lengths.sum()),
    }


def to_jsonable(obj):
    """Recursively coerce numpy scalars so a numbers dict is ``json.dumps``-able."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# I/O + MMseqs drivers (need the gitignored pipeline caches)
# ---------------------------------------------------------------------------


def mmseqs_cache_dir() -> Path:
    """The pipeline's per-family clustering cache -- read-only for this audit."""
    return intermediate_dir() / "mmseqs"


def work_dir() -> Path:
    """Scratch + cache directory for everything this audit writes (gitignored)."""
    return intermediate_dir() / "preprocessing_rationale"


def numbers_json_path() -> Path:
    """Tracked machine-readable output, beside the figure tree's results_numbers.json.

    Deliberately *not* under ``data/intermediate/`` (gitignored): these numbers are
    quoted in the manuscript, so they must survive a clean clone and show up in diffs.
    """
    from paper._paths import figures_output_dir

    return figures_output_dir() / "preprocessing_numbers.json"


def require_pipeline_caches() -> Path:
    """Fail fast, with the fix, when the pipeline caches this audit reads are absent.

    ``data/intermediate/mmseqs/`` is **not** distributed by ``toxfam download-data``
    (``DATA_ASSETS`` ships only ``sp6_cache.zip`` from that tree) and is gitignored, so
    a clean clone cannot run this audit without a full preprocess run. Half the audit
    reads it; failing here beats failing five sections in with a bare FileNotFoundError.
    """
    mm = mmseqs_cache_dir()
    if not (mm / "representatives" / "all.csv").exists():
        raise FileNotFoundError(
            f"Per-family clustering cache not found at {mm}.\n"
            "It is gitignored and is NOT part of `toxfam download-data` (that ships "
            "only the SignalP6 cache from data/intermediate/).\n"
            "Regenerate it with:  uv run toxfam preprocess"
        )
    return mm


def load_raw_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Raw toxin/non-toxin TSVs plus the length-cut summary, as the pipeline stages them.

    Returns ``(toxins_with_family, non_toxins_after_length_cut, funnel)``. The toxin
    frame keeps its raw ``Protein families`` strings -- consolidation would destroy the
    shared-vocabulary signal F3 measures.
    """
    tox_raw = pd.read_csv(raw_dir() / "0800.tsv", sep="\t")
    nt_raw = pd.read_csv(raw_dir() / "nontox.tsv", sep="\t")
    tox = tox_raw.dropna(subset=["Protein families"]).copy()
    cut = length_cut_effect(nt_raw["Sequence"].str.len())
    nt = nt_raw[nt_raw["Sequence"].str.len() <= cut["cutoff"]].reset_index(drop=True)
    funnel = {
        "tox_raw": int(len(tox_raw)),
        "nt_raw": int(len(nt_raw)),
        "tox_with_family": int(len(tox)),
        "nt_after_length_cut": int(len(nt)),
        **{f"length_{k}": v for k, v in cut.items()},
    }
    return tox, nt, funnel


def read_cluster_cache(mm: Path | None = None) -> pd.DataFrame:
    """Per-family clustering outcome, read from the pipeline's own cluster assignments.

    These are the exact clusters the released split was built from, so this is a read
    of history rather than a recomputation.
    """
    mm = mm or require_pipeline_caches()
    records = []
    for d in sorted(mm.iterdir()):
        tsv = d / "cluster_cluster.tsv"
        if not d.is_dir() or d.name == "representatives" or not tsv.exists():
            continue
        cl = pd.read_csv(tsv, sep="\t", names=["rep", "member"])
        records.append(
            {
                "family": d.name,
                "members": int(len(cl)),
                "clusters": int(cl["rep"].nunique()),
            }
        )
    return cluster_reduction_stats(records)


def load_representatives(mm: Path | None = None) -> pd.DataFrame:
    """Representatives joined to the git-tracked split manifest.

    Uses :func:`toxfam.data.split_manifest.load_manifest`, which validates columns,
    split values and identifier uniqueness -- a raw ``read_csv`` + ``dict(zip(...))``
    would map a duplicated or missing identifier to a silent ``NaN`` and shrink every
    downstream denominator without complaint.
    """
    mm = mm or require_pipeline_caches()
    reps = pd.read_csv(mm / "representatives" / "all.csv")
    manifest = load_manifest()
    merged = reps.merge(manifest, on="identifier", how="left", validate="one_to_one")
    missing = int(merged["Split"].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} representative(s) are absent from the split manifest; "
            "the manifest and the clustering cache disagree. Re-run `toxfam preprocess`."
        )
    merged["is_toxin_lane"] = merged["Protein families"] != NONTOX_LABEL
    merged["is_real_family"] = merged["Protein families"].map(is_real_family)
    return merged


def write_fasta_subset(
    df: pd.DataFrame, path: Path, *, seq_col: str = "Sequence"
) -> Path:
    """Write a UniProt-style frame to FASTA with the pipeline's own writer."""
    d = ensure_identifier_column(df)
    write_fasta(d, path, seq_col=seq_col)
    return path


def split_fasta_by(src: Path, route, outs: dict[str, Path]) -> None:
    """Partition a FASTA into named output files by ``route(record_id) -> key or None``.

    One implementation for what was two near-identical hand-rolled streaming loops
    that differed subtly in their None-handling. Records routed to an unknown key are
    dropped. No-ops when every output already exists.
    """
    if all(p.exists() for p in outs.values()):
        return
    handles = {k: open(p, "w") for k, p in outs.items()}
    try:
        for rec in parse_fasta(src):
            f = handles.get(route(rec.id))
            if f is not None:
                f.write(f">{rec.id}\n{rec.seq}\n")
    finally:
        for f in handles.values():
            f.close()


def count_fasta_records(path: Path) -> int:
    """Number of records in a FASTA file."""
    return sum(1 for _ in parse_fasta(path))


def _cache_stem(name: str, params: dict) -> str:
    """Fold search parameters into the cache filename.

    ``preprocessing._cluster_cache_key`` learned this the hard way: keyed on the name
    alone, editing ``s``/``e``/``max_seqs`` and re-running silently reuses the old TSV
    while printing the new parameters.
    """
    parts = "_".join(f"{k}{params[k]:g}" for k in sorted(params))
    return f"{name}__{parts}"


def mmseqs_search(
    query: Path,
    target: Path,
    name: str,
    *,
    s: float = 7.5,
    e: float = 1e-3,
    max_seqs: int = 1000,
    work: Path | None = None,
    force: bool = False,
    quiet: bool = False,
) -> pd.DataFrame:
    """Cached MMseqs2 search returning a tidy hit frame with ``alncov`` added.

    The cache key includes the search parameters (see :func:`_cache_stem`).
    """
    from pymmseqs.commands import easy_search

    work = work or work_dir()
    work.mkdir(parents=True, exist_ok=True)
    stem = _cache_stem(name, {"s": s, "e": e, "max": max_seqs})
    out = work / f"{stem}.tsv"
    if force or not out.exists():
        with contextlib.redirect_stdout(
            io.StringIO()
        ):  # pymmseqs prints unconditionally
            easy_search(
                query_fasta=str(query),
                target_fasta_or_db=str(target),
                alignment_file=str(out),
                tmp_dir=str(work / f"tmp_{stem}"),
                s=s,
                e=e,
                max_seqs=max_seqs,
                format_output=SEARCH_COLS,
            )
    elif not quiet:
        print(f"  [cached] {out.name}")
    return add_alignment_coverage(pd.read_csv(out, sep="\t"))


def mmseqs_cluster_rep_count(
    fasta: Path, *, min_seq_id: float, prefix: Path, tmp: Path
) -> int:
    """Cluster ``fasta`` at ``min_seq_id`` and return the representative count."""
    from pymmseqs.commands import easy_cluster

    rep_fa = Path(f"{prefix}_rep_seq.fasta")
    if not rep_fa.exists():
        prefix.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            easy_cluster(
                fasta_files=str(fasta),
                cluster_prefix=str(prefix),
                tmp_dir=str(tmp),
                min_seq_id=min_seq_id,
            )
    return count_fasta_records(rep_fa)


def threshold_ladder(
    thresholds: Sequence[float] = (0.3, 0.5, 0.7, 0.9, 1.0),
    *,
    mm: Path | None = None,
    work: Path | None = None,
    cluster_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Re-cluster every real toxin family across a ladder of identity thresholds.

    Reuses the pipeline's own per-family input FASTAs (post-SignalP6, pre-clustering),
    so only ``min_seq_id`` varies. The pipeline's own cutoff is **not** re-clustered:
    that answer already exists in ``cluster_cluster.tsv`` and is seeded from
    ``cluster_stats``, which makes the ladder agree with the §2.1 numbers by
    construction (and saves ~80 s of the run).

    Cached to ``ladder__<thresholds>.csv``; the key includes the threshold set.
    """
    mm = mm or require_pipeline_caches()
    work = work or work_dir()
    work.mkdir(parents=True, exist_ok=True)
    stats = cluster_stats if cluster_stats is not None else read_cluster_cache(mm)

    key = "-".join(f"{t:g}" for t in sorted(thresholds))
    cache = work / f"ladder__{key}.csv"
    if cache.exists():
        print(f"  [cached] {cache.name}")
        return pd.read_csv(cache)

    fam_dirs = [
        d
        for d in sorted(mm.iterdir())
        if d.is_dir() and is_real_family(d.name) and (d / "input.fasta").exists()
    ]
    seeded = {
        r.family: (r.members, r.clusters)
        for r in stats[stats["is_family"]].itertuples()
    }
    records = []
    for thr in thresholds:
        for d in fam_dirs:
            if np.isclose(thr, PIPELINE_MIN_SEQ_ID) and d.name in seeded:
                n_in, n_rep = seeded[d.name]
            else:
                n_in = count_fasta_records(d / "input.fasta")
                n_rep = mmseqs_cluster_rep_count(
                    d / "input.fasta",
                    min_seq_id=thr,
                    prefix=work / "ladder" / f"{d.name}__{thr:g}",
                    tmp=work / "ladder" / "tmp",
                )
            records.append(
                {"threshold": thr, "family": d.name, "n_in": n_in, "n_rep": n_rep}
            )
        print(f"  clustered {len(fam_dirs)} toxin families @ {thr:.2f}")
    ladder = pd.DataFrame(records)
    ladder.to_csv(cache, index=False)
    return ladder


def sp_frame_comparison(
    tox_sp: pd.DataFrame,
    families: pd.Series,
    mixed: Sequence[str],
    *,
    work: Path | None = None,
    min_side: int = 5,
) -> dict:
    """Align SP-bearing members against SP-less members of their OWN family, ±trimming.

    Two batched searches, not two per family: every family's queries go into one FASTA
    tagged by family and are searched against one target DB of all SP-less members,
    then hits are filtered to same-family pairs. Equivalent to the per-family loop
    (a cross-family hit is discarded either way) and ~20x cheaper -- an ``easy_search``
    on a 12 KB file is almost entirely process and DB-build overhead.

    Returns the **paired per-query** comparison. See :func:`paired_alignment_shift` for
    why a per-family median comparison answers a much weaker question.
    """
    work = work or work_dir()
    work.mkdir(parents=True, exist_ok=True)
    df = tox_sp.assign(_family=list(families))
    df = df[df["_family"].isin(list(mixed))]

    queries, targets = [], []
    for fam, g in df.groupby("_family"):
        has, lacks = g[g["trimmed"]], g[~g["trimmed"]]
        if len(has) < min_side or len(lacks) < min_side:
            continue
        queries.append(has)
        targets.append(lacks)
    if not queries:
        return {"n_paired": 0, "n_families": 0}
    q_df, t_df = pd.concat(queries), pd.concat(targets)

    target_fa = write_fasta_subset(t_df, work / "sp_target.fasta")
    fam_of_id = pd.concat(
        [
            q_df.set_index(ensure_identifier_column(q_df)["identifier"])["_family"],
            t_df.set_index(ensure_identifier_column(t_df)["identifier"])["_family"],
        ]
    )

    per_mode = {}
    for mode, col in [("untrimmed", "Sequence"), ("trimmed", "seq_trimmed")]:
        q_fa = write_fasta_subset(q_df, work / f"sp_query_{mode}.fasta", seq_col=col)
        hits = mmseqs_search(
            q_fa, target_fa, f"sp_frames_{mode}", work=work, quiet=True
        )
        same_family = hits[
            hits["query"].map(fam_of_id).eq(hits["target"].map(fam_of_id))
        ]
        per_mode[mode] = best_hits(same_family, "query", rank="bits")

    shift = paired_alignment_shift(per_mode["untrimmed"], per_mode["trimmed"])
    shift["n_families"] = int(q_df["_family"].nunique())
    return shift


def sp_clustering_comparison(
    tox_sp: pd.DataFrame,
    families: pd.Series,
    mixed_by_size: Sequence[str],
    *,
    work: Path | None = None,
    top_n: int = 8,
    min_seq_id: float = PIPELINE_MIN_SEQ_ID,
) -> pd.DataFrame:
    """Representative counts when clustering a family untrimmed vs trimmed.

    Tests whether the mixed frame changes the one alignment-based pipeline step that
    is coverage-sensitive. Returns per-family counts so the *spread* is visible; see
    :func:`clustering_delta_spread` for why the net alone overstates the result.
    """
    work = work or work_dir()
    df = tox_sp.assign(_family=list(families))
    rows = {}
    for fam in list(mixed_by_size)[:top_n]:
        g = df[df["_family"] == fam]
        if g.empty:
            continue
        counts = {}
        for mode, col in [("untrimmed", "Sequence"), ("trimmed", "seq_trimmed")]:
            fa = write_fasta_subset(g, work / f"clust_{mode}.fasta", seq_col=col)
            safe = "".join(c if c.isalnum() else "_" for c in fam)[:40]
            counts[mode] = mmseqs_cluster_rep_count(
                fa,
                min_seq_id=min_seq_id,
                prefix=work / "clust" / f"{safe}__{mode}",
                tmp=work / "clust" / "tmp",
            )
        rows[fam] = {"n": int(len(g)), **counts}
    return pd.DataFrame(rows).T


def write_numbers(numbers: dict, path: Path | None = None) -> Path:
    """Write the audit's numbers to the tracked JSON manifest."""
    path = path or numbers_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(numbers), indent=2, sort_keys=True) + "\n")
    return path


def provenance() -> dict:
    """Commit, dirty flag and manifest hash, so numbers can be traced to a tree state."""
    from toxfam._git import git_commit_short, git_dirty
    from toxfam.data.split_manifest import manifest_sha256

    return {
        "commit": git_commit_short(),
        "dirty": bool(git_dirty()),
        "split_manifest_sha256": manifest_sha256(),
        "project_root": str(get_project_root()),
    }


def load_signal_peptides(df: pd.DataFrame) -> pd.DataFrame:
    """Attach SignalP6 trimming outcome from the pipeline's own per-sequence cache.

    Reuses ``preprocessing._seq_hash`` / ``_load_sp6_cache`` rather than re-deriving
    the MD5 keying. That matters more than it looks: if the key scheme ever changed, a
    hand-rolled lookup would return ``None`` for every sequence and this audit would
    report a 0% trim rate *as a finding* instead of failing.
    """
    from toxfam.data.preprocessing import _load_sp6_cache, _seq_hash

    cache = _load_sp6_cache()
    d = df.copy()
    hashes = d["Sequence"].map(_seq_hash)
    missing = int((~hashes.isin(cache.keys())).sum())
    if missing:
        raise ValueError(
            f"{missing} of {len(d)} sequences are absent from the SignalP6 cache "
            f"({_load_sp6_cache.__module__}). Re-run `toxfam preprocess`, or "
            "`toxfam download-data` to restore data/intermediate/sp6/."
        )
    d["mature"] = [cache.get(h) for h in hashes]
    d["trimmed"] = d["mature"].notna()  # SP6 reported a cleavage site at p > 0.8
    d["seq_trimmed"] = d["mature"].fillna(d["Sequence"])
    d["full_len"] = d["Sequence"].str.len()
    d["sp_len"] = d["full_len"] - d["seq_trimmed"].str.len()
    d["sp_frac"] = d["sp_len"] / d["full_len"]
    d["uni_sp"] = d["Signal peptide"].notna()  # UniProt's curated SIGNAL annotation
    return d


def signalp_vs_curation(df: pd.DataFrame) -> dict:
    """Agreement between SignalP6 at p>0.8 and UniProt's curated SIGNAL annotation."""
    tp = int((df["trimmed"] & df["uni_sp"]).sum())
    fp = int((df["trimmed"] & ~df["uni_sp"]).sum())
    fn = int((~df["trimmed"] & df["uni_sp"]).sum())
    tn = int((~df["trimmed"] & ~df["uni_sp"]).sum())
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "agreement": (tp + tn) / len(df) if len(df) else float("nan"),
    }


def cleavage_site_agreement(df: pd.DataFrame) -> dict:
    """How often SP6's cleavage site matches UniProt's, where both call a peptide."""
    uni_len = df["Signal peptide"].str.extract(r"SIGNAL 1\.\.(\d+)")[0].astype(float)
    both = df[df["trimmed"] & uni_len.notna()]
    if both.empty:
        return {"n": 0}
    delta = both["sp_len"].to_numpy() - uni_len.loc[both.index].to_numpy()
    return {
        "n": int(len(both)),
        "exact": float((delta == 0).mean()),
        "within_2": float((np.abs(delta) <= 2).mean()),
    }


def phospholipase_cohorts(tox: pd.DataFrame, nt: pd.DataFrame) -> dict:
    """Both the loose "phospholipase-related" cohort and the strict PLA2 one.

    The substring match also collects phospholipase B/D and the alpha-type PLA2
    *inhibitor* families, so quoting its counts under the heading "Phospholipase A2"
    overstates the PLA2 population by ~a third. Both are returned; F4's prose should
    quote whichever it names.
    """
    col = "Protein families"
    loose_t = tox[tox[col].str.contains("phospholipase", case=False, na=False)].copy()
    loose_n = nt[nt[col].str.contains("phospholipase", case=False, na=False)].copy()
    strict = lambda d: d[  # noqa: E731 - one-line predicate, clearer inline
        first_family(d[col]).str.startswith("Phospholipase A2 family")
    ].copy()
    return {
        "loose_tox": loose_t,
        "loose_nt": loose_n,
        "strict_tox": strict(loose_t),
        "strict_nt": strict(loose_n),
    }


def run_full_audit(*, thresholds: Sequence[float] = (0.3, 0.5, 0.7, 0.9, 1.0)) -> dict:
    """Run every measurement in the audit and return frames + a flat numbers dict.

    The single entry point shared by ``make preprocessing-audit`` and the narrative
    notebook, so the prose and the manifest cannot be computed two different ways.
    """
    mm = require_pipeline_caches()
    work = work_dir()
    work.mkdir(parents=True, exist_ok=True)
    all_fa = mm / "representatives" / "all.fasta"

    print("[1/7] raw frames + length cut")
    tox, nt, funnel = load_raw_frames()

    print("[2/7] per-family clustering cache")
    stats = read_cluster_cache(mm)
    lanes = lane_reduction(stats)
    conc = redundancy_concentration(stats)

    print("[3/7] threshold ladder")
    ladder = threshold_ladder(thresholds, mm=mm, work=work, cluster_stats=stats)
    summary = ladder_summary(ladder)

    print("[4/7] shared family vocabulary + phospholipase case study")
    tox_first = first_family(tox["Protein families"])
    nt_first = first_family(nt["Protein families"].fillna(""))
    shared = shared_family_table(tox_first, nt_first[nt_first != ""])
    pla = phospholipase_cohorts(tox, nt)
    pla_hits = mmseqs_search(
        write_fasta_subset(pla["loose_tox"], work / "pla_tox.fasta"),
        write_fasta_subset(pla["loose_nt"], work / "pla_nontox.fasta"),
        "pla_cross",
        work=work,
    )
    pla_best = best_hits(pla_hits, "target", min_cov=DUP_COV)
    twins = near_duplicate_pairs(pla_hits, key="target")

    print("[5/7] representative lanes + nearest-neighbour audit")
    reps = load_representatives(mm)
    fam_of = reps.set_index("identifier")["Protein families"]
    split_of = reps.set_index("identifier")["Split"]
    tox_rep_ids = set(reps.loc[reps["is_toxin_lane"], "identifier"])
    n_tox_rep = len(tox_rep_ids)

    ftox, fnt = work / "rep_tox.fasta", work / "rep_nontox.fasta"
    split_fasta_by(
        all_fa,
        lambda i: "tox" if i in tox_rep_ids else "nontox",
        {"tox": ftox, "nontox": fnt},
    )
    # One search against ALL representatives serves both questions: the non-toxin
    # subset answers "does a non-toxin homolog exist?", the full set answers "is the
    # nearest neighbour a non-toxin?". HBI's parameters, because the second question
    # is a statement about what best-hit transfer does.
    nn_hits = mmseqs_search(
        ftox, all_fa, "tox_vs_all", s=9.0, e=float("inf"), max_seqs=1000, work=work
    )
    nn = nearest_neighbour_labels(nn_hits, fam_of)
    nn_fail = int(nn["nn_is_nontox"].sum())
    cross_lane = nn_hits[
        (nn_hits["query"] != nn_hits["target"])
        & (nn_hits["target"].map(fam_of) == NONTOX_LABEL)
    ]
    glob_best = best_hits(cross_lane, "query", min_cov=DUP_COV)

    twin_splits = twins.assign(
        tox_split=twins["query"].map(split_of), nt_split=twins["target"].map(split_of)
    )
    both_rep = twin_splits.dropna(subset=["tox_split", "nt_split"])
    straddle = split_straddling_pairs(both_rep)

    print("[6/7] residual train/test homology")
    ftrain, ftest = work / "rep_train.fasta", work / "rep_test.fasta"
    split_fasta_by(all_fa, split_of.get, {"train": ftrain, "test": ftest})
    tt = mmseqs_search(ftest, ftrain, "test_vs_train", s=5.7, max_seqs=100, work=work)
    tt_best = best_hits(tt, "query", min_cov=DUP_COV)
    tt_best = tt_best.assign(family=tt_best["query"].map(fam_of))
    test_ids = set(reps.loc[reps["Split"] == "test", "identifier"])
    tox_test = {i for i in test_ids if fam_of.get(i) != NONTOX_LABEL}
    tox_only = tt_best[tt_best["family"] != NONTOX_LABEL]
    hi_pairs = tt_best[tt_best["fident"] >= DUP_IDENT].assign(
        test_family=lambda d: d["query"].map(fam_of),
        train_family=lambda d: d["target"].map(fam_of),
    )
    hi_pairs["cross_family"] = hi_pairs["test_family"] != hi_pairs["train_family"]

    print("[7/7] signal peptides")
    tox_sp, nt_sp = load_signal_peptides(tox), load_signal_peptides(nt)
    # The mixed-frame claim is about the model's *label space*, so it must use the
    # pipeline's consolidated families (which is also what collapses rare ones into
    # `other`), not the raw first-listed strings §3 needs.
    tox_fam_norm = normalize_protein_families(tox_sp.copy())["Protein families"]
    fam_sp = mixed_frame_families(tox_fam_norm, tox_sp["trimmed"])
    shortcut = signal_peptide_shortcut(tox_sp["trimmed"], nt_sp["trimmed"])
    agree_tox, agree_nt = signalp_vs_curation(tox_sp), signalp_vs_curation(nt_sp)
    cleavage = cleavage_site_agreement(tox_sp)
    mixed_names = list(fam_sp[fam_sp["mixed"]].index)
    frame_shift = sp_frame_comparison(tox_sp, tox_fam_norm, mixed_names, work=work)
    mixed_by_size = list(
        fam_sp[fam_sp["mixed"]].sort_values("n", ascending=False).index
    )
    clust = sp_clustering_comparison(tox_sp, tox_fam_norm, mixed_by_size, work=work)
    clust_spread = clustering_delta_spread(clust)

    numbers = {
        "provenance": provenance(),
        "funnel": funnel,
        "lanes": lanes,
        "redundancy": conc,
        "ladder": {
            "thresholds": list(thresholds),
            "min_members": MIN_MEMBERS,
            "rows": summary.to_dict("records"),
        },
        "shared_families": {
            "n_shared": int(len(shared)),
            "toxins_in_shared": int(shared["toxins"].sum()),
            "toxin_share": float(shared["toxins"].sum() / len(tox)),
            "nt_without_family": int(nt["Protein families"].isna().sum()),
        },
        "phospholipase": {
            "loose_tox": int(len(pla["loose_tox"])),
            "loose_nt": int(len(pla["loose_nt"])),
            "strict_tox": int(len(pla["strict_tox"])),
            "strict_nt": int(len(pla["strict_nt"])),
            "nt_with_venom_homolog": int(len(pla_best)),
            "twins": int(len(twins)),
            **{f"twin_{k}": v for k, v in straddle.items()},
        },
        "nearest_neighbour": {
            "n_toxin_reps": n_tox_rep,
            "with_nontoxin_homolog": int(len(glob_best)),
            "with_nontoxin_homolog_frac": float(len(glob_best) / n_tox_rep),
            "nearest_is_nontoxin": nn_fail,
            "nearest_is_nontoxin_frac": float(nn_fail / n_tox_rep),
            "params": "hbi defaults: s=9.0, e=inf, max_seqs=1000, min-evalue, no cov filter",
        },
        "train_test_homology": {
            "n_toxin_test": int(len(tox_test)),
            "ge70": int((tox_only["fident"] >= 0.7).sum()),
            "ge70_frac": float((tox_only["fident"] >= 0.7).sum() / len(tox_test)),
            "ge90": int((tox_only["fident"] >= 0.9).sum()),
            "n_test_reps_ge90": int(len(hi_pairs)),
            "ge90_cross_family": int(hi_pairs["cross_family"].sum()),
            "ge90_same_family": int((~hi_pairs["cross_family"]).sum()),
        },
        "signal_peptide": {
            "tox_trim_rate": float(tox_sp["trimmed"].mean()),
            "nt_trim_rate": float(nt_sp["trimmed"].mean()),
            "sp_frac_tox": float(tox_sp.loc[tox_sp["trimmed"], "sp_frac"].median()),
            "sp_frac_nt": float(nt_sp.loc[nt_sp["trimmed"], "sp_frac"].median()),
            "families_total": int(len(fam_sp)),
            "families_mixed": int(fam_sp["mixed"].sum()),
            "mixed_share": float(
                fam_sp.loc[fam_sp["mixed"], "n"].sum() / fam_sp["n"].sum()
            ),
            "shortcut": shortcut,
            "curation_toxins": agree_tox,
            "curation_nontoxins": agree_nt,
            "cleavage": cleavage,
            "frame_shift": frame_shift,
            "clustering_spread": clust_spread,
        },
    }

    return {
        "numbers": numbers,
        "tox": tox,
        "nt": nt,
        "stats": stats,
        "ladder": ladder,
        "summary": summary,
        "shared": shared,
        "pla": pla,
        "pla_hits": pla_hits,
        "pla_best": pla_best,
        "twins": twin_splits,
        "reps": reps,
        "fam_of": fam_of,
        "split_of": split_of,
        "nn": nn,
        "glob_best": glob_best,
        "tt_best": tt_best,
        "hi_pairs": hi_pairs,
        "tox_sp": tox_sp,
        "nt_sp": nt_sp,
        "fam_sp": fam_sp,
        "tox_fam_norm": tox_fam_norm,
        "clust": clust,
    }


def main() -> None:
    """Run the audit and write the tracked numbers manifest."""
    result = run_full_audit()
    path = write_numbers(result["numbers"])
    n = result["numbers"]
    print(f"\nwrote {path.relative_to(get_project_root())}")
    print(
        f"  nearest neighbour is a non-toxin: "
        f"{n['nearest_neighbour']['nearest_is_nontoxin']:,} / "
        f"{n['nearest_neighbour']['n_toxin_reps']:,} "
        f"({n['nearest_neighbour']['nearest_is_nontoxin_frac']:.1%})"
    )


if __name__ == "__main__":
    main()
