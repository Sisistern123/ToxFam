"""Tests for paper.preprocessing_audit — preprocessing-decision statistics.

Hermetic by construction: every test here builds its own tiny frames. The audit's
I/O drivers read ``data/intermediate/mmseqs/``, which is gitignored and absent in CI,
so they are exercised by ``make preprocessing-audit`` rather than by the suite.

Several of these tests pin behaviour that a previous version of this analysis got
wrong, and they say so, so the regression cannot come back quietly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paper.preprocessing_audit import (
    DUP_COV,
    DUP_IDENT,
    NONTOX_LABEL,
    OTHER_LABEL,
    add_alignment_coverage,
    best_hits,
    cluster_reduction_stats,
    clustering_delta_spread,
    first_family,
    homology_band_table,
    identity_ladder,
    is_real_family,
    ladder_summary,
    lane_reduction,
    length_cut_effect,
    mixed_frame_families,
    near_duplicate_pairs,
    nearest_neighbour_labels,
    paired_alignment_shift,
    real_family_mask,
    redundancy_concentration,
    shared_family_table,
    signal_peptide_shortcut,
    split_straddling_pairs,
    to_jsonable,
)

# ---------------------------------------------------------------------------
# The pseudo-class gate
# ---------------------------------------------------------------------------


def test_is_real_family_excludes_both_pseudo_classes():
    assert is_real_family("Conotoxin superfamily")
    assert not is_real_family(NONTOX_LABEL)
    assert not is_real_family(OTHER_LABEL)


def test_real_family_mask_over_a_series():
    labels = ["A", OTHER_LABEL, "B", NONTOX_LABEL]
    assert real_family_mask(labels).tolist() == [True, False, True, False]


def _cluster_records():
    """`other` is deliberately large enough to enter any top-5 by membership.

    This mirrors the real cache, where `other` is the 4th-largest bucket -- which is
    exactly why treating "not nontox" as "is a toxin family" corrupts F1/F2/F8.
    """
    return [
        {"family": "Fam-A", "members": 1000, "clusters": 700},  # removed 300
        {"family": "Fam-B", "members": 900, "clusters": 700},  # removed 200
        {"family": "Fam-C", "members": 800, "clusters": 700},  # removed 100
        {"family": OTHER_LABEL, "members": 700, "clusters": 300},  # removed 400
        {"family": "Fam-D", "members": 50, "clusters": 40},  # removed 10
        {"family": "Fam-E", "members": 30, "clusters": 30},  # removed 0
        {"family": NONTOX_LABEL, "members": 5000, "clusters": 3000},
    ]


def test_cluster_reduction_stats_flags_rather_than_drops():
    stats = cluster_reduction_stats(_cluster_records())
    assert set(stats.columns) >= {"removed", "reduction", "is_family"}
    assert stats.loc[stats["family"] == "Fam-A", "removed"].item() == 300
    assert not stats.loc[stats["family"] == OTHER_LABEL, "is_family"].item()
    # Nothing is silently dropped: the caller chooses.
    assert len(stats) == len(_cluster_records())


def test_redundancy_concentration_excludes_other_from_the_top_n():
    """Regression: `other` had been counted as a toxin family and it is top-5 by size.

    Real families remove 300+200+100+10+0 = 610; the top 3 (Fam-A/B/C) remove 600.
    Including `other` would put its 400 removals in the numerator and 1010 in the
    denominator, giving a different -- and meaningless -- concentration.
    """
    stats = cluster_reduction_stats(_cluster_records())
    conc = redundancy_concentration(stats, top_n=3)
    assert conc["n_families"] == 5  # A, B, C, D, E -- not other, not nontox
    assert conc["total_removed"] == 610
    assert conc["top_n_share"] == pytest.approx(600 / 610)
    assert conc["zero_redundancy"] == 1  # Fam-E
    assert OTHER_LABEL not in conc["top_families"]


def test_lane_reduction_reports_other_separately():
    lanes = lane_reduction(cluster_reduction_stats(_cluster_records()))
    assert lanes["toxin"]["families"] == 5
    assert lanes["other"]["families"] == 1
    assert lanes["nontox"]["members"] == 5000
    assert lanes["nontox"]["removed_frac"] == pytest.approx(1 - 3000 / 5000)


# ---------------------------------------------------------------------------
# Threshold ladder / label-space survival
# ---------------------------------------------------------------------------


def test_ladder_summary_counts_families_over_the_floor():
    ladder = pd.DataFrame(
        [
            {"threshold": 0.9, "family": "A", "n_in": 100, "n_rep": 40},
            {"threshold": 0.9, "family": "B", "n_in": 50, "n_rep": 12},
            {"threshold": 0.9, "family": "C", "n_in": 20, "n_rep": 9},
            {"threshold": 0.3, "family": "A", "n_in": 100, "n_rep": 11},
            {"threshold": 0.3, "family": "B", "n_in": 50, "n_rep": 4},
            {"threshold": 0.3, "family": "C", "n_in": 20, "n_rep": 2},
        ]
    )
    s = ladder_summary(ladder, min_members=10).set_index("threshold")
    assert s.loc[0.9, "families_kept"] == 2  # A(40), B(12); C(9) is below the floor
    assert s.loc[0.9, "families_dissolved"] == 1
    assert s.loc[0.9, "toxin_reps"] == 61
    assert s.loc[0.3, "families_kept"] == 1  # only A survives
    assert s.loc[0.3, "families_dissolved"] == 2
    # Representatives dissolved into `other` at 0.3: B(4) + C(2)
    assert s.loc[0.3, "seqs_into_other"] == 6


def test_ladder_summary_dissolved_uses_the_full_family_set():
    """families_kept + families_dissolved must equal the family count at every rung."""
    ladder = pd.DataFrame(
        [
            {"threshold": t, "family": f, "n_in": 100, "n_rep": n}
            for t, f, n in [
                (0.9, "A", 30),
                (0.9, "B", 30),
                (0.5, "A", 3),
                (0.5, "B", 3),
            ]
        ]
    )
    s = ladder_summary(ladder, min_members=10)
    assert (s["families_kept"] + s["families_dissolved"] == 2).all()


# ---------------------------------------------------------------------------
# Family strings / shared vocabulary
# ---------------------------------------------------------------------------


def test_first_family_takes_the_first_listed_token():
    s = pd.Series(
        [
            "  Phospholipase A2 family, Group I subfamily  ",
            "Conotoxin superfamily; something else",
            "Three-finger toxin family",
        ]
    )
    assert first_family(s).tolist() == [
        "Phospholipase A2 family",
        "Conotoxin superfamily",
        "Three-finger toxin family",
    ]


def test_shared_family_table_intersects_both_sides():
    tox = pd.Series(["PLA2", "PLA2", "3FTx", "3FTx", "3FTx"])
    nt = pd.Series(["PLA2", "Peptidase", "Peptidase", "Peptidase"])
    shared = shared_family_table(tox, nt)
    assert shared["family"].tolist() == ["PLA2"]
    assert shared.loc[0, "toxins"] == 2
    assert shared.loc[0, "non_toxins"] == 1
    # smaller_side is what makes a family viable as a two-sided case study
    assert shared.loc[0, "smaller_side"] == 1


def test_shared_family_table_ranks_by_smaller_side_correctly():
    """A family huge on one side but empty-ish on the other is a poor case study."""
    tox = pd.Series(["Big"] * 100 + ["Balanced"] * 20)
    nt = pd.Series(["Big"] * 2 + ["Balanced"] * 18)
    shared = shared_family_table(tox, nt)
    best = shared.sort_values("smaller_side", ascending=False).iloc[0]
    assert best["family"] == "Balanced"


# ---------------------------------------------------------------------------
# Hits, coverage, near-duplicates
# ---------------------------------------------------------------------------


def _hits():
    return pd.DataFrame(
        {
            "query": ["q1", "q1", "q2", "q2", "q3"],
            "target": ["t1", "t2", "t1", "t3", "q3"],
            "fident": [0.95, 0.99, 0.70, 0.92, 1.00],
            "alnlen": [100, 20, 90, 95, 100],
            "qlen": [100, 100, 100, 100, 100],
            "tlen": [100, 100, 100, 100, 100],
            "evalue": [1e-50, 1e-3, 1e-20, 1e-40, 0.0],
            "bits": [200, 30, 150, 180, 300],
        }
    )


def test_add_alignment_coverage_uses_the_longer_sequence():
    h = add_alignment_coverage(
        pd.DataFrame({"alnlen": [50], "qlen": [100], "tlen": [200]})
    )
    assert h["alncov"].item() == pytest.approx(0.25)


def test_near_duplicate_pairs_requires_identity_and_coverage():
    """q1->t2 is 99% identical but over 20/100 residues: not a duplicate."""
    dup = near_duplicate_pairs(
        add_alignment_coverage(_hits()),
        key="query",
        min_ident=DUP_IDENT,
        min_cov=DUP_COV,
    )
    pairs = set(zip(dup["query"], dup["target"]))
    assert ("q1", "t2") not in pairs
    assert ("q1", "t1") in pairs


def test_best_hits_rank_choices_can_disagree():
    h = add_alignment_coverage(_hits())
    by_ident = best_hits(h, "query", rank="fident").set_index("query")
    by_evalue = best_hits(h, "query", rank="evalue").set_index("query")
    # q1's most-identical hit is the short, weak one; its best e-value is the real one.
    assert by_ident.loc["q1", "target"] == "t2"
    assert by_evalue.loc["q1", "target"] == "t1"


def test_nearest_neighbour_labels_drops_self_hits():
    h = add_alignment_coverage(_hits())
    fam = {"t1": "Fam-A", "t2": NONTOX_LABEL, "t3": NONTOX_LABEL, "q3": "Fam-B"}
    nn = nearest_neighbour_labels(h, fam)
    assert "q3" not in set(nn["query"])  # its only hit was itself


def test_nearest_neighbour_defaults_match_the_hbi_baseline():
    """Regression: a coverage filter here silently measures a different protein.

    q1's genuine nearest neighbour (best e-value) is the toxin t1 at 100% coverage.
    A 0.80 coverage floor combined with identity ranking would instead select the
    20-residue non-toxin hit t2 and count q1 as a transfer failure.
    """
    h = add_alignment_coverage(_hits())
    fam = {"t1": "Fam-A", "t2": NONTOX_LABEL, "t3": NONTOX_LABEL}
    nn = nearest_neighbour_labels(h, fam).set_index("query")
    assert not nn.loc["q1", "nn_is_nontox"]
    # ...whereas ranking by identity flips it, which is the defect being pinned.
    flipped = nearest_neighbour_labels(h, fam, rank="fident").set_index("query")
    assert flipped.loc["q1", "nn_is_nontox"]


def test_identity_ladder_is_monotone_and_shares_a_denominator():
    lad = identity_ladder(
        pd.Series([0.95, 0.85, 0.75, 0.4]),
        n_total=10,
        label="reps",
        thresholds=(0.9, 0.7, 0.3),
    )
    assert lad["reps"].tolist() == [1, 3, 4]
    assert lad["% of reps"].tolist() == ["10.0%", "30.0%", "40.0%"]


def test_homology_band_table_partitions_the_subset():
    best = pd.DataFrame({"fident": [0.95, 0.8, 0.75, 0.6, 0.2]})
    row = homology_band_table(best, subset_label="toxin test reps", n_total=8)
    assert row["no hit"] == 3
    assert row["90%-100%"] == 1
    assert row["70%-90%"] == 2
    assert row["50%-70%"] == 1
    assert row["0%-50%"] == 1
    assert sum(row[k] for k in ["90%-100%", "70%-90%", "50%-70%", "0%-50%"]) == len(
        best
    )


# ---------------------------------------------------------------------------
# Signal peptides
# ---------------------------------------------------------------------------


def test_paired_alignment_shift_detects_per_query_movement():
    """Regression: per-family medians hid that individual alignments do move."""
    untrimmed = pd.DataFrame(
        {
            "query": ["a", "b", "c", "d"],
            "alnlen": [100, 100, 100, 100],
            "bits": [200, 200, 200, 200],
            "fident": [0.9, 0.9, 0.9, 0.9],
            "alncov": [0.5, 0.5, 0.5, 0.5],
        }
    )
    trimmed = untrimmed.assign(
        alnlen=[100, 100, 105, 95],  # 2 of 4 move
        bits=[200, 210, 190, 200],  # 2 of 4 move
        alncov=[0.8, 0.8, 0.8, 0.8],  # all move (shrunken denominator)
    )
    shift = paired_alignment_shift(untrimmed, trimmed)
    assert shift["n_paired"] == 4
    assert shift["alnlen_changed"] == 2
    assert shift["bits_changed"] == 2
    assert shift["alncov_changed"] == 4
    # The median is unmoved even though half the alignments changed -- the exact
    # artefact that produced the "provably inert to alignment" over-claim.
    assert shift["alnlen_median_delta"] == 0.0


def test_paired_alignment_shift_only_compares_queries_present_in_both():
    a = pd.DataFrame(
        {
            "query": ["x", "y"],
            "alnlen": [10, 10],
            "bits": [1, 1],
            "fident": [0.5, 0.5],
            "alncov": [0.5, 0.5],
        }
    )
    b = pd.DataFrame(
        {
            "query": ["y", "z"],
            "alnlen": [10, 10],
            "bits": [1, 1],
            "fident": [0.5, 0.5],
            "alncov": [0.5, 0.5],
        }
    )
    assert paired_alignment_shift(a, b)["n_paired"] == 1


def test_signal_peptide_shortcut_arithmetic():
    # 8 of 10 toxins carry an SP; 2 of 10 non-toxins do.
    tox = pd.Series([True] * 8 + [False] * 2)
    nt = pd.Series([True] * 2 + [False] * 8)
    r = signal_peptide_shortcut(tox, nt)
    assert r["tox_rate"] == pytest.approx(0.8)
    assert r["nt_rate"] == pytest.approx(0.2)
    assert r["odds_ratio"] == pytest.approx((8 / 2) / (2 / 8))
    assert r["precision"] == pytest.approx(8 / 10)  # 8 TP, 2 FP
    assert r["recall"] == pytest.approx(0.8)
    assert r["base_rate"] == pytest.approx(0.5)
    assert r["lift_factor"] == pytest.approx(1.6)


def test_mixed_frame_families_flags_only_the_middle_band():
    families = ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + [OTHER_LABEL] * 10
    trimmed = (
        [True] * 10  # A: 100% trimmed -> homogeneous
        + [False] * 10  # B: 0% -> homogeneous
        + [True] * 5
        + [False] * 5  # C: 50% -> mixed
        + [True] * 5
        + [False] * 5  # other: excluded entirely
    )
    fam = mixed_frame_families(pd.Series(families), pd.Series(trimmed), min_members=10)
    assert OTHER_LABEL not in fam.index
    assert fam.loc["C", "mixed"]
    assert not fam.loc["A", "mixed"]
    assert not fam.loc["B", "mixed"]


def test_mixed_frame_families_respects_the_member_floor():
    fam = mixed_frame_families(
        pd.Series(["Small"] * 4), pd.Series([True, False, True, False]), min_members=10
    )
    assert fam.empty


# ---------------------------------------------------------------------------
# Splits and clustering spread
# ---------------------------------------------------------------------------


def test_split_straddling_pairs_counts_both_directions():
    """Regression: only train-toxin/eval-nontoxin was counted, halving the leakage."""
    pairs = pd.DataFrame(
        {
            "tox_split": ["train", "test", "train", "train"],
            "nt_split": ["test", "train", "train", "val"],
        }
    )
    r = split_straddling_pairs(pairs)
    assert r["different_split"] == 3
    assert r["same_split"] == 1
    assert r["toxin_train_nontoxin_eval"] == 2
    assert r["toxin_eval_nontoxin_train"] == 1
    assert r["cross_train_eval"] == 3


def test_split_straddling_pairs_ignores_unassigned_members():
    pairs = pd.DataFrame({"tox_split": ["train", None], "nt_split": ["test", "test"]})
    assert split_straddling_pairs(pairs)["n_pairs"] == 1


def test_clustering_delta_spread_exposes_cancelling_shifts():
    """A ~0 net can hide large per-family movement in both directions."""
    clust = pd.DataFrame(
        {"untrimmed": [220, 731, 100], "trimmed": [190, 753, 100]},
        index=["Long", "Conotoxin", "Flat"],
    )
    r = clustering_delta_spread(clust)
    assert r["net"] == -8
    assert abs(r["net_frac"]) < 0.01  # looks like "a wash"
    assert r["n_down"] == 1 and r["n_up"] == 1 and r["n_unchanged"] == 1
    # ...but one family moved by 13.6%, which the net conceals.
    assert r["max_rel_decrease"] == pytest.approx(-30 / 220)
    assert r["mean_abs_rel"] > abs(r["net_frac"])


# ---------------------------------------------------------------------------
# Length cut + serialization
# ---------------------------------------------------------------------------


def test_length_cut_effect_matches_the_pipeline_rule():
    lengths = pd.Series(list(range(1, 101)))  # 1..100
    cut = length_cut_effect(lengths, pct=0.01)
    assert cut["cutoff"] == 100  # ceil(100*0.01)=1 -> the single longest
    assert cut["n_removed"] == 0  # the cutoff itself is retained (<=)
    assert cut["n_after"] == 100


def test_length_cut_effect_removes_a_heavy_tail_without_moving_the_median():
    lengths = pd.Series([100] * 98 + [5000, 6000])
    # pct=0.02 -> the two longest set the ceiling at 5000, so only 6000 is dropped.
    cut = length_cut_effect(lengths, pct=0.02)
    assert cut["median_before"] == cut["median_after"] == 100
    assert cut["cutoff"] == 5000
    assert cut["n_removed"] == 1
    # 1 of 100 sequences, but 6000/20800 = 29% of the total residue mass -- the
    # asymmetry F7 says the manuscript's "1% of sequences" framing understates.
    assert cut["residue_mass_removed"] == pytest.approx(6000 / 20800)


def test_to_jsonable_unwraps_numpy_scalars():
    import json

    payload = {"a": np.int64(3), "b": [np.float64(1.5)], "c": {"d": np.bool_(True)}}
    out = to_jsonable(payload)
    assert json.dumps(out) == '{"a": 3, "b": [1.5], "c": {"d": true}}'
