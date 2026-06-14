"""Tests for toxfam.evaluation.manuscript — manuscript statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from toxfam.evaluation.manuscript import (
    accuracy_by_identity_bins,
    accuracy_by_length_bins,
    adjudication_summary,
    aligned_correctness,
    binary_reliability,
    correctness,
    macro_f1_by_support,
    macro_f1_conventions,
    mcnemar_test,
    paired_bootstrap_accuracy_diff,
    per_family_f1_difference,
    rolling_accuracy_vs_length,
    subset_accuracy,
    toxin_mask,
)


def _toy_preds():
    # 4 samples: a correct on all but #3; b correct on #0 only
    return pd.DataFrame(
        {
            "identifier": ["x0", "x1", "x2", "x3"],
            "actual_label": ["A", "A", "B", "B"],
            "predicted_label": ["A", "A", "B", "A"],  # a: wrong on x3
        }
    )


def test_subset_accuracy_all():
    df = _toy_preds()
    assert subset_accuracy(df) == pytest.approx(0.75)


def test_subset_accuracy_masked():
    df = _toy_preds()
    mask = df["actual_label"] == "A"  # x0,x1 both correct
    assert subset_accuracy(df, mask) == pytest.approx(1.0)


def test_toxin_mask_case_insensitive():
    # Mixed toxin / non-toxin, including a non-standard-cased non-toxin label "NonTox".
    df = pd.DataFrame(
        {
            "actual_label": ["A", "nontox", "B", "NonTox", "C"],
            "predicted_label": ["A", "A", "nontoxic", "C", "C"],
        }
    )
    # actual: non-toxin at idx1 (nontox) and idx3 (NonTox) -> False there.
    assert toxin_mask(df).tolist() == [True, False, True, False, True]
    # predicted: only idx2 ("nontoxic") is a non-toxin label.
    assert toxin_mask(df, label_col="predicted_label").tolist() == [True, True, False, True, True]


def test_aligned_correctness_paired_and_disjoint():
    a = pd.DataFrame(
        {
            "identifier": ["x0", "x1", "x2", "x3"],
            "actual_label": ["A", "A", "B", "B"],
            "predicted_label": ["A", "A", "B", "A"],
        }
    )
    b = pd.DataFrame(
        {
            "identifier": ["x0", "x1", "x2", "x3"],
            "actual_label": ["A", "A", "B", "B"],
            "predicted_label": ["A", "B", "B", "B"],
        }
    )
    # (a) identically-ordered frames return correctness() on each.
    ca, cb = aligned_correctness(a, b)
    assert np.array_equal(ca, correctness(a))
    assert np.array_equal(cb, correctness(b))

    # (b) a reordered b is realigned to a's identifier order.
    b_re = b.iloc[[2, 0, 3, 1]].reset_index(drop=True)
    ca2, cb2 = aligned_correctness(a, b_re)
    assert np.array_equal(ca2, correctness(a))
    assert np.array_equal(cb2, correctness(b))  # aligned back to a's order

    # (c) disjoint identifier sets raise ValueError.
    b_disjoint = b.assign(identifier=["y0", "y1", "y2", "y3"])
    with pytest.raises(ValueError):
        aligned_correctness(a, b_disjoint)


def test_mcnemar_counts_and_significance():
    correct_a = np.array([1, 1, 1, 0, 1])
    correct_b = np.array([1, 0, 0, 0, 1])  # b wrong where a right on idx1,2
    with pytest.warns(UserWarning):  # n_discordant=2 < 25 with exact=False
        res = mcnemar_test(correct_a, correct_b)
    assert res["b01"] == 2  # a right, b wrong
    assert res["b10"] == 0  # a wrong, b right
    assert res["n_discordant"] == 2
    assert res["method"] == "chi2_continuity"
    assert res["chi2"] == pytest.approx(0.5)
    assert res["p_value"] == pytest.approx(0.4795001221869534, abs=1e-4)


def test_mcnemar_large_case():
    a = np.array([1] * 20 + [0] * 5 + [1] * 10)
    b = np.array([0] * 20 + [1] * 5 + [1] * 10)  # b01=20, b10=5, n=25
    res = mcnemar_test(a, b)
    assert res["b01"] == 20
    assert res["b10"] == 5
    assert res["n_discordant"] == 25
    assert res["method"] == "chi2_continuity"
    assert res["chi2"] == pytest.approx(7.84)
    assert res["p_value"] == pytest.approx(0.005110260660855867, abs=1e-4)


def test_mcnemar_exact_binomial():
    correct_a = np.array([1, 1, 1, 0, 1])
    correct_b = np.array([1, 0, 0, 0, 1])  # b01=2, b10=0, n=2
    res = mcnemar_test(correct_a, correct_b, exact=True)
    assert res["method"] == "exact_binomial"
    assert res["p_value"] == pytest.approx(0.5, abs=1e-4)  # two-sided exact binomial, n=2
    # default (chi2) path differs in p_value and method.
    with pytest.warns(UserWarning):
        res_default = mcnemar_test(correct_a, correct_b)
    assert res_default["method"] == "chi2_continuity"


def test_paired_bootstrap_diff_sign_and_ci():
    rng_correct_a = np.array([1] * 90 + [0] * 10)
    rng_correct_b = np.array([1] * 80 + [0] * 20)
    res = paired_bootstrap_accuracy_diff(rng_correct_a, rng_correct_b, n_boot=2000, seed=42)
    assert res["diff"] == pytest.approx(0.10, abs=1e-9)
    assert res["ci_low"] < res["diff"] < res["ci_high"]


def test_paired_bootstrap_deterministic_and_pinned():
    ca = np.array([1] * 90 + [0] * 10)
    cb = np.array([1] * 80 + [0] * 20)
    r1 = paired_bootstrap_accuracy_diff(ca, cb, n_boot=500, seed=7)
    r2 = paired_bootstrap_accuracy_diff(ca, cb, n_boot=500, seed=7)
    assert r1 == r2  # same seed -> identical dict
    assert r1["diff"] == pytest.approx(0.10, abs=1e-9)
    assert r1["ci_low"] == pytest.approx(0.040000000000000036, abs=1e-9)
    assert r1["ci_high"] == pytest.approx(0.16524999999999976, abs=1e-9)
    assert r1["n"] == 100


def test_accuracy_by_length_bins_basic():
    preds = pd.DataFrame(
        {
            "identifier": [f"x{i}" for i in range(6)],
            "actual_label": ["A"] * 6,
            "predicted_label": ["A", "B", "A", "A", "B", "A"],  # wrong at idx1,4
        }
    )
    lengths = pd.Series([10, 20, 40, 60, 80, 200], index=preds["identifier"])
    out = accuracy_by_length_bins(preds, lengths, bins=[0, 30, 100, 1000])
    assert list(out["n"]) == [2, 3, 1]
    assert out.loc[out["bin_label"] == "0-30", "accuracy"].iloc[0] == pytest.approx(0.5)


def test_rolling_accuracy_monotone_length_sorted():
    preds = pd.DataFrame(
        {
            "identifier": [f"x{i}" for i in range(5)],
            "actual_label": ["A"] * 5,
            "predicted_label": ["A"] * 5,
        }
    )
    lengths = pd.Series([5, 4, 3, 2, 1], index=preds["identifier"])
    out = rolling_accuracy_vs_length(preds, lengths, window=2)
    assert (out["length"].values == np.array([1, 2, 3, 4, 5])).all()
    assert (out["accuracy"].values == 1.0).all()


def _two_method_preds():
    actual = ["A"] * 5 + ["B"] * 5 + ["nontox"] * 5
    a_pred = ["A"] * 5 + ["B"] * 4 + ["A"] + ["nontox"] * 5          # NN-like
    b_pred = ["A"] * 4 + ["no hit"] + ["B"] * 5 + ["nontox"] * 5     # HBI-like (one no hit)
    return (
        pd.DataFrame({"identifier": [f"x{i}" for i in range(15)], "actual_label": actual, "predicted_label": a_pred}),
        pd.DataFrame({"identifier": [f"x{i}" for i in range(15)], "actual_label": actual, "predicted_label": b_pred}),
    )


def test_per_family_f1_difference_columns():
    a, b = _two_method_preds()
    out = per_family_f1_difference(a, b, class_list=["A", "B", "nontox"])
    assert set(["family", "f1_a", "f1_b", "diff", "support"]).issubset(out.columns)
    assert (out["diff"] == (out["f1_a"] - out["f1_b"])).all()
    assert "nontox" not in set(out["family"])  # non-toxin excluded from family view
    fams = out.set_index("family")
    # Family A: a slightly better; family B: b perfect.
    assert fams.loc["A", "f1_a"] == pytest.approx(0.9090909090909091)
    assert fams.loc["A", "f1_b"] == pytest.approx(0.8888888888888888)
    assert fams.loc["B", "f1_a"] == pytest.approx(0.8888888888888888)
    assert fams.loc["B", "f1_b"] == pytest.approx(1.0)
    assert int(fams.loc["A", "support"]) == 5
    assert int(fams.loc["B", "support"]) == 5


def test_macro_f1_by_support_threshold():
    a, b = _two_method_preds()
    out = macro_f1_by_support(a, b, class_list=["A", "B", "nontox"], support_threshold=4)
    assert {"group", "macro_f1_a", "macro_f1_b", "n_families"}.issubset(out.columns)
    g = out.set_index("group")
    # Both toxin families have support 5 > 4 -> all in the high bucket.
    hi = g.loc["support>4"]
    assert hi["macro_f1_a"] == pytest.approx(0.898989898989899)
    assert hi["macro_f1_b"] == pytest.approx(0.9444444444444444)
    assert int(hi["n_families"]) == 2
    assert int(hi["n_sequences"]) == 10
    lo = g.loc["support<=4"]
    assert int(lo["n_families"]) == 0
    assert int(lo["n_sequences"]) == 0
    assert np.isnan(lo["macro_f1_a"]) and np.isnan(lo["macro_f1_b"])


def test_macro_f1_conventions_nohit_penalised_le_restricted():
    _, b = _two_method_preds()
    conv = macro_f1_conventions(b, class_list=["A", "B", "nontox"])
    assert conv["macro_f1_nohit_wrong"] <= conv["macro_f1_restricted"] + 1e-9
    assert conv["n_no_hit"] == 1  # exactly one 'no hit' prediction in b
    assert conv["macro_f1_nohit_wrong"] == pytest.approx(0.9629629629629629)
    assert conv["macro_f1_restricted"] == pytest.approx(1.0)


def test_binary_reliability_perfect_calibration():
    # scores equal to true probability in two clean bins
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    out = binary_reliability(y, p, n_bins=2)
    assert out["ece"] == pytest.approx(0.0, abs=1e-9)


def test_binary_reliability_known_nonzero_ece():
    # 5 correct of 10, all predicted toxic at p=0.9 (conf 0.9), n_bins=2.
    # Confidence 0.9 falls in the upper bin (0.75-1.0): acc=0.5, conf=0.9 -> ECE=0.4.
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    p = np.full(10, 0.9)
    out = binary_reliability(y, p, n_bins=2)
    assert out["ece"] == pytest.approx(0.4)
    # Lower bin empty, upper bin holds all 10 samples.
    assert out["bin_proportion"][0] == pytest.approx(0.0)
    assert out["bin_proportion"][1] == pytest.approx(1.0)
    assert out["bin_accuracy"][1] == pytest.approx(0.5)
    assert out["bin_confidence"][1] == pytest.approx(0.9)


def test_adjudication_summary_counts(tmp_path):
    csv = tmp_path / "adj.csv"
    csv.write_text(
        "identifier,verdict,actual_label,predicted_label,assessment,assessment_category\n"
        "p1,tox,nontox,Phospholipase family,correct,family_correct\n"
        "p2,nontox,nontox,other,incorrect,false_positive_nonspecific\n"
        "p3,tox,nontox,Venom Kunitz-type family,partial,family_adjacent\n"
    )
    s = adjudication_summary(csv)
    assert s["n"] == 3
    assert s["assessment"]["correct"] == 1
    assert s["assessment"]["incorrect"] == 1
    assert s["n_annotation_gaps"] == 2  # nontox-labelled & verdict tox (p1,p3)


# ---------- MCC-based evaluation + bootstrap CIs ----------

from toxfam.evaluation.manuscript import (  # noqa: E402
    bootstrap_accuracy_ci,
    bootstrap_label_metric_ci,
    macro_mcc_by_support,
    micro_mcc,
    overall_mcc,
    per_family_mcc_difference,
)


def test_overall_mcc_perfect():
    y = ["A", "B", "A", "nontox", "B"]
    assert overall_mcc(y, y) == pytest.approx(1.0)


def test_overall_mcc_nohit_counts_wrong():
    yt = ["A", "B", "A", "B"]
    yp = ["A", "no hit", "A", "B"]  # one no-hit -> imperfect
    assert overall_mcc(yt, yp) < 1.0


def test_overall_mcc_one_error_pinned():
    yt = ["A", "B", "A", "B"]
    yp = ["A", "A", "A", "B"]  # single error at idx1
    assert overall_mcc(yt, yp) == pytest.approx(0.5773502691896258)


def test_micro_mcc_in_range():
    yt = ["A", "B", "A", "nontox", "B"]
    yp = ["A", "B", "B", "nontox", "B"]
    v = micro_mcc(yt, yp, class_list=["A", "B", "nontox"])
    assert -1.0 <= v <= 1.0


def test_micro_mcc_oov_no_hit_scores_wrong():
    yt = ["A", "B", "A", "B"]
    # One 'no hit' (out-of-vocab) prediction must score as wrong: < perfect.
    yp = ["A", "no hit", "A", "B"]
    assert micro_mcc(yt, yp, class_list=["A", "B"]) == pytest.approx(0.625)
    # Same labels with the OOV slot corrected -> perfect, confirming the OOV path is the only error.
    assert micro_mcc(yt, yt, class_list=["A", "B"]) == pytest.approx(1.0)


def test_bootstrap_accuracy_ci_brackets_point():
    correct = np.array([1] * 80 + [0] * 20)
    ci = bootstrap_accuracy_ci(correct, n_boot=1000, seed=1)
    assert ci["point"] == pytest.approx(0.8)
    assert ci["ci_low"] < 0.8 < ci["ci_high"]


def test_bootstrap_accuracy_ci_deterministic_and_pinned():
    correct = np.array([1] * 80 + [0] * 20)
    ci1 = bootstrap_accuracy_ci(correct, n_boot=500, seed=3)
    ci2 = bootstrap_accuracy_ci(correct, n_boot=500, seed=3)
    assert ci1 == ci2  # same seed -> identical dict
    assert ci1["point"] == pytest.approx(0.8)
    assert ci1["ci_low"] == pytest.approx(0.72, abs=1e-9)
    assert ci1["ci_high"] == pytest.approx(0.88, abs=1e-9)
    assert ci1["n"] == 100


def test_bootstrap_label_metric_ci_brackets_point():
    yt = np.array(["A", "B"] * 50)
    yp = yt.copy()
    yp[:10] = "B"  # introduce errors
    ci = bootstrap_label_metric_ci(yt, yp, overall_mcc, n_boot=300, seed=1)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_per_family_mcc_difference_columns():
    a, b = _two_method_preds()
    out = per_family_mcc_difference(a, b, class_list=["A", "B", "nontox"])
    assert {"family", "mcc_a", "mcc_b", "diff", "support"}.issubset(out.columns)
    assert (out["diff"] == (out["mcc_a"] - out["mcc_b"])).all()
    assert "nontox" not in set(out["family"])  # non-toxin excluded
    fams = out.set_index("family")
    assert fams.loc["A", "mcc_a"] == pytest.approx(0.8660254037844386)
    assert fams.loc["A", "mcc_b"] == pytest.approx(0.8528028654224417)
    assert fams.loc["B", "mcc_a"] == pytest.approx(0.8528028654224417)
    assert fams.loc["B", "mcc_b"] == pytest.approx(1.0)
    assert int(fams.loc["A", "support"]) == 5
    assert int(fams.loc["B", "support"]) == 5


def test_macro_mcc_by_support_columns():
    a, b = _two_method_preds()
    out = macro_mcc_by_support(a, b, class_list=["A", "B", "nontox"], support_threshold=4)
    assert {"group", "macro_mcc_a", "macro_mcc_b", "n_families"}.issubset(out.columns)
    g = out.set_index("group")
    hi = g.loc["support>4"]
    assert hi["macro_mcc_a"] == pytest.approx(0.8594141346034401)
    assert hi["macro_mcc_b"] == pytest.approx(0.9264014327112209)
    assert int(hi["n_families"]) == 2
    assert int(hi["n_sequences"]) == 10
    lo = g.loc["support<=4"]
    assert int(lo["n_families"]) == 0
    assert np.isnan(lo["macro_mcc_a"]) and np.isnan(lo["macro_mcc_b"])


def test_accuracy_by_identity_bins():
    # HBI frame: confidence == best-hit fractional identity.
    # x2 is a no-hit (excluded); x3 is non-toxin (excluded by toxin_only).
    hbi = pd.DataFrame(
        {
            "identifier": ["x0", "x1", "x2", "x3", "x4"],
            "actual_label": ["A", "A", "B", "nontox", "A"],
            "predicted_label": ["A", "B", "no hit", "nontox", "A"],
            "confidence": [0.90, 0.50, 0.00, 0.95, 0.85],
        }
    )
    other = pd.DataFrame(
        {
            "identifier": ["x0", "x1", "x2", "x3", "x4"],
            "actual_label": ["A", "A", "B", "nontox", "A"],
            "predicted_label": ["A", "A", "B", "nontox", "B"],  # other wrong only on x4
        }
    )
    out = accuracy_by_identity_bins(
        hbi, other, bins=[0, 0.4, 0.6, 0.8, 1.0001],
        labels=["<0.4", "0.4-0.6", "0.6-0.8", ">0.8"],
    )
    by = {r["bin_label"]: r for _, r in out.iterrows()}
    # Only occupied bins after toxin-only + no-hit exclusion: x1->0.4-0.6, x0&x4->>0.8.
    assert set(by) == {"0.4-0.6", ">0.8"}
    assert by["0.4-0.6"]["n"] == 1
    assert by["0.4-0.6"]["hbi_accuracy"] == pytest.approx(0.0)      # x1 hbi wrong
    assert by["0.4-0.6"]["other_accuracy"] == pytest.approx(1.0)    # x1 other right
    assert by["0.4-0.6"]["diff"] == pytest.approx(1.0)
    assert by[">0.8"]["n"] == 2
    assert by[">0.8"]["hbi_accuracy"] == pytest.approx(1.0)         # x0,x4 hbi right
    assert by[">0.8"]["other_accuracy"] == pytest.approx(0.5)       # x4 other wrong
    assert by[">0.8"]["diff"] == pytest.approx(-0.5)
    # no-hit (x2) and non-toxin (x3) excluded -> total n is 3, not 5.
    assert int(out["n"].sum()) == 3


def test_accuracy_by_identity_bins_requires_full_coverage():
    hbi = pd.DataFrame(
        {
            "identifier": ["x0"],
            "actual_label": ["A"],
            "predicted_label": ["A"],
            "confidence": [0.9],
        }
    )
    other = pd.DataFrame(
        {"identifier": ["zzz"], "actual_label": ["A"], "predicted_label": ["A"]}
    )
    with pytest.raises(ValueError):
        accuracy_by_identity_bins(hbi, other, bins=[0, 1.0001], labels=["all"])
