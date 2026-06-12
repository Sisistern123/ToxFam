"""Tests for toxfam.evaluation.manuscript — manuscript statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from toxfam.evaluation.manuscript import (
    accuracy_by_length_bins,
    adjudication_summary,
    binary_reliability,
    macro_f1_by_support,
    macro_f1_conventions,
    mcnemar_test,
    paired_bootstrap_accuracy_diff,
    per_family_f1_difference,
    rolling_accuracy_vs_length,
    subset_accuracy,
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


def test_mcnemar_counts_and_significance():
    correct_a = np.array([1, 1, 1, 0, 1])
    correct_b = np.array([1, 0, 0, 0, 1])  # b wrong where a right on idx1,2
    res = mcnemar_test(correct_a, correct_b)
    assert res["b01"] == 2  # a right, b wrong
    assert res["b10"] == 0  # a wrong, b right
    assert res["n_discordant"] == 2
    assert "chi2" in res and "p_value" in res


def test_paired_bootstrap_diff_sign_and_ci():
    rng_correct_a = np.array([1] * 90 + [0] * 10)
    rng_correct_b = np.array([1] * 80 + [0] * 20)
    res = paired_bootstrap_accuracy_diff(rng_correct_a, rng_correct_b, n_boot=2000, seed=42)
    assert res["diff"] == pytest.approx(0.10, abs=1e-9)
    assert res["ci_low"] < res["diff"] < res["ci_high"]


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


def test_macro_f1_by_support_threshold():
    a, b = _two_method_preds()
    out = macro_f1_by_support(a, b, class_list=["A", "B", "nontox"], support_threshold=4)
    assert {"group", "macro_f1_a", "macro_f1_b", "n_families"}.issubset(out.columns)


def test_macro_f1_conventions_nohit_penalised_le_restricted():
    _, b = _two_method_preds()
    conv = macro_f1_conventions(b, class_list=["A", "B", "nontox"])
    assert conv["macro_f1_nohit_wrong"] <= conv["macro_f1_restricted"] + 1e-9


def test_binary_reliability_perfect_calibration():
    # scores equal to true probability in two clean bins
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    out = binary_reliability(y, p, n_bins=2)
    assert out["ece"] == pytest.approx(0.0, abs=1e-9)


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


def test_micro_mcc_in_range():
    yt = ["A", "B", "A", "nontox", "B"]
    yp = ["A", "B", "B", "nontox", "B"]
    v = micro_mcc(yt, yp, class_list=["A", "B", "nontox"])
    assert -1.0 <= v <= 1.0


def test_bootstrap_accuracy_ci_brackets_point():
    correct = np.array([1] * 80 + [0] * 20)
    ci = bootstrap_accuracy_ci(correct, n_boot=1000, seed=1)
    assert ci["point"] == pytest.approx(0.8)
    assert ci["ci_low"] < 0.8 < ci["ci_high"]


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


def test_macro_mcc_by_support_columns():
    a, b = _two_method_preds()
    out = macro_mcc_by_support(a, b, class_list=["A", "B", "nontox"], support_threshold=4)
    assert {"group", "macro_mcc_a", "macro_mcc_b", "n_families"}.issubset(out.columns)
