"""Manuscript-specific evaluation statistics (reusable, unit-tested).

Functions here compute the load-bearing numbers for the Results section:
subset/toxin-only accuracy, paired significance (McNemar + paired bootstrap),
accuracy-vs-length, per-family F1 differences, macro-F1 conventions, binary
calibration/reliability, and the confident-error adjudication summary.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2

from toxfam.evaluation.hbi import NO_HIT_LABEL
from toxfam.evaluation.metrics import NONTOXIN_LABELS, calculate_metrics


def correctness(preds: pd.DataFrame) -> np.ndarray:
    """Boolean array: predicted_label == actual_label."""
    return (preds["predicted_label"].values == preds["actual_label"].values)


def subset_accuracy(preds: pd.DataFrame, mask: np.ndarray | pd.Series | None = None) -> float:
    """Accuracy over all rows, or over rows where ``mask`` is True."""
    correct = correctness(preds)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        correct = correct[mask]
    return float(correct.mean()) if len(correct) else float("nan")


def toxin_mask(preds: pd.DataFrame, label_col: str = "actual_label") -> np.ndarray:
    """True where the ground-truth label is a toxin (not a non-toxin class)."""
    return ~preds[label_col].str.lower().isin(NONTOXIN_LABELS).values


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired McNemar test on two boolean correctness vectors (a vs b).

    b01 = a-correct & b-wrong; b10 = a-wrong & b-correct. Uses the
    continuity-corrected chi-square with 1 dof.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    b01 = int(np.sum(a & ~b))
    b10 = int(np.sum(~a & b))
    n = b01 + b10
    chi2 = ((abs(b01 - b10) - 1) ** 2) / n if n > 0 else 0.0
    p = float(_chi2.sf(chi2, df=1)) if n > 0 else 1.0
    return {"b01": b01, "b10": b10, "n_discordant": n, "chi2": float(chi2), "p_value": p}


def paired_bootstrap_accuracy_diff(
    correct_a: np.ndarray, correct_b: np.ndarray, *, n_boot: int = 10000, seed: int = 42
) -> dict:
    """Paired bootstrap of accuracy(a) - accuracy(b) over the same samples.

    Returns the point difference and a 95% percentile CI.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return {
        "diff": float(a.mean() - b.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n": n,
    }


def _lengths_for(preds: pd.DataFrame, lengths: pd.Series) -> np.ndarray:
    return lengths.reindex(preds["identifier"].values).to_numpy(dtype=float)


def accuracy_by_length_bins(
    preds: pd.DataFrame, lengths: pd.Series, *, bins: list[int]
) -> pd.DataFrame:
    """Accuracy within fixed length bins. ``lengths`` indexed by identifier."""
    ln = _lengths_for(preds, lengths)
    correct = correctness(preds).astype(float)
    labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    cat = pd.cut(ln, bins=bins, labels=labels, include_lowest=True, right=False)
    df = pd.DataFrame({"bin_label": cat, "correct": correct})
    g = df.groupby("bin_label", observed=True)["correct"]
    out = g.agg(accuracy="mean", n="size").reset_index()
    out["se"] = np.sqrt(out["accuracy"] * (1 - out["accuracy"]) / out["n"])
    return out


def rolling_accuracy_vs_length(
    preds: pd.DataFrame, lengths: pd.Series, *, window: int = 50
) -> pd.DataFrame:
    """Length-sorted rolling-window mean accuracy (centered)."""
    ln = _lengths_for(preds, lengths)
    correct = correctness(preds).astype(float)
    order = np.argsort(ln, kind="stable")
    s = pd.Series(correct[order])
    roll = s.rolling(window=window, center=True, min_periods=max(1, window // 2)).mean()
    return pd.DataFrame({"length": ln[order], "accuracy": roll.to_numpy()})


def _per_class_f1(preds: pd.DataFrame, class_list: list[str]) -> dict[str, dict]:
    m = calculate_metrics(preds["actual_label"], preds["predicted_label"], class_list=class_list)
    return m.classification_report


def per_family_f1_difference(
    preds_a: pd.DataFrame, preds_b: pd.DataFrame, *, class_list: list[str]
) -> pd.DataFrame:
    """Per-family F1 for method a minus method b, with true support.

    Non-toxin classes are excluded from the family view.
    """
    rep_a = _per_class_f1(preds_a, class_list)
    rep_b = _per_class_f1(preds_b, class_list)
    rows = []
    for fam in class_list:
        if fam.lower() in NONTOXIN_LABELS:
            continue
        fa = rep_a.get(fam, {})
        fb = rep_b.get(fam, {})
        rows.append(
            {
                "family": fam,
                "f1_a": float(fa.get("f1-score", 0.0)),
                "f1_b": float(fb.get("f1-score", 0.0)),
                "support": int(fa.get("support", 0)),
            }
        )
    out = pd.DataFrame(rows)
    out["diff"] = out["f1_a"] - out["f1_b"]
    return out.sort_values("diff").reset_index(drop=True)


def macro_f1_by_support(
    preds_a: pd.DataFrame, preds_b: pd.DataFrame, *, class_list: list[str], support_threshold: int = 5
) -> pd.DataFrame:
    """Macro-F1 of each method split by family support (> vs <= threshold)."""
    fam = per_family_f1_difference(preds_a, preds_b, class_list=class_list)
    rows = []
    for label, sub in (
        (f"support>{support_threshold}", fam[fam["support"] > support_threshold]),
        (f"support<={support_threshold}", fam[fam["support"] <= support_threshold]),
    ):
        rows.append(
            {
                "group": label,
                "macro_f1_a": float(sub["f1_a"].mean()) if len(sub) else float("nan"),
                "macro_f1_b": float(sub["f1_b"].mean()) if len(sub) else float("nan"),
                "n_families": int(len(sub)),
                "n_sequences": int(sub["support"].sum()),
            }
        )
    return pd.DataFrame(rows)


def macro_f1_conventions(preds: pd.DataFrame, *, class_list: list[str]) -> dict:
    """Macro-F1 under two no-hit conventions for a single method.

    - nohit_wrong: no-hit predictions kept (map to OOV → lower true-class recall).
    - restricted: drop rows whose prediction is 'no hit' before scoring.
    """
    m_all = calculate_metrics(preds["actual_label"], preds["predicted_label"], class_list=class_list)
    keep = preds["predicted_label"] != NO_HIT_LABEL
    sub = preds[keep]
    m_res = calculate_metrics(sub["actual_label"], sub["predicted_label"], class_list=class_list)
    return {
        "macro_f1_nohit_wrong": float(m_all.classification_report["macro avg"]["f1-score"]),
        "macro_f1_restricted": float(m_res.classification_report["macro avg"]["f1-score"]),
        "n_no_hit": int((~keep).sum()),
    }


def binary_reliability(
    y_true: np.ndarray, p_toxic: np.ndarray, *, n_bins: int = 15
) -> dict:
    """Reliability-diagram data + Expected Calibration Error for the binary head.

    Equal-width confidence bins on max(p, 1-p); accuracy = P(predicted class correct).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_toxic, dtype=float)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs, props = [], [], [], []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        prop = in_bin.mean()
        centers.append((lo + hi) / 2)
        if prop > 0:
            acc_bin = correct[in_bin].mean()
            conf_bin = conf[in_bin].mean()
            ece += abs(conf_bin - acc_bin) * prop
            accs.append(acc_bin); confs.append(conf_bin); props.append(prop)
        else:
            accs.append(np.nan); confs.append(np.nan); props.append(0.0)
    return {"bin_center": centers, "bin_accuracy": accs, "bin_confidence": confs,
            "bin_proportion": props, "ece": float(ece)}


def adjudication_summary(csv_path: str | Path) -> dict:
    """Summarize Ivan's confident-error adjudication CSV for Figure 3 Panel B."""
    df = pd.read_csv(csv_path)
    gaps = df[(df["actual_label"].str.lower().isin(NONTOXIN_LABELS)) & (df["verdict"].str.lower() == "tox")]
    return {
        "n": int(len(df)),
        "assessment": dict(Counter(df["assessment"].str.strip())),
        "assessment_category": dict(Counter(df["assessment_category"].str.strip())),
        "verdict": dict(Counter(df["verdict"].str.strip())),
        "n_annotation_gaps": int(len(gaps)),
        "annotation_gap_ids": gaps["identifier"].tolist(),
    }
