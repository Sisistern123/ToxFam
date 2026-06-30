"""Manuscript-specific evaluation statistics (reusable, unit-tested).

Functions here compute the load-bearing numbers for the Results section:
subset/toxin-only accuracy, paired significance (McNemar + paired bootstrap),
accuracy-vs-length, per-family F1 differences, macro-F1 conventions, binary
calibration/reliability, and the confident-error adjudication summary.
"""
from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import label_binarize

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


def aligned_correctness(preds_a: pd.DataFrame, preds_b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Identifier-aligned correctness vectors for two methods (a, b).

    Paired statistics (McNemar, paired bootstrap) require row i of both vectors to refer to the
    same protein. Two independently-loaded prediction frames are not guaranteed to share row
    order or count (the NN path drops rows lacking an embedding). This aligns b onto a's
    identifier order, asserting both frames cover the same identifiers and agree on ground truth.
    """
    ids_a = preds_a["identifier"].to_numpy()
    ids_b = preds_b["identifier"].to_numpy()
    if len(ids_a) != len(ids_b) or not np.array_equal(ids_a, ids_b):
        b_indexed = preds_b.set_index("identifier")
        if len(b_indexed) != len(ids_a) or set(ids_a) != set(b_indexed.index):
            raise ValueError(
                f"prediction frames cover different identifiers "
                f"(|a|={len(ids_a)}, |b|={len(ids_b)}, "
                f"|a&b|={len(set(ids_a) & set(b_indexed.index))})"
            )
        preds_b = b_indexed.reindex(ids_a).reset_index()
    if not np.array_equal(preds_a["actual_label"].to_numpy(), preds_b["actual_label"].to_numpy()):
        raise ValueError("ground-truth actual_label disagrees between the two prediction frames")
    return correctness(preds_a), correctness(preds_b)


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray, *, exact: bool = False) -> dict:
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
    if exact:
        from scipy.stats import binomtest
        p = float(binomtest(min(b01, b10), n, 0.5).pvalue) if n > 0 else 1.0
    else:
        p = float(_chi2.sf(chi2, df=1)) if n > 0 else 1.0
    if 0 < n < 25 and not exact:
        warnings.warn(
            f"McNemar chi-square approximation is unreliable for n_discordant<25 (got {n}); "
            "consider exact=True", stacklevel=2)
    return {"b01": b01, "b10": b10, "n_discordant": n, "chi2": float(chi2), "p_value": p,
            "method": "exact_binomial" if exact else "chi2_continuity"}


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
    ids = preds["identifier"].to_numpy()
    missing = ~pd.Index(ids).isin(lengths.index)
    if missing.any():
        warnings.warn(f"{int(missing.sum())} identifier(s) absent from lengths; their length is NaN "
                      "and they may be dropped from length-binned stats", stacklevel=2)
    return lengths.reindex(ids).to_numpy(dtype=float)


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


def accuracy_by_identity_bins(
    hbi: pd.DataFrame,
    other: pd.DataFrame,
    *,
    bins: list[float],
    labels: list[str] | None = None,
    identity_col: str = "confidence",
    toxin_only: bool = True,
    n_boot: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Accuracy of HBI and a second method within HBI best-hit sequence-identity bins.

    The HBI ``confidence`` column is the fractional identity (``fident``) of the best
    MMseqs2 hit, so binning queries by it shows whether the learned model overtakes
    homology *within* an identity stratum. Queries with no hit (``NO_HIT_LABEL``,
    identity 0) carry no real identity and are excluded -- they are the separate no-hit
    coverage result. ``other`` is aligned to ``hbi`` by identifier and must cover every
    HBI identifier. Returns one row per occupied bin with columns
    ``bin_label, n, hbi_accuracy, other_accuracy, diff`` (``diff = other - hbi``).

    When ``n_boot`` is a positive int, two further columns ``diff_ci_low, diff_ci_high`` carry the
    per-bin paired-bootstrap 95% CI of ``diff`` (resampling proteins within the bin, with
    ``seed``), so callers get point estimates and CIs from a single source of truth rather
    than re-deriving the binning.
    """
    h = hbi[toxin_mask(hbi)] if toxin_only else hbi
    h = h[h["predicted_label"] != NO_HIT_LABEL].copy()
    other_correct = pd.Series(correctness(other), index=other["identifier"].to_numpy())
    if not pd.Index(h["identifier"]).isin(other_correct.index).all():
        raise ValueError("`other` must cover every HBI identifier to align correctness")
    ident = pd.to_numeric(h[identity_col], errors="coerce").to_numpy(dtype=float)
    if labels is None:
        labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    cat = pd.cut(ident, bins=bins, labels=labels, include_lowest=True, right=False)
    df = pd.DataFrame({
        "bin_label": cat,
        "hbi_correct": correctness(h).astype(float),
        "other_correct": h["identifier"].map(other_correct).to_numpy(dtype=float),
    })
    g = df.groupby("bin_label", observed=True)
    out = g.agg(
        n=("hbi_correct", "size"),
        hbi_accuracy=("hbi_correct", "mean"),
        other_accuracy=("other_correct", "mean"),
    ).reset_index()
    out["diff"] = out["other_accuracy"] - out["hbi_accuracy"]
    if n_boot:  # positive int attaches CIs; None/0 skips (0 would empty the bootstrap)
        # Iterate observed groups only (observed=True); key by label and re-attach by row
        # value -- not via Series.map, which on a categorical maps over *all* categories
        # (including empty bins absent from `ci`).
        ci = {
            label: paired_bootstrap_accuracy_diff(
                grp["other_correct"].to_numpy(), grp["hbi_correct"].to_numpy(),
                n_boot=n_boot, seed=seed,
            )
            for label, grp in g
        }
        out["diff_ci_low"] = [ci[lbl]["ci_low"] for lbl in out["bin_label"]]
        out["diff_ci_high"] = [ci[lbl]["ci_high"] for lbl in out["bin_label"]]
    return out


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


# ---------------------------------------------------------------------------
# MCC-based evaluation (primary metric) + bootstrap confidence intervals
# ---------------------------------------------------------------------------


def overall_mcc(y_true, y_pred) -> float:
    """Multiclass Matthews correlation coefficient over all samples.

    'no hit' is simply a label that never matches a true family, so it counts
    as wrong — the same convention used everywhere else in the evaluation.
    """
    return float(matthews_corrcoef(np.asarray(y_true), np.asarray(y_pred)))


def micro_mcc(y_true, y_pred, *, class_list: list[str]) -> float:
    """Micro-averaged MCC on the one-vs-all binarized label matrix.

    Mirrors ``toxfam.evaluation.metrics.calculate_metrics``: predictions outside
    ``class_list`` (e.g. 'no hit') map to an out-of-vocabulary index and count
    as wrong.
    """
    n = len(class_list)
    cls2idx = {c: i for i, c in enumerate(class_list)}
    oov = n
    yt = pd.Series(list(y_true)).map(lambda x: cls2idx.get(x, oov)).to_numpy(int)
    yp = pd.Series(list(y_pred)).map(lambda x: cls2idx.get(x, oov)).to_numpy(int)
    all_labels = list(range(n)) + [oov]
    yt_bin = label_binarize(yt, classes=all_labels)
    yp_bin = label_binarize(yp, classes=all_labels)
    return float(matthews_corrcoef(yt_bin.ravel(), yp_bin.ravel()))


def bootstrap_accuracy_ci(correct, *, n_boot: int = 2000, seed: int = 42) -> dict:
    """Percentile bootstrap 95% CI for a mean (accuracy) over a boolean vector."""
    c = np.asarray(correct, dtype=float)
    n = len(c)
    if n == 0:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "two_se": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = c[idx].mean(axis=1)
    return {"point": float(c.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)),
            "two_se": float(2.0 * means.std(ddof=1)), "n": n}


def bootstrap_label_metric_ci(y_true, y_pred, metric_fn, *, n_boot: int = 1000, seed: int = 42) -> dict:
    """Percentile bootstrap 95% CI for ``metric_fn(y_true, y_pred)`` (e.g. MCC).

    Resamples (y_true, y_pred) pairs with replacement — used for statistics that
    are not a per-sample mean.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    rng = np.random.default_rng(seed)
    point = float(metric_fn(yt, yp))
    vals = np.empty(n_boot, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            vals[i] = metric_fn(yt[idx], yp[idx])
    return {"point": point, "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
            "two_se": float(2.0 * vals.std(ddof=1)), "n": n}


def _ovr_mcc(is_true: np.ndarray, is_pred: np.ndarray) -> float:
    """One-vs-rest MCC; nan if the family has no true members."""
    if is_true.sum() == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(matthews_corrcoef(is_true, is_pred))


def per_family_mcc_difference(
    preds_a: pd.DataFrame, preds_b: pd.DataFrame, *, class_list: list[str]
) -> pd.DataFrame:
    """Per-family one-vs-rest MCC for method a minus method b, with true support.

    Non-toxin classes are excluded from the family view.
    """
    rows = []
    for fam in class_list:
        if fam.lower() in NONTOXIN_LABELS:
            continue
        ta = (preds_a["actual_label"].values == fam)
        pa = (preds_a["predicted_label"].values == fam)
        tb = (preds_b["actual_label"].values == fam)
        pb = (preds_b["predicted_label"].values == fam)
        rows.append({
            "family": fam,
            "mcc_a": _ovr_mcc(ta, pa),
            "mcc_b": _ovr_mcc(tb, pb),
            "support": int(ta.sum()),
        })
    out = pd.DataFrame(rows)
    out["diff"] = out["mcc_a"] - out["mcc_b"]
    return out.sort_values("diff").reset_index(drop=True)


def macro_mcc_by_support(
    preds_a: pd.DataFrame, preds_b: pd.DataFrame, *, class_list: list[str], support_threshold: int = 5
) -> pd.DataFrame:
    """Macro (mean per-family one-vs-rest) MCC of each method, split by support."""
    fam = per_family_mcc_difference(preds_a, preds_b, class_list=class_list)
    rows = []
    for label, sub in (
        (f"support>{support_threshold}", fam[fam["support"] > support_threshold]),
        (f"support<={support_threshold}", fam[fam["support"] <= support_threshold]),
    ):
        rows.append({
            "group": label,
            "macro_mcc_a": float(sub["mcc_a"].mean()) if len(sub) else float("nan"),
            "macro_mcc_b": float(sub["mcc_b"].mean()) if len(sub) else float("nan"),
            "n_families": int(len(sub)),
            "n_sequences": int(sub["support"].sum()),
        })
    return pd.DataFrame(rows)


def binary_reliability(
    y_true: np.ndarray, p_toxic: np.ndarray, *, n_bins: int = 15
) -> dict:
    """Reliability-diagram data + Expected Calibration Error for the binary head.

    Equal-width confidence bins on max(p, 1-p) over [0.5, 1]; accuracy = P(predicted class correct).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_toxic, dtype=float)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    centers, accs, confs, props = [], [], [], []
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        prop = in_bin.mean()
        centers.append((lo + hi) / 2)
        if prop > 0:
            acc_bin = correct[in_bin].mean()
            conf_bin = conf[in_bin].mean()
            ece += abs(conf_bin - acc_bin) * prop
            accs.append(acc_bin)
            confs.append(conf_bin)
            props.append(prop)
        else:
            accs.append(np.nan)
            confs.append(np.nan)
            props.append(0.0)
    return {"bin_center": centers, "bin_accuracy": accs, "bin_confidence": confs,
            "bin_proportion": props, "ece": float(ece)}


def adjudication_summary(csv_path: str | Path) -> dict:
    """Summarize Ivan's confident-error adjudication CSV for Figure 3 Panel B."""
    df = pd.read_csv(csv_path)
    required = {"identifier", "verdict", "actual_label", "assessment", "assessment_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"adjudication CSV missing required columns: {sorted(missing)}")
    gaps = df[(df["actual_label"].fillna("").str.lower().isin(NONTOXIN_LABELS))
              & (df["verdict"].fillna("").str.lower() == "tox")]
    return {
        "n": int(len(df)),
        "assessment": dict(Counter(df["assessment"].fillna("unknown").str.strip())),
        "assessment_category": dict(Counter(df["assessment_category"].fillna("unknown").str.strip())),
        "verdict": dict(Counter(df["verdict"].fillna("unknown").str.strip())),
        "n_annotation_gaps": int(len(gaps)),
        "annotation_gap_ids": gaps["identifier"].tolist(),
    }
