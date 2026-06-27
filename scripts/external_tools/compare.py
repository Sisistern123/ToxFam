"""Unified binary-toxicity comparison across methods.

Consumes per-method per-protein toxicity scores and the shared ground truth,
and emits one comparison table + ROC/PR overlay, scored with identical code so
ToxFam (emb+tax), ToxinPred 3.0 and ToxDL 2.0 are directly comparable.

Reads (whichever exist):
  benchmark/test_set/_shared/test_labels.csv        identifier,seq_len,is_toxic,family
  benchmark/test_set/_shared/val_labels.csv
  benchmark/test_set/<method>/test_scores.csv       identifier,score  (higher = more toxic)
  benchmark/test_set/<method>/val_scores.csv        (optional; enables Youden threshold)

Writes:
  benchmark/test_set/comparison/metrics_full.csv    per method, per-method scored subset
  benchmark/test_set/comparison/metrics_common.csv  all methods on the common scored subset
  benchmark/test_set/comparison/paired_vs_toxfam.csv paired-bootstrap CIs vs ToxFam
  benchmark/test_set/comparison/roc_pr.png
  benchmark/test_set/comparison/summary.txt

Threshold policy per method: Youden-J on val if val_scores.csv exists, else 0.5.
Threshold-free metrics (ROC-AUC, PR-AUC) are the headline given the ~5% prior.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef, f1_score,
    precision_score, recall_score, balanced_accuracy_score, accuracy_score,
    roc_curve, precision_recall_curve,
)

# ToxFam root = nearest ancestor with pyproject.toml (location-independent).
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
BASE = ROOT / "benchmark/test_set"
SHARED = BASE / "_shared"
OUT = BASE / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

# method dir -> display name (order = plotting/report order)
METHODS = {
    "toxfam_embtax": "ToxFam (emb+tax)",
    "toxinpred3": "ToxinPred 3.0",
    "toxdl2": "ToxDL 2.0",
}
SEED = 42
N_BOOT = 2000


def youden_threshold(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:  # empty or single-class val merge -> roc_curve degenerate
        return 0.5
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def load_method(mdir: str) -> dict | None:
    """Return {name, test: df(identifier,score), val: df|None} or None if absent."""
    d = BASE / mdir
    tf = d / "test_scores.csv"
    if not tf.exists():
        return None
    test = pd.read_csv(tf)
    if "score" not in test.columns or "identifier" not in test.columns:
        print(f"  [skip] {mdir}: test_scores.csv missing identifier/score columns")
        return None
    test = test[["identifier", "score"]].drop_duplicates("identifier").copy()
    test["score"] = pd.to_numeric(test["score"], errors="coerce")
    vf = d / "val_scores.csv"
    val = None
    if vf.exists():
        val = pd.read_csv(vf)[["identifier", "score"]].drop_duplicates("identifier").copy()
        val["score"] = pd.to_numeric(val["score"], errors="coerce")
    return {"key": mdir, "name": METHODS[mdir], "test": test, "val": val}


def point_metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = (s >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
        "mcc": matthews_corrcoef(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "bal_acc": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "mcc_at_0.5": matthews_corrcoef(y, (s >= 0.5).astype(int)),
        "threshold": thr,
    }


def main() -> None:
    global BASE, SHARED, OUT
    ap = argparse.ArgumentParser(description="Unified binary-toxicity comparison.")
    ap.add_argument(
        "--scores-base", default=str(BASE),
        help="Dir holding <method>/test_scores.csv (+ val_scores.csv). Default: "
             "benchmark/test_set. For the committed snapshot pass "
             "scripts/external_tools/results/scores.")
    ap.add_argument(
        "--labels-dir", default=str(SHARED),
        help="Dir with test_labels.csv / val_labels.csv (regenerate with "
             "`build_harness.py --shared-only` after `toxfam download-data`).")
    ap.add_argument("--out", default=str(OUT), help="Output dir for comparison artifacts.")
    a = ap.parse_args()
    BASE, SHARED, OUT = Path(a.scores_base), Path(a.labels_dir), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)

    if not (SHARED / "test_labels.csv").exists():
        sys.exit(f"Missing {SHARED}/test_labels.csv — run "
                 "`build_harness.py --shared-only` first.")
    labels = pd.read_csv(SHARED / "test_labels.csv")[["identifier", "is_toxic"]].drop_duplicates("identifier")
    vlabels = None
    if (SHARED / "val_labels.csv").exists():
        vlabels = pd.read_csv(SHARED / "val_labels.csv")[["identifier", "is_toxic"]].drop_duplicates("identifier")

    loaded = [m for k in METHODS if (m := load_method(k))]
    if not loaded:
        sys.exit("No method scores found yet under benchmark/test_set/<method>/test_scores.csv")
    print("Methods present:", ", ".join(m["name"] for m in loaded))

    # Per-method threshold (Youden on val if available) + merged test frame.
    # Skip methods whose run is still incomplete (coverage below MIN_COVERAGE).
    MIN_COVERAGE = 0.90  # ToxDL 2.0 = 94.1% on the 9,779 set (578 proteins lack an AlphaFold structure)
    rows_full, merged, kept = [], {}, []
    for m in loaded:
        scored = m["test"].dropna(subset=["score"])
        t = scored.merge(labels, on="identifier", how="inner")
        n_unlabelled = len(scored) - len(t)
        if n_unlabelled:
            print(f"  [warn] {m['name']}: {n_unlabelled} scored proteins have no matching "
                  "label in --labels-dir (scores/labels snapshot mismatch?)")
        cov = len(t) / len(labels)
        if cov < MIN_COVERAGE:
            print(f"  [skip] {m['name']}: coverage {cov:.1%} < {int(MIN_COVERAGE*100)}% "
                  "— excluded (run still in progress / incomplete)")
            continue
        kept.append(m)
        merged[m["key"]] = t
        if m["val"] is not None and vlabels is not None:
            v = m["val"].merge(vlabels, on="identifier", how="inner").dropna(subset=["score"])
            thr = youden_threshold(v.is_toxic.values, v.score.values)
            thr_src = "youden@val"
        else:
            thr = 0.5
            thr_src = "default@0.5"
        mm = point_metrics(t.is_toxic.values, t.score.values, thr)
        mm.update({"method": m["name"], "n_scored": len(t),
                   "n_total": len(labels), "coverage": len(t) / len(labels),
                   "threshold_src": thr_src})
        rows_full.append(mm)
    loaded = kept
    if not loaded:
        sys.exit("All present methods fell below the coverage threshold "
                 f"({int(MIN_COVERAGE*100)}%) — nothing to compare.")

    full = pd.DataFrame(rows_full).set_index("method")
    cols = ["n_scored", "coverage", "roc_auc", "pr_auc", "mcc", "mcc_at_0.5", "f1",
            "precision", "recall", "bal_acc", "accuracy", "threshold", "threshold_src"]
    full[cols].to_csv(OUT / "metrics_full.csv")

    # Common scored subset (intersection of all methods).
    common_ids = set(labels.identifier)
    for m in loaded:
        common_ids &= set(merged[m["key"]].identifier)
    common_ids = sorted(common_ids)
    lab_c = labels.set_index("identifier").loc[common_ids, "is_toxic"]
    rows_common = []
    for m in loaded:
        t = merged[m["key"]].set_index("identifier").loc[common_ids]
        thr = float(full.loc[m["name"], "threshold"])
        mm = point_metrics(lab_c.values, t.score.values, thr)
        mm["method"] = m["name"]
        rows_common.append(mm)
    common = pd.DataFrame(rows_common).set_index("method")
    common.to_csv(OUT / "metrics_common.csv")

    # Paired bootstrap vs ToxFam on common subset (ROC-AUC and PR-AUC diffs).
    paired_rows = []
    if "toxfam_embtax" in merged and len(loaded) > 1 and len(common_ids) > 10:
        yc = lab_c.values
        scores_c = {m["key"]: merged[m["key"]].set_index("identifier").loc[common_ids].score.values
                    for m in loaded}
        rng = np.random.default_rng(SEED)
        n = len(common_ids)
        base = scores_c["toxfam_embtax"]
        for m in loaded:
            if m["key"] == "toxfam_embtax":
                continue
            other = scores_c[m["key"]]
            d_roc, d_pr = [], []
            for _ in range(N_BOOT):
                idx = rng.integers(0, n, n)
                yb = yc[idx]
                if yb.sum() == 0 or yb.sum() == len(yb):
                    continue
                d_roc.append(roc_auc_score(yb, base[idx]) - roc_auc_score(yb, other[idx]))
                d_pr.append(average_precision_score(yb, base[idx]) - average_precision_score(yb, other[idx]))
            if not d_roc:  # every resample was single-class -> no usable diffs
                print(f"  [skip] paired bootstrap vs {m['name']}: no two-class resamples")
                continue
            paired_rows.append({
                "comparison": f"ToxFam - {m['name']}",
                "d_roc_auc": float(np.mean(d_roc)),
                "d_roc_lo": float(np.percentile(d_roc, 2.5)),
                "d_roc_hi": float(np.percentile(d_roc, 97.5)),
                "d_pr_auc": float(np.mean(d_pr)),
                "d_pr_lo": float(np.percentile(d_pr, 2.5)),
                "d_pr_hi": float(np.percentile(d_pr, 97.5)),
                "n_common": n,
            })
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired.to_csv(OUT / "paired_vs_toxfam.csv", index=False)

    # ROC + PR overlay on the common subset.
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(13, 5.2))
    for m in loaded:
        t = merged[m["key"]].set_index("identifier").loc[common_ids]
        y, s = lab_c.values, t.score.values
        fpr, tpr, _ = roc_curve(y, s)
        axr.plot(fpr, tpr, label=f"{m['name']} (AUC={roc_auc_score(y, s):.3f})")
        prec, rec, _ = precision_recall_curve(y, s)
        axp.plot(rec, prec, label=f"{m['name']} (AP={average_precision_score(y, s):.3f})")
    axr.plot([0, 1], [0, 1], "k--", lw=0.8)
    axr.set(xlabel="False positive rate", ylabel="True positive rate",
            title=f"ROC — common subset (n={len(common_ids)})")
    axr.legend(fontsize=8, loc="lower right")
    axp.axhline(lab_c.mean(), ls="--", c="k", lw=0.8, label=f"prior={lab_c.mean():.3f}")
    axp.set(xlabel="Recall", ylabel="Precision",
            title=f"Precision-Recall — common subset (n={len(common_ids)})")
    axp.legend(fontsize=8, loc="lower left")
    fig.suptitle("Binary toxicity: ToxFam vs external tools (9,779 canonical test set)")
    fig.tight_layout()
    fig.savefig(OUT / "roc_pr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Console + summary.txt
    lines = []
    lines.append("=" * 78)
    lines.append("BINARY TOXICITY COMPARISON (9,779 canonical test set)")
    lines.append("=" * 78)
    lines.append(f"Ground truth: {len(labels)} test proteins, {int(labels.is_toxic.sum())} toxic "
                 f"({100*labels.is_toxic.mean():.2f}% positive prior)")
    lines.append("")
    lines.append("Per-method (own scored subset):")
    lines.append(full[["n_scored", "coverage", "roc_auc", "pr_auc", "mcc", "mcc_at_0.5",
                       "f1", "precision", "recall", "threshold", "threshold_src"]].round(4).to_string())
    lines.append("")
    lines.append(f"Common scored subset: n={len(common_ids)} "
                 f"({int(lab_c.sum())} toxic, {100*lab_c.mean():.2f}% prior)")
    lines.append(common[["roc_auc", "pr_auc", "mcc", "mcc_at_0.5", "f1", "precision", "recall"]].round(4).to_string())
    if not paired.empty:
        lines.append("")
        lines.append("Paired bootstrap vs ToxFam (positive = ToxFam better; CI excludes 0 => significant):")
        lines.append(paired.round(4).to_string(index=False))
    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt)
    print("\n" + txt)
    print(f"\nWrote comparison artifacts to {OUT}")


if __name__ == "__main__":
    main()
