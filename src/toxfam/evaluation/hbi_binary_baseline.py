"""Homology-Based Inference (HBI) binary baselines for toxic/non-toxic prediction.

Runs MMseqs2 sequence search: test sequences against training sequences.
Derives binary toxic/non-toxic predictions from the best hit's family label.

Baselines implemented:
1. Best-hit transfer: assign the best hit's binary label (toxic/nontox)
2. Best-hit + fident score: use fractional identity as p_toxic confidence
3. Best-hit + e-value score: use -log10(evalue) as confidence
4. Top-K voting: use top-K hits and majority vote for binary label
5. E-value thresholded: only transfer label if e-value < threshold
6. Sequence length: predict toxic if length < optimal threshold
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pymmseqs.commands import createdb, search
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from toxfam._paths import get_project_root
from toxfam.evaluation.metrics import (
    calculate_binary_metrics_with_scores,
    to_binary_class,
)


def _write_fasta(df: pd.DataFrame, path: Path) -> None:
    """Write FASTA from DataFrame — delegates to shared helper."""
    from toxfam.data._fasta import write_fasta

    write_fasta(df, path)


def _run_mmseqs2_search(
    query_fasta: Path,
    target_fasta: Path,
    work_dir: Path,
) -> pd.DataFrame:
    """Run MMseqs2 search and return results DataFrame.

    Uses sensible parameters: e-value cutoff 10, max 10 hits per query.
    """
    query_db = createdb(str(query_fasta), str(work_dir / "query_db"))
    target_db = createdb(str(target_fasta), str(work_dir / "target_db"))

    search_res = search(
        query_db.to_path(),
        target_db.to_path(),
        str(work_dir / "search_res"),
        str(work_dir / "tmp"),
        s=7,            # sensitivity (7 is default, fast enough)
        e=10,           # reasonable e-value cutoff
        min_seq_id=0.0,
        max_seqs=10,    # only need top hits, not all-vs-all
    )

    return search_res.to_pandas()


def _length_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, y_true: np.ndarray) -> dict:
    """Sequence length baseline: predict toxic if length < threshold.

    Finds optimal length threshold on training data, evaluates on test.
    """
    train_lengths = train_df["Sequence"].str.len()
    train_labels = train_df["is_toxic"].to_numpy()
    test_lengths = test_df["Sequence"].str.len().to_numpy()

    # Report length distributions
    tox_mask = train_labels == 1
    tox_len = train_lengths[tox_mask]
    nontox_len = train_lengths[~tox_mask]
    print(f"  Train toxic length:     median={tox_len.median():.0f}, mean={tox_len.mean():.0f}")
    print(f"  Train non-toxic length: median={nontox_len.median():.0f}, mean={nontox_len.mean():.0f}")

    # Find optimal length threshold on training data (maximize MCC)
    candidate_thresholds = np.percentile(train_lengths, np.arange(1, 100))
    candidate_thresholds = np.unique(candidate_thresholds.astype(int))

    best_mcc = -1
    best_thresh = 0
    for t in candidate_thresholds:
        preds = (train_lengths < t).astype(int)
        m = matthews_corrcoef(train_labels, preds)
        if m > best_mcc:
            best_mcc = m
            best_thresh = t

    print(f"  Optimal length threshold (train): {best_thresh} aa (train MCC: {best_mcc:.4f})")

    # Evaluate on test set
    y_pred_len = (test_lengths < best_thresh).astype(int)
    acc = accuracy_score(y_true, y_pred_len)
    f1 = f1_score(y_true, y_pred_len)
    mcc = matthews_corrcoef(y_true, y_pred_len)

    # Use normalized length as a continuous score for ROC/PR curves
    # Shorter = more likely toxic → p_toxic = 1 - length/max_length
    max_len = max(train_lengths.max(), test_lengths.max())
    p_toxic_length = 1.0 - (test_lengths / max_len)
    metrics_len = calculate_binary_metrics_with_scores(y_true, p_toxic_length)

    print(
        f"  Test (hard, thresh={best_thresh}): Acc={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}"
    )
    print(
        f"  Test (continuous):   ROC-AUC={metrics_len['roc_auc']:.4f}, "
        f"PR-AUC={metrics_len['pr_auc']:.4f}"
    )

    return {
        "length_threshold": {
            "threshold_aa": int(best_thresh),
            "accuracy": acc,
            "f1": f1,
            "mcc": mcc,
        },
        "length_continuous": {
            "roc_auc": metrics_len["roc_auc"],
            "pr_auc": metrics_len["pr_auc"],
            "f1": metrics_len["f1"],
            "mcc": metrics_len["mcc"],
            "accuracy": metrics_len["accuracy"],
        },
    }


def run_hbi_binary_baselines(
    input_csv: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run all HBI binary baselines and return results dict."""
    root = get_project_root()

    if input_csv is None:
        input_csv = root / "data" / "processed" / "training_data.csv"
    if output_dir is None:
        output_dir = root / "model" / "model_output" / "hbi_baselines"

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Load data and split
    print("Loading data...")
    df = pd.read_csv(input_csv)
    train_df = df[df["Split"] == "train"].copy()
    test_df = df[df["Split"] == "test"].copy()

    print(f"  Train: {len(train_df)} sequences")
    print(f"  Test:  {len(test_df)} sequences")

    # Binary ground truth
    test_df["is_toxic"] = test_df["Protein families"].apply(
        lambda x: 0 if to_binary_class(x) == "nontoxin" else 1
    )
    train_df["is_toxic"] = train_df["Protein families"].apply(
        lambda x: 0 if to_binary_class(x) == "nontoxin" else 1
    )

    y_true = test_df["is_toxic"].to_numpy()
    n_toxic = y_true.sum()
    n_nontox = len(y_true) - n_toxic
    print(f"  Test toxic: {n_toxic}, non-toxic: {n_nontox}")

    all_results = {}

    # =========================================================================
    # Baseline 0: Sequence length
    # =========================================================================
    print("\n--- Baseline 0: Sequence length ---")
    len_results = _length_baseline(train_df, test_df, y_true)
    all_results.update(len_results)

    # =========================================================================
    # MMseqs2 search
    # =========================================================================
    print("\nRunning MMseqs2 search (test vs train)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_fasta = tmpdir / "test.fasta"
        train_fasta = tmpdir / "train.fasta"

        _write_fasta(test_df, test_fasta)
        _write_fasta(train_df, train_fasta)

        res = _run_mmseqs2_search(test_fasta, train_fasta, tmpdir)

    if res.empty:
        print("ERROR: No MMseqs2 hits found!")
        return all_results

    print(f"  Total hits: {len(res)}")
    print(f"  Queries with hits: {res['query'].nunique()} / {len(test_df)}")

    # Add binary label for each target (training) sequence
    train_label_map = train_df.set_index("identifier")["is_toxic"].to_dict()
    res["target_is_toxic"] = res["target"].map(train_label_map)

    # =========================================================================
    # Baseline 1: Best-hit transfer (hard label)
    # =========================================================================
    print("\n--- Baseline 1: Best-hit transfer ---")
    best_hits = res.loc[res.groupby("query")["evalue"].idxmin()].reset_index(drop=True)

    bh_df = test_df[["identifier", "is_toxic"]].merge(
        best_hits[["query", "target_is_toxic", "fident", "evalue"]].rename(
            columns={"query": "identifier"}
        ),
        on="identifier",
        how="left",
    )

    n_no_hit = bh_df["target_is_toxic"].isna().sum()
    print(f"  No hits: {n_no_hit} ({100 * n_no_hit / len(bh_df):.1f}%)")
    bh_df["target_is_toxic"] = bh_df["target_is_toxic"].fillna(0)
    bh_df["fident"] = bh_df["fident"].fillna(0)
    bh_df["evalue"] = bh_df["evalue"].fillna(1e10)

    y_pred_hard = bh_df["target_is_toxic"].astype(int).to_numpy()

    acc = accuracy_score(y_true, y_pred_hard)
    f1 = f1_score(y_true, y_pred_hard)
    mcc = matthews_corrcoef(y_true, y_pred_hard)
    print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")

    all_results["best_hit_transfer"] = {
        "accuracy": acc, "f1": f1, "mcc": mcc, "no_hits": int(n_no_hit),
    }

    # =========================================================================
    # Baseline 2: Best-hit + fractional identity as score
    # =========================================================================
    print("\n--- Baseline 2: Best-hit fident score ---")
    p_toxic_fident = np.where(
        bh_df["target_is_toxic"] == 1,
        bh_df["fident"].to_numpy(),
        1.0 - bh_df["fident"].to_numpy(),
    )

    metrics_fident = calculate_binary_metrics_with_scores(y_true, p_toxic_fident)
    print(
        f"  ROC-AUC: {metrics_fident['roc_auc']:.4f}, "
        f"PR-AUC: {metrics_fident['pr_auc']:.4f}, "
        f"MCC: {metrics_fident['mcc']:.4f}"
    )
    all_results["best_hit_fident"] = {
        "roc_auc": metrics_fident["roc_auc"],
        "pr_auc": metrics_fident["pr_auc"],
        "f1": metrics_fident["f1"],
        "mcc": metrics_fident["mcc"],
        "accuracy": metrics_fident["accuracy"],
    }

    # =========================================================================
    # Baseline 3: Best-hit + e-value as score
    # =========================================================================
    print("\n--- Baseline 3: Best-hit e-value score ---")
    evalues = bh_df["evalue"].to_numpy().astype(float)
    neg_log_evalue = -np.log10(np.clip(evalues, 1e-300, None))
    max_score = neg_log_evalue.max()
    evalue_score = neg_log_evalue / max_score if max_score > 0 else np.zeros_like(neg_log_evalue)

    p_toxic_evalue = np.where(
        bh_df["target_is_toxic"] == 1,
        evalue_score,
        1.0 - evalue_score,
    )

    metrics_evalue = calculate_binary_metrics_with_scores(y_true, p_toxic_evalue)
    print(
        f"  ROC-AUC: {metrics_evalue['roc_auc']:.4f}, "
        f"PR-AUC: {metrics_evalue['pr_auc']:.4f}, "
        f"MCC: {metrics_evalue['mcc']:.4f}"
    )
    all_results["best_hit_evalue"] = {
        "roc_auc": metrics_evalue["roc_auc"],
        "pr_auc": metrics_evalue["pr_auc"],
        "f1": metrics_evalue["f1"],
        "mcc": metrics_evalue["mcc"],
        "accuracy": metrics_evalue["accuracy"],
    }

    # =========================================================================
    # Baseline 4: Top-K voting (K=5)
    # =========================================================================
    print("\n--- Baseline 4: Top-5 voting ---")
    sorted_res = res.sort_values(["query", "evalue"])
    top_k_hits = sorted_res.groupby("query").head(5)

    vote_df = (
        top_k_hits.groupby("query")
        .agg(
            n_hits=("target_is_toxic", "count"),
            n_toxic=("target_is_toxic", "sum"),
        )
        .reset_index()
    )
    vote_df["frac_toxic"] = vote_df["n_toxic"] / vote_df["n_hits"]

    vote_merged = test_df[["identifier", "is_toxic"]].merge(
        vote_df.rename(columns={"query": "identifier"}),
        on="identifier",
        how="left",
    )
    vote_merged["frac_toxic"] = vote_merged["frac_toxic"].fillna(0)

    p_toxic_vote = vote_merged["frac_toxic"].to_numpy()
    metrics_vote = calculate_binary_metrics_with_scores(y_true, p_toxic_vote)
    print(
        f"  ROC-AUC: {metrics_vote['roc_auc']:.4f}, "
        f"PR-AUC: {metrics_vote['pr_auc']:.4f}, "
        f"MCC: {metrics_vote['mcc']:.4f}"
    )
    all_results["top5_voting"] = {
        "roc_auc": metrics_vote["roc_auc"],
        "pr_auc": metrics_vote["pr_auc"],
        "f1": metrics_vote["f1"],
        "mcc": metrics_vote["mcc"],
        "accuracy": metrics_vote["accuracy"],
    }

    # =========================================================================
    # Baseline 5: E-value thresholded best-hit
    # =========================================================================
    print("\n--- Baseline 5: E-value thresholded ---")
    for thresh_name, evalue_thresh in [("1e-3", 1e-3), ("1e-10", 1e-10), ("1e-50", 1e-50)]:
        confident_mask = bh_df["evalue"] < evalue_thresh
        n_confident = confident_mask.sum()

        y_pred_thresh = np.zeros(len(y_true))
        y_pred_thresh[confident_mask] = bh_df.loc[confident_mask, "target_is_toxic"].to_numpy()

        acc_t = accuracy_score(y_true, y_pred_thresh.astype(int))
        f1_t = f1_score(y_true, y_pred_thresh.astype(int))
        mcc_t = matthews_corrcoef(y_true, y_pred_thresh.astype(int))

        print(
            f"  E<{thresh_name}: {n_confident}/{len(y_true)} confident "
            f"({100 * n_confident / len(y_true):.1f}%) | "
            f"Acc: {acc_t:.4f}, F1: {f1_t:.4f}, MCC: {mcc_t:.4f}"
        )
        all_results[f"evalue_thresh_{thresh_name}"] = {
            "evalue_threshold": float(evalue_thresh),
            "n_confident": int(n_confident),
            "accuracy": acc_t, "f1": f1_t, "mcc": mcc_t,
        }

    # =========================================================================
    # Save all results
    # =========================================================================
    (metrics_dir / "hbi_binary_baselines.json").write_text(
        json.dumps(all_results, indent=4)
    )

    # Summary table
    print("\n" + "=" * 80)
    print("HBI & SIMPLE BASELINES — SUMMARY")
    print("=" * 80)
    print(f"{'Method':<30} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8} {'MCC':>8} {'Acc':>8}")
    print("-" * 80)

    for name, m in all_results.items():
        roc = m.get("roc_auc", "—")
        pr = m.get("pr_auc", "—")
        f1_v = m.get("f1", "—")
        mcc_v = m.get("mcc", "—")
        acc_v = m.get("accuracy", "—")
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        print(f"{name:<30} {fmt(roc):>8} {fmt(pr):>8} {fmt(f1_v):>8} {fmt(mcc_v):>8} {fmt(acc_v):>8}")

    print("=" * 80)
    print(f"\nResults saved to: {metrics_dir / 'hbi_binary_baselines.json'}")

    return all_results
