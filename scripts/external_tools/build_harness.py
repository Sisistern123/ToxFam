"""Shared benchmark harness for external-tool comparison.

Builds a single, internally consistent evaluation substrate that every method
(ToxFam emb+tax, ToxinPred 3.0, ToxDL 2.0) is scored against:

  benchmark/test_set/_shared/{test,val}.fasta        FASTA per split
  benchmark/test_set/_shared/{test,val}_labels.csv   identifier, seq_len, is_toxic, family
  benchmark/test_set/toxfam_embtax/{test,val}_scores.csv   identifier, score (= P(toxic))
  benchmark/test_set/toxfam_embtax/metrics.json            sanity binary metrics

ToxFam baseline = the combined (emb+tax) model = model/model_output/combined_augmented
(MultiInputMLP, tax_dim=50, 38 classes). Binary score = 1 - sum P(nontox classes),
exactly as toxfam.evaluation.binary.compute_p_toxic.

NOTE: this uses the LOCAL data snapshot (test = 10,407 / 541 toxins), which differs
from the manuscript's 9,779 / 515 snapshot (newer, not checked out here). The point
is internal consistency: all methods scored on identical proteins + ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from toxfam.evaluation.metrics import (
    NONTOXIN_LABELS,
    calculate_binary_metrics_with_scores,
    find_optimal_threshold,
)
from toxfam.model.inference import load_calibrated_model
from toxfam.device import get_device

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / "data/processed/training_data.csv"
EMB_H5 = ROOT / "data/processed/embeddings.h5"
TAX_H5 = ROOT / "data/processed/taxonomy_vectors.h5"
MODEL_DIR = ROOT / "model/model_output/combined_run"

OUT = ROOT / "benchmark/test_set"
SHARED = OUT / "_shared"
TOXFAM = OUT / "toxfam_embtax"
for d in (SHARED, TOXFAM):
    d.mkdir(parents=True, exist_ok=True)


def is_toxic(label: str) -> int:
    return 0 if str(label).lower() in NONTOXIN_LABELS else 1


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w") as fh:
        for ident, seq in zip(df["identifier"], df["Sequence"]):
            fh.write(f">{ident}\n{seq}\n")


def write_split(df: pd.DataFrame, name: str) -> None:
    write_fasta(df, SHARED / f"{name}.fasta")
    out = pd.DataFrame(
        {
            "identifier": df["identifier"].values,
            "seq_len": df["Sequence"].str.len().values,
            "is_toxic": df["Protein families"].apply(is_toxic).values,
            "family": df["Protein families"].values,
        }
    )
    out.to_csv(SHARED / f"{name}_labels.csv", index=False)
    print(f"  {name}: {len(out)} seqs, {out.is_toxic.sum()} toxic "
          f"({100*out.is_toxic.mean():.2f}% pos)")


@torch.no_grad()
def toxfam_p_toxic(df: pd.DataFrame, model, idx_to_label, tax_dim: int) -> np.ndarray:
    """Replicate compute_p_toxic for the MultiInputMLP (emb+tax) model."""
    device = get_device()
    idents = df["identifier"].tolist()
    with h5py.File(EMB_H5, "r") as f:
        emb = torch.stack([torch.tensor(f[i][:], dtype=torch.float32) for i in idents])
    with h5py.File(TAX_H5, "r") as tf:
        tax = torch.stack([
            torch.tensor(tf[i][:], dtype=torch.float32) if i in tf
            else torch.zeros(tax_dim, dtype=torch.float32)
            for i in idents
        ])
    nontox_idx = [i for i, lbl in idx_to_label.items()
                  if str(lbl).lower() in NONTOXIN_LABELS]
    probs = []
    bs = 512
    for s in range(0, len(emb), bs):
        e = emb[s:s + bs].to(device)
        t = tax[s:s + bs].to(device)
        logits = model(e, t)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    p_nontox = probs[:, nontox_idx].sum(axis=1)
    return 1.0 - p_nontox


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the shared benchmark harness.")
    ap.add_argument(
        "--shared-only", action="store_true",
        help="Only (re)build the shared FASTA + ground-truth labels from "
             "training_data.csv, then exit. Does NOT need the trained model — use this "
             "to regenerate ground truth for compare.py on the committed scores.")
    args = ap.parse_args()

    df = pd.read_csv(TRAIN_CSV, usecols=["identifier", "Sequence", "Protein families", "Split"])
    test = df[df.Split == "test"].reset_index(drop=True)
    val = df[df.Split == "val"].reset_index(drop=True)

    print("Writing shared splits:")
    write_split(test, "test")
    write_split(val, "val")

    if args.shared_only:
        print(f"\n--shared-only: wrote {SHARED}; skipping ToxFam scoring.")
        return

    if not (MODEL_DIR / "models" / "best_model_calibrated.pt").exists():
        print(f"\n[warn] {MODEL_DIR}/models/best_model_calibrated.pt not found.")
        print("       Train it first:  uv run toxfam train configs/combined.yaml")
        print("       Shared splits were written; skipping ToxFam scoring.")
        return

    print(f"\nLoading ToxFam (emb+tax) from {MODEL_DIR.name} ...")
    model, mcfg, idx_to_label = load_calibrated_model(MODEL_DIR)
    assert mcfg.architecture == "MultiInputMLP", mcfg.architecture
    tax_dim = mcfg.tax_dim

    for name, sub in (("test", test), ("val", val)):
        p = toxfam_p_toxic(sub, model, idx_to_label, tax_dim)
        pd.DataFrame({"identifier": sub.identifier.values, "score": p}).to_csv(
            TOXFAM / f"{name}_scores.csv", index=False)

    # Sanity: binary metrics, threshold tuned on val (Youden), as in the paper.
    yv = val["Protein families"].apply(is_toxic).values
    yt = test["Protein families"].apply(is_toxic).values
    sv = pd.read_csv(TOXFAM / "val_scores.csv").score.values
    st = pd.read_csv(TOXFAM / "test_scores.csv").score.values
    thr = find_optimal_threshold(yv, sv, method="youden")["optimal_threshold"]
    m_def = calculate_binary_metrics_with_scores(yt, st, threshold=0.5)
    m_opt = calculate_binary_metrics_with_scores(yt, st, threshold=thr)
    drop = {"fpr", "tpr", "precision_curve", "recall_curve", "roc_thresholds", "pr_thresholds"}
    out = {
        "method": "ToxFam (emb+tax)",
        "model_dir": str(MODEL_DIR.relative_to(ROOT)),
        "snapshot": "local (test=10407)",
        "optimized_threshold": thr,
        "test_default": {k: v for k, v in m_def.items() if k not in drop},
        "test_optimized": {k: v for k, v in m_opt.items() if k not in drop},
    }
    (TOXFAM / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nToxFam (emb+tax) on LOCAL test (t*={thr:.3f}):")
    print(f"  ROC-AUC={m_def['roc_auc']:.4f}  PR-AUC={m_def['pr_auc']:.4f}  "
          f"MCC(t*)={m_opt['mcc']:.4f}  F1(t*)={m_opt['f1']:.4f}  acc(t*)={m_opt['accuracy']:.4f}")
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
