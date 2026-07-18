"""Shared benchmark harness for external-tool comparison.

Builds a single, internally consistent evaluation substrate that every method
(ToxFam emb+tax, ToxinPred 3.0, ToxDL 2.0) is scored against:

  benchmark/test_set/_shared/{test,val}.fasta        FASTA per split
  benchmark/test_set/_shared/{test,val}_labels.csv   identifier, seq_len, is_toxic, family
  benchmark/test_set/toxfam_embtax/{test,val}_scores.csv   identifier, score (= P(toxic))
  benchmark/test_set/toxfam_embtax/metrics.json            sanity binary metrics

ToxFam baseline = the combined (emb+tax) model = model/model_output/combined_run
(MultiInputMLP, tax_dim=50, 38 classes). Binary score = 1 - sum P(nontox classes),
exactly as toxfam.evaluation.binary.compute_p_toxic.

The split comes from the git-tracked manifest (``data/splits/split_manifest.csv``)
via ``registry.load_dataset``, never from ``training_data.csv``'s own ``Split``
column: that CSV is a release artifact a ``download-data --force`` can replace,
and a replacement carrying a different split would silently redefine what "test"
means for every tool scored here. The ToxFam baseline is refused unless its
checkpoint is stamped against that same manifest, so all methods are scored on
identical proteins + ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from toxfam.data.registry import load_dataset
from toxfam.data.split_manifest import manifest_sha256, verify_split_provenance
from toxfam.evaluation.metrics import (
    NONTOXIN_LABELS,
    calculate_binary_metrics_with_scores,
    find_optimal_threshold,
)
from toxfam.model.inference import load_calibrated_model
from toxfam.device import get_device

# ToxFam root = nearest ancestor with pyproject.toml (location-independent).
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
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
        tax_rows, missing_tax = [], []
        for i in idents:
            if i in tf:
                tax_rows.append(torch.tensor(tf[i][:], dtype=torch.float32))
            else:
                missing_tax.append(i)
                tax_rows.append(torch.zeros(tax_dim, dtype=torch.float32))
        tax = torch.stack(tax_rows)
    if missing_tax:
        # The trained pipeline (ToxDataset) raises on a missing tax vector; we zero-fill
        # to stay runnable, but surface it so a silent score divergence can't pass unnoticed.
        print(f"  [warn] {len(missing_tax)} proteins lack a taxonomy vector; zero-filled. "
              f"e.g. {missing_tax[:3]}")
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
        help="Only (re)build the shared FASTA + ground-truth labels from the split "
             "manifest, then exit. Does NOT need the trained model — use this "
             "to regenerate ground truth for compare.py on the committed scores.")
    args = ap.parse_args()

    # Split assignment comes from the manifest (load_dataset applies it), not from
    # training_data.csv's own Split column.
    test = load_dataset("test_set")
    val = load_dataset("val_set")

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

    # Scoring against the split: refuse a checkpoint not stamped to this manifest.
    # (Deliberately after --shared-only: the labels above come from the manifest and
    # are valid regardless of which checkpoint happens to be on disk.)
    verify_split_provenance(MODEL_DIR)

    print(f"\nLoading ToxFam (emb+tax) from {MODEL_DIR.name} ...")
    model, mcfg, idx_to_label = load_calibrated_model(MODEL_DIR)
    assert mcfg.architecture == "MultiInputMLP", mcfg.architecture
    tax_dim = mcfg.tax_dim

    scores = {}
    for name, sub in (("test", test), ("val", val)):
        p = toxfam_p_toxic(sub, model, idx_to_label, tax_dim)
        scores[name] = p
        pd.DataFrame({"identifier": sub.identifier.values, "score": p}).to_csv(
            TOXFAM / f"{name}_scores.csv", index=False)

    # Sanity: binary metrics, threshold tuned on val (Youden), as in the paper.
    yv = val["Protein families"].apply(is_toxic).values
    yt = test["Protein families"].apply(is_toxic).values
    sv = scores["val"]
    st = scores["test"]
    thr = find_optimal_threshold(yv, sv, method="youden")["optimal_threshold"]
    m_def = calculate_binary_metrics_with_scores(yt, st, threshold=0.5)
    m_opt = calculate_binary_metrics_with_scores(yt, st, threshold=thr)
    drop = {"fpr", "tpr", "precision_curve", "recall_curve", "roc_thresholds", "pr_thresholds"}
    out = {
        "method": "ToxFam (emb+tax)",
        "model_dir": str(MODEL_DIR.relative_to(ROOT)),
        "split_manifest_sha256": manifest_sha256(),
        "n_test": len(test),
        "optimized_threshold": thr,
        "test_default": {k: v for k, v in m_def.items() if k not in drop},
        "test_optimized": {k: v for k, v in m_opt.items() if k not in drop},
    }
    (TOXFAM / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nToxFam (emb+tax) on {len(test):,} test (t*={thr:.3f}):")
    print(f"  ROC-AUC={m_def['roc_auc']:.4f}  PR-AUC={m_def['pr_auc']:.4f}  "
          f"MCC(t*)={m_opt['mcc']:.4f}  F1(t*)={m_opt['f1']:.4f}  acc(t*)={m_opt['accuracy']:.4f}")
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
