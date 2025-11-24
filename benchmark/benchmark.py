#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark inference script (no training). Loads a trained model, runs inference on a dataset,
and writes predictions + metrics. Designed to integrate with your existing codebase:

  - `from model.dataset import ToxDataset`
  - `from model.model_architecture import MLP` (or your chosen model)

Defaults try to match your notebook:
  * MODEL_PATH: ../model/model_output/best_model.pt
  * LABEL_COL:  Protein families

Outputs (can be overridden via CLI):
  - predictions CSV (identifier, true_label, pred_label, confidence per class)
  - metrics JSON (overall accuracy, macro/micro F1, per-class precision/recall/F1, confusion matrix)

Example:
  python benchmark_clean.py \
    --csv ../benchmark/test_data.csv \
    --embeds ../benchmark/embeds.h5 \
    --model ../model/model_output/best_model.pt \
    --outdir ../benchmark/results \
    --batch-size 256 --device cuda
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader

from model.dataset import ToxDataset
from model.model_architecture import MLP

# ---------------------- Inference ----------------------
@torch.no_grad()
def run_inference(loader: DataLoader, model: torch.nn.Module, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for batch in loader:
        # ToxDataset should return (features, targets, ids) or similar; adapt if needed.
        if isinstance(batch, dict):
            features = batch["x"]
            targets = batch.get("y", None)
        else:
            # fallback: tuple
            if len(batch) == 3:
                features, targets, _ = batch
            elif len(batch) == 2:
                features, targets = batch
            else:
                features = batch
                targets = None

        features = features.to(device, non_blocking=True).float()
        outputs = model(features)  # shape: [B, C]
        probs = torch.softmax(outputs, dim=-1).cpu().numpy()
        all_probs.append(probs)

        if targets is not None:
            all_targets.append(torch.as_tensor(targets).cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0) if all_targets else None
    return probs, targets

def build_results_df(ids: List[str], class_names: List[str], probs: np.ndarray, y_true: np.ndarray | None) -> pd.DataFrame:
    top_idx = probs.argmax(axis=1)
    y_pred = [class_names[i] for i in top_idx]
    rows = {
        "identifier": ids,
        "pred_label": y_pred,
        "pred_index": top_idx,
        "confidence": probs.max(axis=1),
    }
    for j, cname in enumerate(class_names):
        rows[f"p_{cname}"] = probs[:, j]
    df = pd.DataFrame(rows)
    if y_true is not None:
        if isinstance(y_true[0], (np.integer, int)):
            true_labels = [class_names[i] for i in y_true]
        else:
            true_labels = list(y_true)
        df.insert(1, "true_label", true_labels)
    return df

def compute_metrics(y_true: List[str], y_pred: List[str], class_names: List[str]) -> Dict:
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=class_names, zero_division=0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro"))
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    per_class = {
        cname: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cname in enumerate(class_names)
    }
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": class_names,
    }

# ---------------------- Main ----------------------
def main():
    p = argparse.ArgumentParser(description="Benchmark inference (no training)")
    p.add_argument("--csv", required=True, help="CSV with at least columns: identifier, LABEL_COL")
    p.add_argument("--embeds", required=True, help="HDF5 with embeddings; keys indexed by identifier or stored as a dataset")
    p.add_argument("--label-col", default="Protein families", help="Name of the ground-truth label column in the CSV")
    p.add_argument("--model", default="../model/model_output/best_model.pt", help="Path to trained model .pt")
    p.add_argument("--outdir", default="../benchmark/results", help="Where to write predictions/metrics")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Dataset + loader
    ds = ToxDataset(csv_path=args.csv, embeds_path=args.embeds, label_col=args.label_col)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(args.device=="cuda"))

    # Model
    model = MLP(ds.input_dim, ds.num_classes)

    state = torch.load(args.model, map_location="cpu")
    # Be robust: handle state dict or whole model
    if isinstance(state, dict) and all(k.startswith(("module.", "", "fc")) for k in state.keys()):
        model.load_state_dict(state)
    elif hasattr(state, "state_dict"):
        model.load_state_dict(state.state_dict())
    else:
        # Last resort: try key under 'model'
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        else:
            raise RuntimeError("Could not load model weights from checkpoint")
    model.to(args.device, non_blocking=True)

    # Inference
    probs, y_true_idx = run_inference(loader, model, args.device)
    class_names = ds.class_names  # ensure ToxDataset exposes this (ordered list)
    ids = ds.ids                  # ensure ToxDataset exposes this (list aligned to dataset order)

    # Results table
    df_pred = build_results_df(ids, class_names, probs, y_true_idx)
    pred_csv = outdir / "predictions.csv"
    df_pred.to_csv(pred_csv, index=False)

    # Metrics (if labels available)
    if "true_label" in df_pred.columns:
        y_true = df_pred["true_label"].tolist()
        y_pred = df_pred["pred_label"].tolist()
        metrics = compute_metrics(y_true, y_pred, class_names)
        with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    else:
        metrics = None

    print(f"✅ Wrote {pred_csv}")
    if metrics:
        print(f"📊 Wrote {outdir/'metrics.json'}")

if __name__ == "__main__":
    main()
