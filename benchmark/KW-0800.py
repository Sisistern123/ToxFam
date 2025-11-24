#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refactored toxin classification inference pipeline
Created on: 2025-09-02 09:29:58

This script runs inference on toxin protein data using a pre-trained MLP model.
It processes protein embeddings from HDF5 files and generates classification predictions.
"""

import re
import logging
from types import SimpleNamespace
from typing import Any, List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import h5py
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score

from model.dataset import ToxDataset
from model.model_architecture import MLP

# Configuration constants
MODEL_PATH = "../model/model_output/best_model.pt"
TOX_H5_FILE = "../benchmark/all_tox.h5"
REFERENCE_CSV = "../data/interm/training_data.csv"
BENCHMARK_TSV = "../benchmark/all_tox.tsv"
LABEL_COL = "Protein families"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHOW_PLOTS = True

# Regular expression for normalizing UniProt accessions
ACC_RE = re.compile(r'^(?:[a-z]{2}\|)?([A-Z0-9]+)(?:\|.*)?$', re.IGNORECASE)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def normalize_accession(accession: str) -> str:
    """Normalize UniProt accession to bare format (e.g., 'A0A068B6Q6')."""
    s = str(accession).strip()
    m = ACC_RE.match(s)
    acc = m.group(1) if m else s.split()[0]
    acc = acc.split('-')[0].split('.')[0]  # Remove isoform/version
    return acc.upper()


def preprocess_benchmark_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess benchmark data with protein family mapping."""
    df = pd.read_csv(filepath, sep="\t")

    # Split protein families and take first entry
    df['Protein families'] = df['Protein families'].str.split(';').str[0]
    df['Protein families'] = df['Protein families'].str.split(',').str[0]

    # Fix conotoxin naming
    conotoxin_fixes = {
        'I1 superfamily': 'Conotoxin I1 superfamily',
        'O1 superfamily': 'Conotoxin O1 superfamily',
        'O2 superfamily': 'Conotoxin O2 superfamily',
        'E superfamily': 'Conotoxin E superfamily',
        'F superfamily': 'Conotoxin F superfamily'
    }
    df['Protein families'] = df['Protein families'].replace(conotoxin_fixes)

    # Apply family grouping patterns
    family_mapping = {
        r'Conotoxin.*': 'Conotoxin family',
        r'Neurotoxin.*': 'Neurotoxin family',
        r'Scoloptoxin.*|Scolopendra.*': 'Scoloptoxin family',
        r'Caterpillar.*': 'Caterpillar family',
        r'Teretoxin.*': 'Teretoxin family',
        r'Limacoditoxin.*': 'Limacoditoxin family',
        r'Scutigerotoxin.*': 'Scutigerotoxin family',
        r'Cationic peptide.*': 'Cationic peptide family',
        r'Formicidae venom.*': 'Formicidae venom family',
        r'Bradykinin-potentiating peptide family|Natriuretic peptide family|Natriuretic':
            'Natriuretic, Bradykinin potentiating peptide family',
        r'.*phospholipase.*|.*Phospholipase.*': 'Phospholipase family'
    }

    for pattern, replacement in family_mapping.items():
        df['Protein families'] = df['Protein families'].str.replace(
            pattern, replacement, regex=True
        )

    # Filter to valid families from training data
    train_df = pd.read_csv(REFERENCE_CSV)
    train_df = train_df[train_df["Protein families"] != "nontox"]
    valid_families = set(train_df["Protein families"].unique())

    df["Protein families"] = df["Protein families"].where(
        df["Protein families"].isin(valid_families),
        other="other"
    )

    logger.info(f"Benchmark data loaded: {len(df)} samples")
    logger.info(f"Family distribution:\n{df['Protein families'].value_counts()}")

    return df


@torch.no_grad()
def run_inference(
        loader: DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        log_interval: int = 10
) -> Tuple[List[int], List[List[float]]]:
    """Run model inference on data loader."""
    model.eval()
    all_preds, all_probs = [], []

    for i, (features, _) in enumerate(loader):
        features = features.to(device)
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1).cpu().tolist()
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        all_probs.extend(probs)
        all_preds.extend(preds)

        if (i + 1) % log_interval == 0:
            logger.info(f"Processed {(i + 1) * loader.batch_size} samples...")

    return all_preds, all_probs


def build_results_df(
        entry_ids: List[Any],
        all_preds: List[int],
        all_probs: List[List[float]],
        label_encoder
) -> pd.DataFrame:
    """Build results DataFrame from predictions and probabilities."""
    pred_names = [label_encoder.classes_[p] for p in all_preds]
    confidences = [max(p) for p in all_probs]

    df = pd.DataFrame({
        "identifier": entry_ids,
        "predicted_class": pred_names,
        "prediction_confidence": confidences
    })

    # Add probability columns for each class
    for i, cls in enumerate(label_encoder.classes_):
        df[f"prob_{cls}"] = [p[i] for p in all_probs]

    return df


def compute_statistics(
        all_preds: List[int],
        all_probs: List[List[float]],
        label_encoder
) -> Dict[str, Any]:
    """Compute prediction statistics."""
    pred_names = [label_encoder.classes_[p] for p in all_preds]
    counts = Counter(pred_names)
    total = len(all_preds)
    percentages = {cls: cnt / total * 100 for cls, cnt in counts.items()}
    confidences = [max(p) for p in all_probs]

    return {
        "total_samples": total,
        "class_distribution": {
            "counts": dict(counts),
            "percentages": percentages
        },
        "confidence_stats": {
            "mean": float(np.mean(confidences)),
            "median": float(np.median(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences))
        }
    }


def print_summary(stats: Dict[str, Any], name: str) -> None:
    """Print prediction summary statistics."""
    total = stats["total_samples"]
    cstats = stats["confidence_stats"]
    counts = stats["class_distribution"]["counts"]

    print(f"\n--- {name.upper()} SUMMARY ---")
    print(f"Total samples: {total:,}")
    print(f"Avg. confidence: {cstats['mean']:.3f}")
    print(f"Median confidence: {cstats['median']:.3f}\n")
    print("Top classes:")

    for cls, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        pct = cnt / total * 100
        print(f"  • {cls}: {cnt:,} ({pct:.1f}%)")


def plot_class_distribution(counts: Dict[str, int], title: str) -> None:
    """Plot class distribution with log scale."""
    labels, vals = list(counts.keys()), list(counts.values())

    plt.figure(figsize=(14, 8))
    bars = plt.bar(labels, vals, alpha=0.8)
    plt.yscale('log')
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Class")
    plt.ylabel("Count (log scale)")
    plt.xticks(rotation=45, ha="right")

    # Add count labels on bars
    for bar, v in zip(bars, vals):
        y = max(v, 1)
        plt.text(bar.get_x() + bar.get_width() / 2, y * 1.1, str(v),
                 ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


def plot_confidence_distribution(confidences: List[float], title: str) -> None:
    """Plot confidence score distribution."""
    plt.figure(figsize=(12, 6))
    plt.hist(confidences, bins=50, alpha=0.7, edgecolor='black')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')

    mean_conf = np.mean(confidences)
    median_conf = np.median(confidences)

    plt.axvline(mean_conf, linestyle='--', linewidth=2,
                label=f'Mean: {mean_conf:.3f}')
    plt.axvline(median_conf, linestyle='--', linewidth=2,
                label=f'Median: {median_conf:.3f}')

    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_pipeline(
        loader: DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        label_encoder,
        entry_id_col: str = "identifier",
        name: str = None,
        show_plots: bool = True
) -> Dict[str, Any]:
    """Run complete inference pipeline."""
    logger.info(f"Running inference pipeline for {name}")

    preds, probs = run_inference(loader, model, device)
    entry_ids = loader.dataset.df[entry_id_col].tolist()
    df = build_results_df(entry_ids, preds, probs, label_encoder)
    stats = compute_statistics(preds, probs, label_encoder)

    if name:
        print(f"\n=== Inference on {name} ===")
        print_summary(stats, name)

        if show_plots:
            plot_class_distribution(
                stats["class_distribution"]["counts"],
                f"{name} Class Distribution"
            )
            plot_confidence_distribution(
                [max(p) for p in probs],
                f"{name} Confidence Distribution"
            )

    return {"df": df, "stats": stats}


def evaluate_predictions(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> None:
    """Evaluate predictions against ground truth."""
    # Normalize accessions for matching
    pred_df_eval = pred_df.copy()
    pred_df_eval["acc"] = pred_df_eval["identifier"].map(normalize_accession)

    gt_df_eval = gt_df[["Entry", "Protein families"]].copy()
    gt_df_eval["acc"] = gt_df_eval["Entry"].map(normalize_accession)
    gt_df_eval = gt_df_eval.rename(columns={"Protein families": "actual"})

    # Merge on normalized accession
    eval_df = pred_df_eval.merge(gt_df_eval[["acc", "actual"]], on="acc", how="inner")

    logger.info(f"Matched {len(eval_df)} rows of {len(pred_df)} predictions")

    if len(eval_df) == 0:
        logger.warning("No matches found between predictions and ground truth")
        logger.info(f"Sample predictions: {pred_df['identifier'].head().tolist()}")
        logger.info(f"Sample ground truth: {gt_df['Entry'].head().tolist()}")
        return

    # Compute metrics
    y_true = eval_df["actual"].astype(str).to_numpy()
    y_pred = eval_df["predicted_class"].astype(str).to_numpy()
    labels = sorted(set(y_true) | set(y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels,
                                   zero_division=0, output_dict=True)

    print(f"\n=== Evaluation Results ===")
    print(f"Samples evaluated: {len(eval_df):,}")
    print(f"Accuracy: {accuracy:.4f}")

    # Format per-class results
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
    per_class = report_df[report_df["class"].isin(labels)][
        ["class", "precision", "recall", "f1-score", "support"]
    ].sort_values("support", ascending=False)

    print("\nPer-class Results (sorted by support):")
    print(per_class.to_string(index=False, max_rows=20))


def main():
    """Main execution function."""
    ns = SimpleNamespace()

    logger.info("Starting toxin classification pipeline...")

    # Load and preprocess benchmark data
    logger.info("Loading benchmark data...")
    ns.benchmark_df = preprocess_benchmark_data(BENCHMARK_TSV)

    # Setup label encoder from training data
    logger.info("Setting up label encoder...")
    ref_df = pd.read_csv(REFERENCE_CSV)
    ref_dataset = ToxDataset(ref_df, [TOX_H5_FILE], is_train=True, label_col=LABEL_COL)
    ns.label_encoder = ref_dataset.le
    ref_dataset.close()

    logger.info(f"Classes ({len(ns.label_encoder.classes_)}): {list(ns.label_encoder.classes_)}")

    # Discover HDF5 entries
    logger.info("Loading HDF5 data...")
    with h5py.File(TOX_H5_FILE, "r") as f:
        sample_ids = list(f.keys())
        num_samples = len(sample_ids)
        logger.info(f"Found {num_samples} samples in H5 file")

        if num_samples > 0:
            first_sample = f[sample_ids[0]]
            logger.info(f"First few samples: {sample_ids[:5]}")
            logger.info(f"Embedding shape for first sample: {first_sample.shape}")

    # Create dataset and dataloader
    dummy_df = pd.DataFrame({
        "identifier": sample_ids,
        LABEL_COL: np.repeat(ns.label_encoder.classes_[0], num_samples)  # placeholder
    })

    dataset = ToxDataset(
        dummy_df, [TOX_H5_FILE],
        label_encoder=ns.label_encoder,
        is_train=False,
        label_col=LABEL_COL
    )
    ns.loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    logger.info("Loading model...")
    ns.model = MLP(
        input_dim=1024,
        hidden_dim=128,
        num_family_classes=len(ns.label_encoder.classes_)
    )
    ns.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    ns.model.to(DEVICE)
    logger.info(f"Model loaded on {DEVICE}")

    # Run inference
    logger.info("Running inference...")
    results = run_pipeline(
        loader=ns.loader,
        model=ns.model,
        device=DEVICE,
        label_encoder=ns.label_encoder,
        name="unreviewed",
        show_plots=SHOW_PLOTS
    )

    ns.predictions_df = results["df"]
    ns.stats = results["stats"]

    # Evaluate against benchmark
    logger.info("Evaluating predictions...")
    evaluate_predictions(ns.predictions_df, ns.benchmark_df)

    logger.info("Pipeline completed successfully!")

    return ns


if __name__ == '__main__':
    results = main()