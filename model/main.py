import json
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from model.config import CONFIG
from model.dataset import ToxDataset, analyze_data_splits  # updated dataset module
from model_architecture import MLP
from training import train_model, evaluate_model, get_class_weights
from visualization import plot_loss_curve, plot_confusion_matrix
from utils import custom_logging


# -----------------------------------------------------------------------------
# Utility – discover all HDF5 embed files once and stash in CONFIG
# -----------------------------------------------------------------------------
if "h5_paths" not in CONFIG:
    # Fallback discovery: look in CONFIG['h5_dir'] or sibling of input CSV
    embeds_root = Path(CONFIG.get("h5_dir", Path(CONFIG["input_csv"]).parent))
    CONFIG["h5_paths"] = sorted(str(p) for p in embeds_root.glob("training_embeds_*.h5"))
    if not CONFIG["h5_paths"]:
        raise FileNotFoundError(
            f"No HDF5 embed files found in {embeds_root}. Expected names like 'training_embeds_*.h5'.")


# -----------------------------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------------------------

def analyze_label_distribution_for_split(train_df, val_df, test_df, label_col, output_dir):
    """Save counts + plot + chi‑square for *label_col* across the three splits."""

    train_counts = train_df[label_col].value_counts().sort_index()
    val_counts   = val_df[label_col].value_counts().sort_index()
    test_counts  = test_df[label_col].value_counts().sort_index()

    dist_df = pd.DataFrame({
        "Train":      train_counts,
        "Validation": val_counts,
        "Test":       test_counts,
    }).fillna(0)

    # JSON counts
    dist_json = Path(output_dir, f"{label_col.replace(' ', '_')}_distribution.json")
    dist_df.to_json(dist_json, orient="index")

    # Bar plot
    plt.figure(figsize=(10, 6))
    dist_df.plot(kind="bar")
    plt.title(f"Distribution of {label_col} Across Splits")
    plt.xlabel(label_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(Path(output_dir, f"{label_col.replace(' ', '_')}_distribution.png"))
    plt.close()

    # Chi‑square
    chi2, p, dof, expected = chi2_contingency(dist_df.T)
    chi_json = Path(output_dir, f"{label_col.replace(' ', '_')}_chi_square.json")
    with chi_json.open("w") as fp:
        json.dump({
            "chi2_statistic": chi2,
            "p_value": p,
            "degrees_of_freedom": dof,
            "expected": expected.tolist(),
        }, fp, indent=4)

    print(f"{label_col}: χ²={chi2:.2f}, p={p:.4e}, dof={dof}")


def train_and_return_model(train_df, val_df, label_col):
    # Build datasets
    train_ds = ToxDataset(train_df, CONFIG["h5_paths"], is_train=True,  label_col=label_col)
    val_ds   = ToxDataset(val_df,   CONFIG["h5_paths"], label_encoder=train_ds.le, is_train=False, label_col=label_col)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False)

    # Class weights & model
    _, w_tensor, _ = get_class_weights(train_ds)
    model = MLP(
        input_dim=CONFIG["embedding_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_family_classes=train_ds.num_classes,
    )

    # Point at the single root output dir
    run_cfg = CONFIG.copy()
    run_cfg["output_dir"] = CONFIG["output_dir"]

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.CrossEntropyLoss(weight=w_tensor.to(device))

    # Train + save loss curve
    model, history = train_model(model, train_loader, val_loader, w_tensor, train_ds.le, run_cfg)
    plot_loss_curve(history, Path(run_cfg["output_dir"]) / "loss_plot.png")

    # Cleanup
    train_ds.close()
    val_ds.close()

    return model, train_ds.le, loss_fn

def evaluate_label_on_dataset(model, dataset_df, label_col, label_encoder, loss_fn, tag, out_dir):
    # Note: out_dir should be CONFIG["output_dir"]
    ds     = ToxDataset(dataset_df, CONFIG["h5_paths"], label_encoder=label_encoder, is_train=False, label_col=label_col)
    loader = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get metrics + preds
    metrics, y_true, y_pred = evaluate_model(model, loader, loss_fn, device, dataset_type=tag)

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        ds.le,
        Path(out_dir) / f"{tag.lower()}_confusion_matrix.png"
    )

    # Classification report
    report_path = Path(out_dir) / f"{tag.lower()}_metrics.json"
    report = classification_report(
        y_true, y_pred,
        labels=range(ds.num_classes),
        target_names=ds.le.classes_,
        output_dict=True, zero_division=0
    )
    report_path.write_text(json.dumps({
        "numeric_metrics": metrics,
        "classification_report": report,
    }, indent=4))

    ds.close()

# -----------------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------------
def main():
    # ensure output dir exists
    out_root = Path(CONFIG["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    # load and split data
    df = pd.read_csv(CONFIG["input_csv"])
    train_df, val_df, test_df = analyze_data_splits(df)

    # overall split stats
    counts = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    total  = sum(counts.values())
    (out_root / "overall_split_stats.json").write_text(json.dumps({
        "absolute": {**counts, "total": total},
        "relative": {k: v / total for k, v in counts.items()},
    }, indent=4))

    # single‐label: Protein families
    label_col = "Protein families"
    subdir    = label_col.replace(" ", "_").lower()   # e.g. "protein_families"
    out_dir   = out_root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # split diagnostics
    analyze_label_distribution_for_split(train_df, val_df, test_df, label_col, out_dir)

    # train + eval under one logging context
    with custom_logging(out_dir):
        # 1) train once on Protein families
        model, le, loss_fn = train_and_return_model(train_df, val_df, label_col)

        # 2) evaluate & plot on validation
        evaluate_label_on_dataset(model, val_df,  label_col, le, loss_fn, "Validation", out_dir)

        # 3) evaluate & plot on test
        evaluate_label_on_dataset(model, test_df, label_col, le, loss_fn, "Test",       out_dir)


if __name__ == "__main__":
    main()
