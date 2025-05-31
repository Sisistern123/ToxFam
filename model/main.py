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


# -----------------------------------------------------------------------------
# Per‑label training pipeline
# -----------------------------------------------------------------------------

def run_training_for_label(label_col: str, output_subdir: str, train_df, val_df, test_df):
    print(f"\n=== Training model for: {label_col} ===")
    out_dir = Path(CONFIG["output_dir"], output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Split stats & label distribution diagnostics
    # ------------------------------------------------------------------
    counts = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    total = sum(counts.values())
    (out_dir / "split_stats.json").write_text(json.dumps({
        "absolute": {**counts, "total": total},
        "relative": {k: v / total for k, v in counts.items()},
    }, indent=4))

    analyze_label_distribution_for_split(train_df, val_df, test_df, label_col, out_dir)

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    try:
        train_ds = ToxDataset(train_df, CONFIG["h5_paths"], is_train=True, label_col=label_col)
        val_ds   = ToxDataset(val_df,   CONFIG["h5_paths"], label_encoder=train_ds.le, is_train=False, label_col=label_col)
        test_ds  = ToxDataset(test_df,  CONFIG["h5_paths"], label_encoder=train_ds.le, is_train=False, label_col=label_col)
    except Exception as exc:
        print(f"Failed to build datasets for {label_col}: {exc}")
        return

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False)

    # ------------------------------------------------------------------
    # Class weights (handle imbalance)
    # ------------------------------------------------------------------
    w_dict, w_tensor, enc2lbl = get_class_weights(train_ds)
    print("\nClass counts & weights:")
    # merge the three Counters first, then iterate
    total_counts = (Counter(train_ds.df[f"{label_col}_encoded"])
                    + Counter(val_ds.df[f"{label_col}_encoded"])
                    + Counter(test_ds.df[f"{label_col}_encoded"]))

    for enc, cnt in total_counts.items():  # ← .items() on the result
        print(f"{enc2lbl[enc]}: {cnt} | weight={w_dict[enc2lbl[enc]]:.4f}")

    # ------------------------------------------------------------------
    # Model, train, evaluate
    # ------------------------------------------------------------------
    model = MLP(
        input_dim=CONFIG["embedding_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_family_classes=train_ds.num_classes,  # unified arg name
    )

    run_cfg = CONFIG.copy(); run_cfg["output_dir"] = str(out_dir)
    model, history = train_model(model, train_loader, val_loader, w_tensor, val_ds.le, run_cfg)
    plot_loss_curve(history, out_dir / "loss_plot.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.CrossEntropyLoss(weight=w_tensor.to(device))
    metrics, y_true, y_pred = evaluate_model(model, test_loader, loss_fn, device, dataset_type="Test")

    # Confusion matrix & classification report
    plot_confusion_matrix(y_true, y_pred, test_ds.le, out_dir / "test_confusion_matrix.png")
    (out_dir / "test_metrics.json").write_text(json.dumps({
        "numeric_metrics": metrics,
        "classification_report": classification_report(
            y_true, y_pred, labels=range(test_ds.num_classes),
            target_names=test_ds.le.classes_, output_dict=True, zero_division=0),
    }, indent=4))

    # Clean up to release file handles
    for ds in (train_ds, val_ds, test_ds):
        ds.close()


# -----------------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------------

def main():
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CONFIG["input_csv"])
    train_df, val_df, test_df = analyze_data_splits(df)  # uses the updated 'Split' field

    # Overall split stats
    counts = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    total = sum(counts.values())
    Path(CONFIG["output_dir"], "overall_split_stats.json").write_text(json.dumps({
        "absolute": {**counts, "total": total},
        "relative": {k: v / total for k, v in counts.items()},
    }, indent=4))

    # Targets to model
    for col, subdir in [
        ("Protein families", "protein_families")]:
        with custom_logging(Path(CONFIG["output_dir"], subdir)):
            run_training_for_label(col, subdir, train_df, val_df, test_df)


if __name__ == "__main__":
    main()
