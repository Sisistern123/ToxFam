import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import label_binarize
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_curve, auc

# Custom Modules
from config import CONFIG
from dataset import ToxDataset, analyze_data_splits
from model_architecture import ModularMLP, MultiInputMLP
from training import train_model, evaluate_model, get_class_weights
from visualization import plot_loss_curve, plot_confusion_matrix

# -----------------------------------------------------------------------------
# Setup & Config Checks
# -----------------------------------------------------------------------------
if "h5_paths" not in CONFIG:
    embeds_root = Path(CONFIG.get("h5_dir", Path(CONFIG["input_csv"]).parent))
    CONFIG["h5_paths"] = sorted(str(p) for p in embeds_root.glob("training_embeds_*.h5"))
    if not CONFIG["h5_paths"]:
        raise FileNotFoundError(f"No HDF5 embed files found in {embeds_root}.")


# -----------------------------------------------------------------------------
# Analysis Helpers (Stats & Plots)
# -----------------------------------------------------------------------------
def analyze_label_distribution_for_split(train_df, val_df, test_df, label_col, output_dir):
    """Save counts + plot + chi‑square for *label_col* across the three splits."""
    train_counts = train_df[label_col].value_counts().sort_index()
    val_counts = val_df[label_col].value_counts().sort_index()
    test_counts = test_df[label_col].value_counts().sort_index()

    dist_df = pd.DataFrame({
        "Train": train_counts,
        "Validation": val_counts,
        "Test": test_counts,
    }).fillna(0)

    # Save JSON
    dist_json = Path(output_dir, f"{label_col.replace(' ', '_')}_distribution.json")
    dist_df.to_json(dist_json, orient="index")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    dist_df.plot(kind="bar", logy=True, ax=ax)
    ax.tick_params(axis="x", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=50, ha="right")
    fig.subplots_adjust(bottom=0.4)
    ax.set_title(f"Distribution of {label_col} Across Splits (log scale)")
    ax.set_ylabel("Count (log scale)")
    fig.savefig(Path(output_dir, f"{label_col.replace(' ', '_')}_distribution_log.png"))
    plt.close(fig)

    # Chi-Square
    chi2, p, dof, expected = chi2_contingency(dist_df.T)
    with Path(output_dir, f"{label_col.replace(' ', '_')}_chi_square.json").open("w") as fp:
        json.dump({"chi2": chi2, "p_value": p, "dof": dof}, fp, indent=4)


def plot_multiclass_roc_from_scores(y_true, y_scores, classes, output_path, legend_cols=3):
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    n_classes = y_bin.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    cmap = plt.cm.get_cmap('rainbow', n_classes)
    colors = [(0.8 * r, 0.8 * g, 0.8 * b) for (r, g, b, a) in cmap(np.arange(n_classes))]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=180, constrained_layout=True)
    for i, cname in enumerate(classes):
        ax.plot(fpr[i], tpr[i], color=colors[i], lw=1.5, label=f"{cname} (AUC {roc_auc[i]:.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="gray")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=legend_cols, fontsize="small", frameon=False)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Input Helper
# -----------------------------------------------------------------------------
class DataSelector:
    """Wraps the loader to yield only the specific input needed by the strategy.
    Handles both Single-Input (Standard) and Multi-Input (Combined) Datasets automatically.
    """

    def __init__(self, loader, mode):
        self.loader = loader
        self.mode = mode  # 'emb_only', 'tax_only', 'both'

    def __iter__(self):
        for batch in self.loader:
            # PyTorch DataLoader yields [features, labels]
            features, label = batch

            # SCENARIO 1: Dataset returned single input (Standard Config / tax path is None)
            # features is just a Tensor
            if isinstance(features, torch.Tensor):
                if self.mode == 'emb_only':
                    yield features, label
                else:
                    # If we are here, you asked for 'tax' or 'both' but dataset only has embeddings
                    raise RuntimeError(
                        f"Strategy requested '{self.mode}' but Dataset provided only Embeddings. Check if tax_h5_path is set in config.")

            # SCENARIO 2: Dataset returned dual input (Combined/Pretrain Config)
            # features is a list/tuple: [emb_tensor, tax_tensor]
            elif isinstance(features, (list, tuple)):
                emb, tax = features
                if self.mode == 'emb_only':
                    yield emb, label
                elif self.mode == 'tax_only':
                    yield tax, label
                else:
                    # Returns tuple for MultiInputMLP
                    yield (emb, tax), label

    def __len__(self):
        return len(self.loader)


# -----------------------------------------------------------------------------
# STRATEGY 1: Standard
# -----------------------------------------------------------------------------
def run_standard_strategy(train_loader, val_loader, w_tensor, num_classes, out_dir):
    print(">>> Running Strategy: STANDARD (Embeddings Only)")
    model = ModularMLP(
        input_dim=CONFIG["embedding_dim"],
        hidden_dims=CONFIG["hidden_dims"],
        num_classes=num_classes,
        dropout=CONFIG["dropout"]
    )
    # Wrap loaders to only give embeddings
    model, hist = train_model(
        model,
        DataSelector(train_loader, 'emb_only'),
        DataSelector(val_loader, 'emb_only'),
        w_tensor, CONFIG
    )
    plot_loss_curve(hist, Path(out_dir) / "loss_standard.png")
    return model


# -----------------------------------------------------------------------------
# STRATEGY 2: Combined
# -----------------------------------------------------------------------------
def run_combined_strategy(train_loader, val_loader, w_tensor, num_classes, out_dir):
    print(">>> Running Strategy: COMBINED (Branched Architecture)")
    model = MultiInputMLP(
        embed_dim=CONFIG["embedding_dim"],
        tax_dim=CONFIG["tax_dim"],
        hidden_dims=CONFIG["hidden_dims"],
        num_classes=num_classes,
        dropout=CONFIG["dropout"]
    )
    # Wrap loaders to give (emb, tax) tuple
    model, hist = train_model(
        model,
        DataSelector(train_loader, 'both'),
        DataSelector(val_loader, 'both'),
        w_tensor, CONFIG
    )
    plot_loss_curve(hist, Path(out_dir) / "loss_combined.png")
    return model


# -----------------------------------------------------------------------------
# STRATEGY 3: Pretrain -> Finetune
# -----------------------------------------------------------------------------
def run_pretrain_finetune_strategy(train_loader, val_loader, w_tensor, num_classes, out_dir):
    print(">>> Running Strategy: PRETRAIN-FINETUNE (Optimized)")

    # --- 1. Initialize ONE Model ---
    # Start with Tax dimensions
    model = ModularMLP(
        input_dim=CONFIG["tax_dim"],
        hidden_dims=CONFIG["hidden_dims"],
        num_classes=num_classes,
        dropout=CONFIG["dropout"]
    )

    # --- 2. Stage 1: Pretraining on Taxonomy ---
    print("--- Stage 1: Pretraining on Taxonomy ---")
    s1_cfg = CONFIG.copy()
    s1_cfg['num_epochs'] = CONFIG['tax_epochs']
    s1_cfg['learning_rate'] = CONFIG['tax_lr']

    # Train the model (Tax mode)
    model, _ = train_model(
        model,
        DataSelector(train_loader, 'tax_only'),
        DataSelector(val_loader, 'tax_only'),
        w_tensor, s1_cfg
    )

    # --- 3. The Swap (In-Place) ---
    print("--- Swapping Input Layer (Tax -> Embeddings) ---")
    # This keeps the trained backbone weights!
    # No new model, no dictionary copying.
    model.swap_input_layer(new_input_dim=CONFIG["embedding_dim"])

    # Optional: Freeze backbone
    if CONFIG.get("freeze_backbone", False):
        print("Freezing backbone layers...")
        for param in model.backbone.parameters():
            param.requires_grad = False

    # --- 4. Stage 2: Finetuning on Embeddings ---
    print("--- Stage 2: Finetuning on Embeddings ---")

    # Notice we pass the SAME model instance
    model, hist = train_model(
        model,
        DataSelector(train_loader, 'emb_only'),
        DataSelector(val_loader, 'emb_only'),
        w_tensor, CONFIG
    )

    plot_loss_curve(hist, Path(out_dir) / "loss_finetuned.png")
    return model


# -----------------------------------------------------------------------------
# Unified Evaluation Function
# -----------------------------------------------------------------------------
def evaluate_label_on_dataset(model, dataset_df, label_col, label_encoder, loss_fn, tag, out_dir):
    """
    Evaluates the model on a dataframe.
    Crucially: Checks CONFIG['training_strategy'] to know how to feed data.
    """
    strategy = CONFIG["training_strategy"]

    # Always init dataset with Tax if available, DataSelector handles the rest
    ds = ToxDataset(
        dataset_df, CONFIG["h5_paths"],
        label_encoder=label_encoder, is_train=False,
        label_col=label_col, tax_h5_path=CONFIG["tax_h5_path"]
    )
    loader = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=False)

    # Wrap the loader based on strategy
    if strategy == "combined":
        selector = DataSelector(loader, "both")
    elif strategy == "pretrain_finetune":
        # The FINAL model in this strategy uses Embeddings only
        selector = DataSelector(loader, "emb_only")
    else:  # standard
        selector = DataSelector(loader, "emb_only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get metrics
    metrics, y_true, y_pred, y_scores = evaluate_model(model, selector, loss_fn, device, dataset_type=tag)

    # Save Results
    confidences = y_scores.max(axis=1)
    conf_df = pd.DataFrame({
        "identifier": dataset_df["identifier"].reset_index(drop=True),
        "actual_label": label_encoder.inverse_transform(y_true),
        "predicted_label": label_encoder.inverse_transform(y_pred),
        "confidence": confidences,
    })

    # Save CSVs
    conf_df.to_csv(Path(out_dir) / f"{tag.lower()}_predictions.csv", index=False)

    # Save Confusion Matrix & Reports
    plot_confusion_matrix(y_true, y_pred, ds.le, Path(out_dir) / f"{tag.lower()}_confusion_matrix.png")

    report = classification_report(y_true, y_pred, target_names=ds.le.classes_, output_dict=True, zero_division=0)
    (Path(out_dir) / f"{tag.lower()}_metrics.json").write_text(json.dumps({
        "numeric_metrics": metrics, "classification_report": report
    }, indent=4))

    # ROC
    plot_multiclass_roc_from_scores(y_true, y_scores, ds.le.classes_, Path(out_dir) / f"{tag.lower()}_roc.png")
    ds.close()


# -----------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------
def main():
    out_root = Path(CONFIG["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(CONFIG["input_csv"])
    train_df, val_df, test_df = analyze_data_splits(df)

    label_col = "Protein families"
    analyze_label_distribution_for_split(train_df, val_df, test_df, label_col, out_root)

    # 2. Init Datasets (Load both inputs; DataSelector filters them later)
    # We create the datasets here to get Class Weights and Encoders
    train_ds = ToxDataset(train_df, CONFIG["h5_paths"], is_train=True, tax_h5_path=CONFIG["tax_h5_path"])
    val_ds = ToxDataset(val_df, CONFIG["h5_paths"], label_encoder=train_ds.le, is_train=False,
                        tax_h5_path=CONFIG["tax_h5_path"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    _, w_tensor, _ = get_class_weights(train_ds)

    # 3. DISPATCH STRATEGY
    strategy = CONFIG["training_strategy"]
    final_model = None

    if strategy == "standard":
        final_model = run_standard_strategy(train_loader, val_loader, w_tensor, train_ds.num_classes, out_root)
    elif strategy == "combined":
        final_model = run_combined_strategy(train_loader, val_loader, w_tensor, train_ds.num_classes, out_root)
    elif strategy == "pretrain_finetune":
        final_model = run_pretrain_finetune_strategy(train_loader, val_loader, w_tensor, train_ds.num_classes,
                                                     out_root)
    else:
        raise ValueError(f"Unknown training strategy: {strategy}")

    # 4. Final Evaluation
    print("\nRunning Final Evaluation...")
    loss_fn = torch.nn.CrossEntropyLoss()
    evaluate_label_on_dataset(final_model, val_df, label_col, train_ds.le, loss_fn, "Validation", out_root)
    evaluate_label_on_dataset(final_model, test_df, label_col, train_ds.le, loss_fn, "Test", out_root)

    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()