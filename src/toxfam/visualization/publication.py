"""Publication-quality figures for ToxFam method comparison.

All figures use matplotlib only (no seaborn dependency for these).
Designed for inclusion in a research paper or thesis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Consistent color palette for methods
METHOD_COLORS = {
    "ToxinPred2": "#d62728",
    "ToxinPred3": "#17becf",
    "TOXIFY (reimpl.)": "#bcbd22",
    "Length baseline": "#7f7f7f",
    "NN binary": "#1f77b4",
    "HBI best-hit": "#ff7f0e",
    "NN augmented": "#2ca02c",
    "NN augmented+CP": "#9467bd",
    "NN binary+CPP": "#e377c2",
    "Confidence routing": "#8c564b",
    "Ensemble": "#e377c2",
}


def plot_overall_metrics_bar(
    metrics: dict[str, dict],
    output_path: Path,
) -> None:
    """Figure 1: Grouped bar chart of overall binary metrics per method.

    Parameters
    ----------
    metrics : {method_name: {roc_auc, pr_auc, f1, mcc}} for each method.
    """
    metric_names = ["ROC-AUC", "PR-AUC", "F1", "MCC"]
    metric_keys = ["roc_auc", "pr_auc", "f1", "mcc"]

    methods = list(metrics.keys())
    n_methods = len(methods)
    n_metrics = len(metric_names)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        vals = [metrics[method].get(k, 0) or 0 for k in metric_keys]
        color = METHOD_COLORS.get(method, f"C{i}")
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=color, edgecolor="white", linewidth=0.5)
        # Add value labels on top
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Binary Classification Metrics — Method Comparison")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_family_mcc(
    per_family_df: pd.DataFrame,
    method_columns: list[str],
    output_path: Path,
    *,
    top_n: int = 20,
) -> None:
    """Figure 2: Horizontal grouped bar chart of MCC per family per method."""
    # Filter to families with toxic members and take top N
    df = per_family_df[per_family_df["n_toxic"] > 0].head(top_n).copy()
    df = df.sort_values("n_total", ascending=True)

    families = df["family"].tolist()
    n_fam = len(families)
    n_methods = len(method_columns)

    fig, ax = plt.subplots(figsize=(10, max(6, n_fam * 0.4)))
    y = np.arange(n_fam)
    height = 0.8 / n_methods

    for i, method_col in enumerate(method_columns):
        method_name = method_col.replace("_mcc", "")
        vals = df[method_col].fillna(0).tolist()
        color = METHOD_COLORS.get(method_name, f"C{i}")
        offset = (i - n_methods / 2 + 0.5) * height
        ax.barh(y + offset, vals, height, label=method_name, color=color, edgecolor="white", linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{f} (n={df.iloc[j]['n_total']})" for j, f in enumerate(families)], fontsize=8)
    ax.set_xlabel("MCC")
    ax.set_title("Per-Family MCC — Method Comparison")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_dataset_composition(
    df: pd.DataFrame,
    output_path: Path,
    *,
    top_n: int = 20,
) -> None:
    """Figure 3: Stacked bar chart of train/val/test per family."""
    family_counts = df["Protein families"].value_counts().head(top_n)
    families = family_counts.index.tolist()

    train_counts = []
    val_counts = []
    test_counts = []

    for fam in families:
        sub = df[df["Protein families"] == fam]
        train_counts.append(len(sub[sub["Split"] == "train"]))
        val_counts.append(len(sub[sub["Split"] == "val"]))
        test_counts.append(len(sub[sub["Split"] == "test"]))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(families))

    ax.bar(x, train_counts, label="Train", color="#1f77b4")
    ax.bar(x, val_counts, bottom=train_counts, label="Val", color="#ff7f0e")
    ax.bar(x, test_counts,
           bottom=[t + v for t, v in zip(train_counts, val_counts)],
           label="Test", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of sequences")
    ax.set_title("Dataset Composition by Family and Split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_confusion_matrix_grid(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path,
    *,
    threshold: float = 0.5,
) -> None:
    """Figure 4: Grid of binary confusion matrices for each method."""
    methods = list(predictions.keys())
    n = len(methods)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, method in enumerate(methods):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        y_pred = (predictions[method] >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        total = cm.sum()

        ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Nontox", "Toxic"], fontsize=9)
        ax.set_yticklabels(["Nontox", "Toxic"], fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(method, fontsize=10, fontweight="bold")

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = 100 * val / total if total > 0 else 0
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=9, color=color)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Binary Confusion Matrices", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_length_distribution(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Figure 6: Histogram of sequence lengths for toxic vs nontoxic."""
    from toxfam.evaluation.metrics import to_binary_class

    df = df.copy()
    df["binary"] = df["Protein families"].apply(
        lambda x: "Toxic" if to_binary_class(x) != "nontoxin" else "Nontoxic"
    )
    df["length"] = df["Sequence"].str.len()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    bins = np.linspace(0, 1000, 80)
    toxic = df[df["binary"] == "Toxic"]["length"]
    nontox = df[df["binary"] == "Nontoxic"]["length"]

    ax1.hist(toxic, bins=bins, alpha=0.7, label=f"Toxic (n={len(toxic)}, med={toxic.median():.0f}aa)",
             color="#d62728", density=True)
    ax1.hist(nontox, bins=bins, alpha=0.5, label=f"Nontoxic (n={len(nontox)}, med={nontox.median():.0f}aa)",
             color="#1f77b4", density=True)
    ax1.set_xlabel("Sequence Length (aa)")
    ax1.set_ylabel("Density")
    ax1.set_title("Sequence Length Distribution")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1000)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Log-scale box plot
    data = [toxic.values, nontox.values]
    bp = ax2.boxplot(data, labels=["Toxic", "Nontoxic"], patch_artist=True,
                     widths=0.5, showfliers=False)
    bp["boxes"][0].set_facecolor("#d6272844")
    bp["boxes"][1].set_facecolor("#1f77b444")
    ax2.set_ylabel("Sequence Length (aa)")
    ax2.set_title("Length by Class")
    ax2.set_yscale("log")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Figure 7: Overlay ROC curves for all methods."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (method, p_toxic) in enumerate(predictions.items()):
        fpr, tpr, _ = roc_curve(y_true, p_toxic)
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)
        color = METHOD_COLORS.get(method, f"C{i}")
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{method} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Method Comparison")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_pr_curves(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Figure 8: Overlay Precision-Recall curves for all methods."""
    fig, ax = plt.subplots(figsize=(7, 6))

    prevalence = y_true.mean()

    for i, (method, p_toxic) in enumerate(predictions.items()):
        precision, recall, _ = precision_recall_curve(y_true, p_toxic)
        from sklearn.metrics import auc
        pr_auc = auc(recall, precision)
        color = METHOD_COLORS.get(method, f"C{i}")
        ax.plot(recall, precision, color=color, lw=1.5, label=f"{method} (AUC={pr_auc:.3f})")

    ax.axhline(y=prevalence, color="gray", ls="--", lw=0.8, alpha=0.5, label=f"Random ({prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Method Comparison")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_error_venn(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    name_a: str,
    name_b: str,
    output_path: Path,
    *,
    threshold: float = 0.5,
) -> None:
    """Figure 9: Two-circle Venn diagram of errors between two methods."""
    errors_a = set(np.where((pred_a >= threshold).astype(int) != y_true)[0])
    errors_b = set(np.where((pred_b >= threshold).astype(int) != y_true)[0])

    only_a = len(errors_a - errors_b)
    only_b = len(errors_b - errors_a)
    both = len(errors_a & errors_b)
    neither = len(y_true) - len(errors_a | errors_b)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Draw circles
    c1 = Circle((-0.3, 0), 1.0, fill=True, facecolor="#1f77b433", edgecolor="#1f77b4", lw=2)
    c2 = Circle((0.3, 0), 1.0, fill=True, facecolor="#ff7f0e33", edgecolor="#ff7f0e", lw=2)
    ax.add_patch(c1)
    ax.add_patch(c2)

    ax.text(-0.8, 0, f"{only_a}", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0.0, 0, f"{both}", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0.8, 0, f"{only_b}", ha="center", va="center", fontsize=18, fontweight="bold")

    ax.text(-0.8, -0.35, f"{name_a}\nonly", ha="center", va="center", fontsize=9, color="#1f77b4")
    ax.text(0.8, -0.35, f"{name_b}\nonly", ha="center", va="center", fontsize=9, color="#ff7f0e")
    ax.text(0.0, -0.35, "Both", ha="center", va="center", fontsize=9)

    ax.text(-0.3, 1.15, name_a, ha="center", fontsize=11, fontweight="bold", color="#1f77b4")
    ax.text(0.3, 1.15, name_b, ha="center", fontsize=11, fontweight="bold", color="#ff7f0e")

    ax.text(0.0, -1.4, f"Correct: {neither}/{len(y_true)}  |  Total errors: {len(errors_a | errors_b)}",
            ha="center", fontsize=10)

    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.6, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Error Analysis — Complementary Errors", fontsize=13, pad=10)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_family_confusion_mini(
    y_true: np.ndarray,
    families: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path,
    *,
    top_n: int = 10,
    threshold: float = 0.5,
) -> None:
    """Figure 5: Mini confusion matrices for top problematic families."""
    # Find families with most errors (across first method)
    first_method = list(predictions.keys())[0]
    p = predictions[first_method]
    y_pred = (p >= threshold).astype(int)

    unique_fams = sorted(set(families))
    fam_errors = {}
    for fam in unique_fams:
        mask = families == fam
        if mask.sum() < 5:
            continue
        n_err = (y_true[mask] != y_pred[mask]).sum()
        if n_err > 0:
            fam_errors[fam] = n_err

    top_fams = sorted(fam_errors.keys(), key=lambda f: fam_errors[f], reverse=True)[:top_n]

    if not top_fams:
        print("  Skipping per-family confusion matrices (no errors found)")
        return

    n_methods = min(len(predictions), 3)
    method_names = list(predictions.keys())[:n_methods]

    fig, axes = plt.subplots(len(top_fams), n_methods, figsize=(3.5 * n_methods, 2 * len(top_fams)))
    if len(top_fams) == 1:
        axes = axes.reshape(1, -1)
    if n_methods == 1:
        axes = axes.reshape(-1, 1)

    for row, fam in enumerate(top_fams):
        mask = families == fam
        for col, method in enumerate(method_names):
            ax = axes[row, col]
            y_p = (predictions[method][mask] >= threshold).astype(int)
            cm = confusion_matrix(y_true[mask], y_p, labels=[0, 1])

            ax.imshow(cm, cmap="Blues", aspect="auto")
            for i in range(2):
                for j in range(2):
                    color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10, color=color)

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["NT", "T"], fontsize=8)
            ax.set_yticklabels(["NT", "T"], fontsize=8)

            if row == 0:
                ax.set_title(method, fontsize=9, fontweight="bold")
            if col == 0:
                short_name = fam[:25] + "..." if len(fam) > 25 else fam
                ax.set_ylabel(f"{short_name}\n(n={mask.sum()})", fontsize=7)

    fig.suptitle("Per-Family Confusion Matrices (Top Problematic)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_all_figures(
    output_dir: Path,
    *,
    training_csv: Path | None = None,
    metrics: dict[str, dict] | None = None,
    y_true: np.ndarray | None = None,
    predictions: dict[str, np.ndarray] | None = None,
    per_family_df: pd.DataFrame | None = None,
) -> list[Path]:
    """Generate all publication figures. Returns list of saved paths.

    Call this with pre-computed data, or use the CLI command which loads
    everything automatically.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    if training_csv is not None:
        df = pd.read_csv(training_csv)
    else:
        df = None

    # Figure 1: Overall metrics bar
    if metrics:
        p = output_dir / "fig1_overall_metrics.png"
        plot_overall_metrics_bar(metrics, p)
        saved.append(p)

    # Figure 2: Per-family MCC
    if per_family_df is not None:
        mcc_cols = [c for c in per_family_df.columns if c.endswith("_mcc")]
        if mcc_cols:
            p = output_dir / "fig2_per_family_mcc.png"
            plot_per_family_mcc(per_family_df, mcc_cols, p)
            saved.append(p)

    # Figure 3: Dataset composition
    if df is not None:
        p = output_dir / "fig3_dataset_composition.png"
        plot_dataset_composition(df, p)
        saved.append(p)

    # Figure 4: Confusion matrix grid
    if y_true is not None and predictions:
        p = output_dir / "fig4_confusion_matrices.png"
        plot_confusion_matrix_grid(y_true, predictions, p)
        saved.append(p)

    # Figure 5: Per-family confusion matrices
    if y_true is not None and predictions and df is not None:
        test_df = df[df["Split"] == "test"]
        families = test_df["Protein families"].values
        if len(families) == len(y_true):
            p = output_dir / "fig5_per_family_confusion.png"
            plot_per_family_confusion_mini(y_true, families, predictions, p)
            saved.append(p)

    # Figure 6: Length distribution
    if df is not None:
        p = output_dir / "fig6_length_distribution.png"
        plot_length_distribution(df, p)
        saved.append(p)

    # Figure 7: ROC curves
    if y_true is not None and predictions:
        p = output_dir / "fig7_roc_curves.png"
        plot_roc_curves(y_true, predictions, p)
        saved.append(p)

    # Figure 8: PR curves
    if y_true is not None and predictions:
        p = output_dir / "fig8_pr_curves.png"
        plot_pr_curves(y_true, predictions, p)
        saved.append(p)

    # Figure 9: Error Venn
    if y_true is not None and predictions and len(predictions) >= 2:
        method_list = list(predictions.keys())
        # Compare the two best methods
        p = output_dir / "fig9_error_venn.png"
        plot_error_venn(y_true, predictions[method_list[0]], predictions[method_list[1]],
                        method_list[0], method_list[1], p)
        saved.append(p)

    print(f"\n  Generated {len(saved)} figures in {output_dir}")
    return saved
