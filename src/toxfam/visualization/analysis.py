from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def analyze_label_distribution_for_split(
    train_df, val_df, test_df, label_col, output_dir
):
    """Save counts + plot + chi-square for *label_col* across the three splits."""
    train_counts = train_df[label_col].value_counts().sort_index()
    val_counts = val_df[label_col].value_counts().sort_index()
    test_counts = test_df[label_col].value_counts().sort_index()

    dist_df = pd.DataFrame(
        {
            "Train": train_counts,
            "Validation": val_counts,
            "Test": test_counts,
        }
    ).fillna(0)

    out = Path(output_dir)
    metrics_dir = out / "metrics"
    plots_dir = out / "plots"

    dist_json = metrics_dir / f"{label_col.replace(' ', '_')}_distribution.json"
    dist_df.to_json(dist_json, orient="index")

    fig, ax = plt.subplots(figsize=(10, 6))
    dist_df.plot(kind="bar", logy=True, ax=ax)
    ax.tick_params(axis="x", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=50, ha="right")
    fig.subplots_adjust(bottom=0.4)
    ax.set_title(f"Distribution of {label_col} Across Splits (log scale)")
    ax.set_ylabel("Count (log scale)")
    fig.savefig(plots_dir / f"{label_col.replace(' ', '_')}_distribution_log.png")
    plt.close(fig)

    chi2, p, dof, expected = chi2_contingency(dist_df.T)
    with (metrics_dir / f"{label_col.replace(' ', '_')}_chi_square.json").open(
        "w"
    ) as fp:
        json.dump({"chi2": chi2, "p_value": p, "dof": dof}, fp, indent=4)


def plot_multiclass_roc_from_scores(
    y_true, y_scores, classes, output_path, legend_cols=3
):
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    n_classes = y_bin.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    cmap = plt.cm.get_cmap("rainbow", n_classes)
    colors = [
        (0.8 * r, 0.8 * g, 0.8 * b) for (r, g, b, a) in cmap(np.arange(n_classes))
    ]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=180, constrained_layout=True)
    for i, cname in enumerate(classes):
        ax.plot(
            fpr[i],
            tpr[i],
            color=colors[i],
            lw=1.5,
            label=f"{cname} (AUC {roc_auc[i]:.2f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="gray")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=legend_cols,
        fontsize="small",
        frameon=False,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_binary_roc(fpr, tpr, roc_auc, output_path, title="Binary ROC Curve"):
    """Plot ROC curve for binary toxic/nontoxin classification."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_binary_pr(precision, recall, pr_auc, output_path, title="Binary PR Curve"):
    """Plot Precision-Recall curve for binary classification."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
