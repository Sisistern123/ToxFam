"""Tests for toxfam.visualization — smoke tests verifying plot functions produce files."""

from __future__ import annotations

import numpy as np

# Use non-interactive backend for CI
import matplotlib

matplotlib.use("Agg")


def test_plot_loss_curve(tmp_path):
    """plot_loss_curve produces a PNG file."""
    from toxfam.visualization.plots import plot_loss_curve

    history = {
        "train_losses": [1.0, 0.8, 0.6],
        "val_losses": [1.1, 0.9, 0.7],
    }
    out = tmp_path / "loss.png"
    plot_loss_curve(history, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_confusion_matrix(tmp_path):
    """plot_confusion_matrix produces a PNG file."""
    from toxfam.visualization.plots import plot_confusion_matrix

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    classes = ["A", "B", "C"]
    out = tmp_path / "cm.png"
    plot_confusion_matrix(y_true, y_pred, classes, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_confusion_matrix_creates_parent_dir(tmp_path):
    """plot_confusion_matrix creates parent directories if needed."""
    from toxfam.visualization.plots import plot_confusion_matrix

    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    out = tmp_path / "sub" / "dir" / "cm.png"
    plot_confusion_matrix(y_true, y_pred, ["A", "B"], str(out))
    assert out.exists()


def test_plot_binary_roc(tmp_path):
    """plot_binary_roc produces a PNG file."""
    from toxfam.visualization.analysis import plot_binary_roc

    fpr = [0.0, 0.5, 1.0]
    tpr = [0.0, 0.8, 1.0]
    roc_auc = 0.85
    out = tmp_path / "roc.png"
    plot_binary_roc(fpr, tpr, roc_auc, str(out))
    assert out.exists()


def test_plot_binary_pr(tmp_path):
    """plot_binary_pr produces a PNG file."""
    from toxfam.visualization.analysis import plot_binary_pr

    precision = [0.5, 0.8, 1.0]
    recall = [1.0, 0.5, 0.0]
    pr_auc = 0.75
    out = tmp_path / "pr.png"
    plot_binary_pr(precision, recall, pr_auc, str(out))
    assert out.exists()
