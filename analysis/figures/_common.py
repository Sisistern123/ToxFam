"""Shared loaders and matplotlib style for manuscript figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from toxfam._paths import benchmark_dir, get_project_root, processed_dir

FIG_DIR = get_project_root() / "analysis" / "manuscript_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_preds(dataset: str, method: str) -> pd.DataFrame:
    return pd.read_csv(benchmark_dir() / dataset / method / "predictions.csv")


def test_class_list() -> list[str]:
    """The 38-class label space = sorted unique actual labels on the test set."""
    df = load_preds("test_set", "nn_combined_run")
    return sorted(df["actual_label"].unique().tolist())


def sequence_lengths() -> pd.Series:
    df = pd.read_csv(processed_dir() / "training_data.csv")
    return pd.Series(df["Sequence"].str.len().values, index=df["identifier"].values)


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save both PNG (300 dpi) and PDF (vector) into FIG_DIR."""
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}.png / .pdf")


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120, "savefig.bbox": "tight",
    })


# Consistent method colors/labels across all figures
METHODS = {
    "hbi": ("HBI", "#7f7f7f"),
    "nn_standard_run": ("ToxFam (emb)", "#5b9bd5"),
    "nn_combined_run": ("ToxFam (emb+tax)", "#c0504d"),
}
