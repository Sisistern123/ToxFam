"""Supplementary Figure — accuracy views of the capability comparison.

Interpretable but redundant in ranking with the main-figure multiclass MCC, so
these live in the supplement. Panel A: toxin-only accuracy (n=515). Panel B:
all-class accuracy, near-ceiling for every method just above the 94.73% non-toxin
prior (dashed line) -- which is why it is uninformative as a headline. +-2 SE.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from paper.figures._common import (
    METHOD_ORDER, METHODS, SINGLE_COL, apply_style, fmt_pm, load_preds, panel_label, save_fig,
)
from paper.stats import bootstrap_accuracy_ci, correctness, toxin_mask


def _acc_panel(ax, preds, mask_fn, ylim, title, letter, *, prior=None):
    x = np.arange(len(METHOD_ORDER))
    for i, k in enumerate(METHOD_ORDER):
        _, color = METHODS[k]
        d = preds[k]
        c = correctness(d)[mask_fn(d)] if mask_fn else correctness(d)
        ci = bootstrap_accuracy_ci(c)
        ax.bar(i, ci["point"], 0.62, yerr=ci["two_se"], capsize=3, color=color,
               edgecolor="white", linewidth=0.0, error_kw={"elinewidth": 0.8, "capthick": 0.8})
        ax.text(i, ci["point"] + ci["two_se"] + 0.004, fmt_pm(ci["point"], ci["two_se"], sep="\n±"),
                ha="center", va="bottom", fontsize=7, linespacing=0.95)
    if prior is not None:  # drawn on top of the bars so the ceiling is unmistakable
        ax.axhline(prior, color="#444444", lw=1.0, ls=(0, (4, 2)), zorder=5)
        ax.text(len(METHOD_ORDER) - 0.55, prior + 0.002, f"non-toxin prior {prior * 100:.2f}%",
                ha="right", va="bottom", fontsize=6.5, color="#444444", zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels([METHODS[k][0].replace(" (", "\n(") for k in METHOD_ORDER])
    ax.set_ylim(*ylim)
    ax.set_ylabel("Accuracy")
    ax.set_title(title, loc="left", pad=8)
    panel_label(ax, letter)


def main() -> None:
    apply_style()
    preds = {k: load_preds("test_set", k) for k in METHOD_ORDER}
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(SINGLE_COL, 4.7), layout="constrained")
    n_tox = int(toxin_mask(preds["nn_combined_run"]).sum())  # from data, not a frozen literal
    _acc_panel(axA, preds, toxin_mask, (0.70, 1.0), f"Toxin-only accuracy ($n$={n_tox})", "A")
    prior = float((preds["nn_combined_run"]["actual_label"].str.lower() == "nontox").mean())
    _acc_panel(axB, preds, None, (0.90, 1.0), "All-class accuracy", "B", prior=prior)
    save_fig(fig, "figure_supp_accuracy")


if __name__ == "__main__":
    main()
