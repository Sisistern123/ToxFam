"""Figure 1 — validated superiority over homology (multiclass MCC).

Single panel, single metric: the multiclass (Gorodkin) MCC for HBI, ToxFam (emb)
and ToxFam (emb+tax), with +-2 bootstrap SE. This is the imbalance-robust measure
the Methods designate as primary, and it carries the whole comparison on its own.

The interpretable-but-ranking-redundant accuracy views (toxin-only and all-class)
live in Supplementary Figure S1 so the main figure shows only what is needed.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.figures._common import (
    MCC_CI_N_BOOT, METHODS, SINGLE_COL, apply_style, fmt_pm, load_preds, save_fig,
)
from toxfam.evaluation.manuscript import bootstrap_label_metric_ci, overall_mcc

METHOD_ORDER = ["hbi", "nn_standard_run", "nn_combined_run"]


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.9), layout="constrained")
    x = np.arange(len(METHOD_ORDER))
    for i, k in enumerate(METHOD_ORDER):
        _, color = METHODS[k]
        d = load_preds("test_set", k)
        ci = bootstrap_label_metric_ci(d["actual_label"].values, d["predicted_label"].values,
                                       overall_mcc, n_boot=MCC_CI_N_BOOT)
        ax.bar(i, ci["point"], 0.62, yerr=ci["two_se"], capsize=3, color=color,
               edgecolor="white", linewidth=0.0, error_kw={"elinewidth": 0.8, "capthick": 0.8})
        ax.text(i, ci["point"] + ci["two_se"] + 0.006, fmt_pm(ci["point"], ci["two_se"], sep="\n±"),
                ha="center", va="bottom", fontsize=7.5, linespacing=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels([METHODS[k][0].replace(" (", "\n(") for k in METHOD_ORDER])
    ax.set_ylim(0.75, 0.95)
    ax.set_ylabel("Multiclass MCC")
    save_fig(fig, "figure1_capability")


if __name__ == "__main__":
    main()
