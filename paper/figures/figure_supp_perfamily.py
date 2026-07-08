"""Supplementary figure — per-family resolution dumbbell (ToxFam vs. homology).

Per-family one-vs-rest MCC for HBI and ToxFam as a dumbbell: the absolute
capability of both methods AND the per-family gap in one panel. Markers are
uniform size with each family's support printed in its y-axis label;
semi-transparent markers + explicit draw order keep both endpoints readable
when they nearly coincide. No per-point error bars: 37x2 whiskers would bury
the gap, and the 20 families with <=5 toxins cannot support a valid 95% CI
(n<10).

Relocated from the main text (was Figure 3, panel A). The confident-error
adjudication that used to share its figure is now carried entirely by
figure_confidence_curation, so this is a single-panel supplementary figure.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from paper.figures._common import (
    DOUBLE_COL, METHOD_DARK, METHODS, apply_style, load_preds, save_fig, test_set_class_list,
)
from paper.stats import per_family_mcc_difference

MARKER_S = 30  # uniform marker size; family support now lives in the y-axis labels


def _fam_label(name, support):
    """Drop the redundant trailing ' family', spell out 'other', append exact support."""
    if name == "other":
        base = "other toxin family"
    elif name.endswith(" family"):
        base = name[: -len(" family")]
    else:
        base = name  # keep 'superfamily' etc. intact
    return f"{base} ($n$={int(support)})"


def main() -> None:
    apply_style()
    classes = test_set_class_list()
    hbi = load_preds("test_set", "hbi")
    nn = load_preds("test_set", "nn_combined_run")

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 6.4), layout="constrained")

    # --- per-family MCC dumbbell (HBI vs ToxFam), sorted by ToxFam MCC ---
    fam = (per_family_mcc_difference(nn, hbi, class_list=classes)
           .dropna(subset=["mcc_a", "mcc_b"]).sort_values("mcc_a").reset_index(drop=True))
    y = np.arange(len(fam))
    hbi_lbl, hbi_col = METHODS["hbi"]
    nn_lbl, nn_col = METHODS["nn_combined_run"]
    # connector first, then HBI, then ToxFam on top; uniform marker size (support is in labels).
    ax.hlines(y, fam["mcc_b"], fam["mcc_a"], color="#cccccc", lw=0.9, zorder=1)
    ax.scatter(fam["mcc_b"], y, s=MARKER_S, color=hbi_col, marker="o",
               edgecolor=METHOD_DARK["hbi"], linewidth=0.5, alpha=0.7, label=hbi_lbl, zorder=2)
    ax.scatter(fam["mcc_a"], y, s=MARKER_S, color=nn_col, marker="s",
               edgecolor=METHOD_DARK["nn_combined_run"], linewidth=0.5, alpha=0.7,
               label=nn_lbl, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([_fam_label(n, s) for n, s in zip(fam["family"], fam["support"])],
                       fontsize=7)
    ax.set_ylim(-0.8, len(fam) - 0.2)
    ax.set_xlim(-0.05, 1.06)
    ax.set_xlabel("One-vs-rest MCC")
    # Single frameless method legend in the empty upper-left wedge (top families sit at
    # high MCC, so their left side carries no markers). Exact support is printed in each
    # y-axis label, so marker area is a self-evident echo and needs no size legend.
    method_leg = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.99), handletextpad=0.5,
                           title="method", frameon=False, labelspacing=0.5)
    method_leg.get_title().set_fontsize(7.5)

    save_fig(fig, "figure_supp_perfamily")


if __name__ == "__main__":
    main()
