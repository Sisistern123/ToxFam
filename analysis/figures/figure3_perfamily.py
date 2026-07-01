"""Figure 3 — per-family resolution (dumbbell) + confident-error adjudication.

Panel A: per-family one-vs-rest MCC for HBI and ToxFam as a dumbbell (absolute
capability of both methods AND the gap, in one panel). Markers are uniform size
with each family's support printed in its y-axis label; semi-transparent markers
+ explicit draw order keep both endpoints readable when they nearly coincide. No
per-point error bars: 37x2 whiskers would bury the gap, and the 20 families with
<=5 toxins cannot support a valid 95% CI (n<10).
Panel B: adjudication of the 63 most-confident errors, luminance-ordered ramp.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.figures._common import (
    ADJUDICATION, DOUBLE_COL, METHOD_DARK, METHODS, apply_style, load_preds, panel_label,
    save_fig, test_set_class_list,
)
from toxfam._paths import get_project_root
from toxfam.evaluation.manuscript import adjudication_summary, per_family_mcc_difference

ADJ_CSV = get_project_root() / "analysis" / "model_test_wrong_conf_annotated.csv"
MARKER_S = 30  # uniform marker size; family support now lives in the y-axis labels
# (An "adjudicated" third series was explored -- recompute per-family MCC counting the
# expert-confirmed confident errors as correct -- but the 63 adjudicated proteins span ALL
# splits (37 train / 15 val / 11 test) while this panel is test-only, so only 2 of the 33
# 'correct' proteins fall in it. A per-family adjudicated overlay is therefore not meaningful
# on a test-only figure, so it was dropped; Fig 3B summarises the adjudication instead.)


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

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL, 7.6), gridspec_kw={"width_ratios": [2.7, 1]}, layout="constrained")

    # --- Panel A: per-family MCC dumbbell (HBI vs ToxFam), sorted by ToxFam MCC ---
    fam = (per_family_mcc_difference(nn, hbi, class_list=classes)
           .dropna(subset=["mcc_a", "mcc_b"]).sort_values("mcc_a").reset_index(drop=True))
    y = np.arange(len(fam))
    hbi_lbl, hbi_col = METHODS["hbi"]
    nn_lbl, nn_col = METHODS["nn_combined_run"]
    # connector first, then HBI, then ToxFam on top; uniform marker size (support is in labels).
    axA.hlines(y, fam["mcc_b"], fam["mcc_a"], color="#cccccc", lw=0.9, zorder=1)
    axA.scatter(fam["mcc_b"], y, s=MARKER_S, color=hbi_col, marker="o",
                edgecolor=METHOD_DARK["hbi"], linewidth=0.5, alpha=0.7, label=hbi_lbl, zorder=2)
    axA.scatter(fam["mcc_a"], y, s=MARKER_S, color=nn_col, marker="s",
                edgecolor=METHOD_DARK["nn_combined_run"], linewidth=0.5, alpha=0.7,
                label=nn_lbl, zorder=3)
    axA.set_yticks(y)
    axA.set_yticklabels([_fam_label(n, s) for n, s in zip(fam["family"], fam["support"])],
                        fontsize=7)
    axA.set_ylim(-0.8, len(fam) - 0.2)
    axA.set_xlim(-0.05, 1.06)
    axA.set_xlabel("One-vs-rest MCC")
    axA.set_title("Per-family resolution: ToxFam vs. homology", loc="left", pad=6)
    panel_label(axA, "A", dx=-0.30)
    # Single frameless method legend in the empty upper-left wedge (top families sit at
    # high MCC, so their left side carries no markers). The size legend is gone: exact
    # support is now printed in each y-axis label, so marker area is a self-evident echo.
    method_leg = axA.legend(loc="upper left", bbox_to_anchor=(0.02, 0.99), handletextpad=0.5,
                            title="method", frameon=False, labelspacing=0.5)
    method_leg.get_title().set_fontsize(7.5)

    # --- Panel B: confident-error adjudication, luminance-ordered ramp ---
    s = adjudication_summary(ADJ_CSV)
    order = ["correct", "partial", "incorrect"]
    counts = [s["assessment"].get(k, 0) for k in order]
    total = s["n"]
    text_col = {"correct": "white", "partial": "black", "incorrect": "white"}
    bottom = 0
    for k, c in zip(order, counts):
        axB.bar(0, c, bottom=bottom, width=0.6, color=ADJUDICATION[k], edgecolor="white",
                linewidth=0.6)
        axB.text(0, bottom + c / 2, f"{k}\n{c} ({c / total:.0%})", ha="center", va="center",
                 fontsize=7.5, color=text_col[k], fontweight="bold")
        bottom += c
    vindicated = s["assessment"].get("correct", 0) + s["assessment"].get("partial", 0)
    axB.text(0, total + 1.2, f"{vindicated}/{total} ({vindicated / total:.0%})\nmodel-vindicated",
             ha="center", va="bottom", fontsize=7.5)
    # No numeric y-axis: the stacked segments are directly labelled (count + %), so the
    # 0--63 scale is redundant; the descriptive title carries what the data are.
    axB.set_xlim(-0.6, 0.6)
    axB.set_xticks([])
    axB.set_ylim(0, total + 6)
    axB.set_yticks([])
    for sp in axB.spines.values():
        sp.set_visible(False)
    # smaller title: "adjudication" is wider than this narrow panel at the 9 pt default,
    # so it would overhang the figure's right edge (clipped once savefig.bbox is not tight)
    axB.set_title("Confident-error adjudication\n($\\geq$0.8 prob.; $n$=63)", loc="left", pad=6,
                  fontsize=8)
    panel_label(axB, "B", dx=-0.05)

    save_fig(fig, "figure3_perfamily")


if __name__ == "__main__":
    main()
