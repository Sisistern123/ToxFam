"""Figure 3 — per-family resolution + confident-error adjudication."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.figures._common import apply_style, load_preds, save_fig, test_class_list
from toxfam._paths import get_project_root
from toxfam.evaluation.manuscript import (
    adjudication_summary, macro_mcc_by_support, per_family_mcc_difference,
)

ADJ_CSV = get_project_root() / "analysis" / "model_test_wrong_conf_annotated.csv"


def main() -> None:
    apply_style()
    classes = test_class_list()
    hbi = load_preds("test_set", "hbi")
    nn = load_preds("test_set", "nn_combined_run")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: per-family one-vs-rest MCC difference (NN - HBI), sorted, sized by support ---
    fam = per_family_mcc_difference(nn, hbi, class_list=classes)
    axA = axes[0]
    colors = np.where(fam["diff"] >= 0, "#c0504d", "#7f7f7f")
    sizes = 10 + 3 * np.sqrt(fam["support"].clip(lower=1))
    axA.scatter(fam["diff"], np.arange(len(fam)), s=sizes, color=colors)
    axA.axvline(0, color="black", lw=0.6)
    axA.set_yticks(np.arange(len(fam))); axA.set_yticklabels(fam["family"], fontsize=6)
    axA.set_xlabel("one-vs-rest MCC difference (ToxFam - HBI)")
    strat = macro_mcc_by_support(nn, hbi, class_list=classes, support_threshold=5)
    sup = strat[strat["group"] == "support>5"].iloc[0]
    low = strat[strat["group"] == "support<=5"].iloc[0]
    axA.set_title(
        f"A. Per-family MCC (marker∝support)\n"
        f"support>5: ToxFam {sup['macro_mcc_a']:.3f} vs HBI {sup['macro_mcc_b']:.3f} | "
        f"support≤5 (n={low['n_sequences']}): {low['macro_mcc_a']:.3f} vs {low['macro_mcc_b']:.3f}"
    )

    # --- Panel B: confident-error adjudication stacked bar ---
    axB = axes[1]
    s = adjudication_summary(ADJ_CSV)
    order = ["correct", "partial", "incorrect"]
    counts = [s["assessment"].get(k, 0) for k in order]
    colors_b = ["#4caf50", "#ffb300", "#7f7f7f"]
    bottom = 0
    for k, c, col in zip(order, counts, colors_b):
        axB.bar(0, c, bottom=bottom, color=col, label=f"{k} ({c})"); bottom += c
    axB.set_xlim(-1, 1); axB.set_xticks([]); axB.set_ylabel("Confident (≥0.8) errors")
    axB.legend(loc="upper right", fontsize=8)
    axB.set_title(
        f"B. Adjudicated confident errors (n={s['n']})\n"
        f"{s['assessment'].get('correct',0)+s['assessment'].get('partial',0)}/{s['n']} model-vindicated; "
        f"{s['n_annotation_gaps']} candidate ToxProt gaps"
    )
    # worked examples annotation
    axB.text(0.0, -0.12, "e.g. P00601 (PLA2), F8J2F6 (Kunitz) — labelled nontox, absent from ToxProt",
             transform=axB.transAxes, ha="center", fontsize=7, color="gray")

    fig.tight_layout()
    save_fig(fig, "figure3_perfamily")


if __name__ == "__main__":
    main()
