"""Figure 1 — capability across 38 families + validated superiority over HBI."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.figures._common import apply_style, load_preds, save_fig, test_class_list
from toxfam.evaluation.manuscript import (
    correctness, mcnemar_test, paired_bootstrap_accuracy_diff, per_family_f1_difference,
    subset_accuracy, toxin_mask,
)
from toxfam.evaluation.metrics import calculate_metrics


def main() -> None:
    apply_style()
    classes = test_class_list()
    hbi = load_preds("test_set", "hbi")
    nn = load_preds("test_set", "nn_combined_run")
    std = load_preds("test_set", "nn_standard_run")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # --- Panel A: per-family support (log) vs NN F1 (capability across Metazoa) ---
    fam = per_family_f1_difference(nn, hbi, class_list=classes)  # has f1_a (NN), support
    fam = fam.sort_values("support", ascending=False)
    axA = axes[0]
    axA.scatter(fam["support"], fam["f1_a"], s=18, color="#c0504d", alpha=0.8)
    axA.set_xscale("log")
    axA.set_xlabel("Family support (test, log)"); axA.set_ylabel("ToxFam F1")
    axA.set_title(f"A. {len(fam)} toxin families resolved across Metazoa")
    axA.set_ylim(0, 1.02)

    # --- Panel B: toxin-only accuracy headline + all-class reference ---
    axB = axes[1]
    methods = [("HBI", hbi, "#7f7f7f"), ("ToxFam (emb+tax)", nn, "#c0504d")]
    tox = [subset_accuracy(d, toxin_mask(d)) for _, d, _ in methods]
    allc = [subset_accuracy(d) for _, d, _ in methods]
    x = np.arange(len(methods))
    axB.bar(x - 0.18, tox, 0.36, label="toxin-only", color=[c for *_, c in methods])
    axB.bar(x + 0.18, allc, 0.36, label="all-class", color=[c for *_, c in methods], alpha=0.45)
    for xi, (t, a) in enumerate(zip(tox, allc)):
        axB.text(xi - 0.18, t + 0.005, f"{t:.3f}", ha="center", fontsize=8)
        axB.text(xi + 0.18, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
    mc = mcnemar_test(correctness(nn), correctness(hbi))
    bs = paired_bootstrap_accuracy_diff(correctness(nn), correctness(hbi))
    axB.set_xticks(x); axB.set_xticklabels([m for m, *_ in methods])
    axB.set_ylim(0.8, 1.0); axB.set_ylabel("Accuracy"); axB.legend(loc="lower left")
    axB.set_title(f"B. Toxin-only headline (McNemar p={mc['p_value']:.3f};\n"
                  f"Δacc {bs['diff']:+.4f} [{bs['ci_low']:+.4f},{bs['ci_high']:+.4f}])")
    axB.text(0.5, 0.80, "non-toxin prior = 94.73%", transform=axB.transAxes,
             ha="center", va="bottom", fontsize=7, color="gray")

    # --- Panel C: macro & weighted P/R/F1 for the three methods ---
    axC = axes[2]
    rep = {n: calculate_metrics(d["actual_label"], d["predicted_label"], class_list=classes)
           for n, d in [("HBI", hbi), ("emb", std), ("emb+tax", nn)]}
    metrics = [("macro\nprec", "macro avg", "precision"), ("macro\nrecall", "macro avg", "recall"),
               ("macro\nF1", "macro avg", "f1-score"), ("weighted\nF1", "weighted avg", "f1-score")]
    width = 0.25
    for j, (mname, m) in enumerate(rep.items()):
        vals = [m.classification_report[avg][k] for _, avg, k in metrics]
        axC.bar(np.arange(len(metrics)) + (j - 1) * width, vals, width, label=mname)
    axC.set_xticks(np.arange(len(metrics))); axC.set_xticklabels([m for m, *_ in metrics])
    axC.set_ylim(0, 1.05); axC.legend(fontsize=7); axC.set_title("C. Macro / weighted P-R-F1")

    fig.tight_layout()
    save_fig(fig, "figure1_capability")


if __name__ == "__main__":
    main()
