"""Figure 1 — capability across 38 families + validated superiority over HBI.

Panel A: per-family one-vs-rest MCC vs support (capability across Metazoa).
Panel B: toxin-only + all-class accuracy with bootstrap 95% CI error bars + paired test.
Panel C: MCC and micro-MCC per method with bootstrap 95% CI error bars.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.figures._common import apply_style, load_preds, save_fig, test_class_list
from toxfam.evaluation.manuscript import (
    bootstrap_accuracy_ci, bootstrap_label_metric_ci, correctness, mcnemar_test, micro_mcc,
    overall_mcc, paired_bootstrap_accuracy_diff, per_family_mcc_difference, subset_accuracy,
    toxin_mask,
)


def _acc_bars(ax, points, cis, x, width, colors, label, alpha=1.0):
    yerr = np.array([[p - c["ci_low"] for p, c in zip(points, cis)],
                     [c["ci_high"] - p for p, c in zip(points, cis)]])
    ax.bar(x, points, width, yerr=yerr, capsize=3, label=label, color=colors, alpha=alpha)


def main() -> None:
    apply_style()
    classes = test_class_list()
    hbi = load_preds("test_set", "hbi")
    nn = load_preds("test_set", "nn_combined_run")
    std = load_preds("test_set", "nn_standard_run")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # --- Panel A: per-family one-vs-rest MCC vs support (capability across Metazoa) ---
    fam = per_family_mcc_difference(nn, hbi, class_list=classes).sort_values("support", ascending=False)
    axA = axes[0]
    axA.scatter(fam["support"], fam["mcc_a"], s=18, color="#c0504d", alpha=0.8)
    axA.set_xscale("log")
    axA.set_xlabel("Family support (test, log)"); axA.set_ylabel("ToxFam one-vs-rest MCC")
    axA.set_title(f"A. {len(fam)} toxin families resolved across Metazoa")
    axA.set_ylim(-0.1, 1.05)

    # --- Panel B: toxin-only accuracy headline + all-class reference (bootstrap CIs) ---
    axB = axes[1]
    methods = [("HBI", hbi, "#7f7f7f"), ("ToxFam (emb+tax)", nn, "#c0504d")]
    colors = [c for *_, c in methods]
    tox_ci = [bootstrap_accuracy_ci(correctness(d)[toxin_mask(d)]) for _, d, _ in methods]
    all_ci = [bootstrap_accuracy_ci(correctness(d)) for _, d, _ in methods]
    tox = [c["point"] for c in tox_ci]; allc = [c["point"] for c in all_ci]
    x = np.arange(len(methods))
    _acc_bars(axB, tox, tox_ci, x - 0.18, 0.36, colors, "toxin-only")
    _acc_bars(axB, allc, all_ci, x + 0.18, 0.36, colors, "all-class", alpha=0.45)
    for xi, (t, a) in enumerate(zip(tox, allc)):
        axB.text(xi - 0.18, t + 0.012, f"{t:.3f}", ha="center", fontsize=8)
        axB.text(xi + 0.18, a + 0.012, f"{a:.3f}", ha="center", fontsize=8)
    mc = mcnemar_test(correctness(nn), correctness(hbi))
    bs = paired_bootstrap_accuracy_diff(correctness(nn), correctness(hbi))
    axB.set_xticks(x); axB.set_xticklabels([m for m, *_ in methods])
    axB.set_ylim(0.8, 1.0); axB.set_ylabel("Accuracy"); axB.legend(loc="lower left")
    axB.set_title(f"B. Toxin-only headline (McNemar p={mc['p_value']:.3f};\n"
                  f"Δacc {bs['diff']:+.4f} [{bs['ci_low']:+.4f},{bs['ci_high']:+.4f}])")
    axB.text(0.5, 0.80, "non-toxin prior = 94.73%", transform=axB.transAxes,
             ha="center", va="bottom", fontsize=7, color="gray")

    # --- Panel C: MCC and micro-MCC per method, with bootstrap 95% CIs ---
    axC = axes[2]
    metric_defs = [("MCC", overall_mcc),
                   ("micro-MCC", lambda yt, yp: micro_mcc(yt, yp, class_list=classes))]
    methods_c = [("HBI", hbi, "#7f7f7f"), ("ToxFam (emb)", std, "#5b9bd5"),
                 ("ToxFam (emb+tax)", nn, "#c0504d")]
    xm = np.arange(len(metric_defs)); width = 0.25
    for k, (label, d, color) in enumerate(methods_c):
        pts, lo, hi = [], [], []
        for _, mfn in metric_defs:
            ci = bootstrap_label_metric_ci(d["actual_label"].values, d["predicted_label"].values,
                                           mfn, n_boot=500)
            pts.append(ci["point"]); lo.append(ci["point"] - ci["ci_low"]); hi.append(ci["ci_high"] - ci["point"])
        axC.bar(xm + (k - 1) * width, pts, width, yerr=[lo, hi], capsize=3, label=label, color=color)
    axC.set_xticks(xm); axC.set_xticklabels([m for m, _ in metric_defs])
    axC.set_ylim(0.7, 1.0); axC.legend(fontsize=7); axC.set_title("C. MCC (bootstrap 95% CI)")

    fig.tight_layout()
    save_fig(fig, "figure1_capability")


if __name__ == "__main__":
    main()
