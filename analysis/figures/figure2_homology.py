"""Figure 2 — ToxFam's advantage is concentrated where homology breaks.

Two panels, both in-Metazoa strengths: (A) toxin-only accuracy vs sequence length
with HBI overlaid, and (B) no-hit coverage. The non-metazoan OOD result is a
negative/scope-boundary finding (ToxFam is metazoan-specific) and is reported as a
limitation in the Discussion/Supplementary, not here.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.figures._common import apply_style, load_preds, save_fig, sequence_lengths
from toxfam.evaluation.manuscript import (
    accuracy_by_length_bins, bootstrap_accuracy_ci, correctness, rolling_accuracy_vs_length,
    toxin_mask,
)
from toxfam.evaluation.hbi import NO_HIT_LABEL


def main() -> None:
    apply_style()
    hbi = load_preds("test_set", "hbi")
    nn = load_preds("test_set", "nn_combined_run")
    lengths = sequence_lengths()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    # --- Panel A: toxin-only rolling accuracy vs length, HBI vs NN ---
    axA = axes[0]
    for d, (name, color) in ((hbi, ("HBI", "#7f7f7f")), (nn, ("ToxFam", "#c0504d"))):
        tox = d[toxin_mask(d)]
        roll = rolling_accuracy_vs_length(tox, lengths, window=50)
        axA.plot(roll["length"], roll["accuracy"], color=color, label=name, lw=1.6)
    axA.axvspan(0, 30, color="orange", alpha=0.08)
    axA.set_xscale("log"); axA.set_xlabel("Sequence length (aa, log)")
    axA.set_ylabel("Toxin-only accuracy"); axA.set_ylim(0.4, 1.02); axA.legend()
    # annotate the <30 aa collapse from fixed bins
    binsA = accuracy_by_length_bins(hbi[toxin_mask(hbi)], lengths, bins=[0, 30, 50, 75, 150, 5000])
    binsN = accuracy_by_length_bins(nn[toxin_mask(nn)], lengths, bins=[0, 30, 50, 75, 150, 5000])
    a30 = binsN.loc[binsN["bin_label"] == "0-30", "accuracy"].iloc[0]
    h30 = binsA.loc[binsA["bin_label"] == "0-30", "accuracy"].iloc[0]
    axA.set_title(f"A. <30 aa: HBI {h30:.3f} vs ToxFam {a30:.3f}")

    # --- Panel B: no-hit coverage, split toxin vs non-toxin ---
    axB = axes[1]
    nohit_ids = hbi.loc[hbi["predicted_label"] == NO_HIT_LABEL, "identifier"]
    nn_nh = nn[nn["identifier"].isin(nohit_ids)]
    tox_m = toxin_mask(nn_nh)
    groups = [("toxin no-hit", nn_nh[tox_m]), ("non-toxin no-hit", nn_nh[~tox_m])]
    labels, nn_acc, hbi_acc = [], [], []
    nn_err = [[], []]  # asymmetric [lower, upper] for ToxFam bars
    for gname, g in groups:
        ci = bootstrap_accuracy_ci(correctness(g))
        labels.append(f"{gname}\n(n={len(g)})"); nn_acc.append(ci["point"]); hbi_acc.append(0.0)
        nn_err[0].append(ci["point"] - ci["ci_low"]); nn_err[1].append(ci["ci_high"] - ci["point"])
    x = np.arange(len(groups))
    axB.bar(x - 0.2, hbi_acc, 0.4, label="HBI (no hit)", color="#7f7f7f")
    axB.bar(x + 0.2, nn_acc, 0.4, yerr=nn_err, capsize=3, label="ToxFam", color="#c0504d")
    axB.set_xticks(x); axB.set_xticklabels(labels); axB.set_ylim(0, 1.05); axB.legend()
    axB.set_title(f"B. No-hit coverage (n={len(nn_nh)}: HBI 0% by construction)")

    fig.tight_layout()
    save_fig(fig, "figure2_homology")


if __name__ == "__main__":
    main()
