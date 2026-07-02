"""Capability figure --- ToxFam's family-level advantage over homology, and where
it concentrates.

Merges the former Fig. 1 (multiclass-MCC bars) and Fig. 2 (homology break-down)
into one figure so the whole HBI comparison reads as a single message:
  (A) multiclass (Gorodkin) MCC for HBI, ToxFam (emb) and ToxFam (emb+tax) --- the
      imbalance-robust headline metric, +-2 bootstrap SE;
  (B) toxin-only accuracy across sequence-length bins: HBI collapses on short toxins
      while ToxFam stays roughly flat; +-2 bootstrap SE, with a length rug for density;
  (C) ToxFam coverage on the proteins where HBI returns no hit (HBI = 0% there by
      construction).

Interpretable-but-ranking-redundant accuracy views (toxin-only, all-class) live in
Supplementary Fig. S1 so the main figure shows only what is needed.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from analysis.figures._common import (
    DOUBLE_COL, LEN_BINS, MCC_CI_N_BOOT, METHOD_DARK, METHOD_LINESTYLE, METHOD_MARKER,
    METHOD_ORDER, METHODS, apply_style, fmt_pm, load_preds, panel_label, save_fig,
    sequence_lengths,
)
from toxfam.evaluation.hbi import NO_HIT_LABEL
from toxfam.evaluation.manuscript import (
    bootstrap_accuracy_ci, bootstrap_label_metric_ci, correctness, overall_mcc, toxin_mask,
)

XTICKS = [10, 30, 50, 100, 300, 1000]
LABEL_COL = {"hbi": METHOD_DARK["hbi"], "nn_combined_run": METHODS["nn_combined_run"][1]}


def _toxin_lengths(preds, lengths):
    tox = preds[toxin_mask(preds)]
    ln = lengths.reindex(tox["identifier"].to_numpy()).to_numpy(dtype=float)
    return ln, correctness(tox).astype(float)


def _binned(preds, lengths):
    """Per-bin toxin-only accuracy + 2 bootstrap SE, at geometric-mean bin centres."""
    ln, correct = _toxin_lengths(preds, lengths)
    centres, acc, se2, ns = [], [], [], []
    for a, b in zip(LEN_BINS[:-1], LEN_BINS[1:]):
        m = (ln >= a) & (ln < b)
        ci = bootstrap_accuracy_ci(correct[m])
        centres.append(float(np.exp(np.log(ln[m]).mean())))
        acc.append(ci["point"])
        se2.append(ci["two_se"])
        ns.append(int(m.sum()))
    return np.array(centres), np.array(acc), np.array(se2), ns


def _panel_mcc(ax):
    """(A) Multiclass (Gorodkin) MCC bars, HBI / emb / emb+tax, +-2 bootstrap SE."""
    for i, k in enumerate(METHOD_ORDER):
        _, color = METHODS[k]
        d = load_preds("test_set", k)
        ci = bootstrap_label_metric_ci(d["actual_label"].values, d["predicted_label"].values,
                                       overall_mcc, n_boot=MCC_CI_N_BOOT)
        ax.bar(i, ci["point"], 0.62, yerr=ci["two_se"], capsize=3, color=color,
               edgecolor="white", linewidth=0.0, error_kw={"elinewidth": 0.8, "capthick": 0.8})
        ax.text(i, ci["point"] + ci["two_se"] + 0.006, fmt_pm(ci["point"], ci["two_se"], sep="\n±"),
                ha="center", va="bottom", fontsize=7.5, linespacing=0.95)
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels([METHODS[k][0].replace(" (", "\n(") for k in METHOD_ORDER], fontsize=7)
    ax.set_ylim(0.75, 0.95)
    ax.set_ylabel("Multiclass MCC")
    ax.set_title("Family-level performance", loc="left", pad=6, fontsize=8.5)
    panel_label(ax, "A")


def _panel_length(ax, hbi, nn, lengths):
    """(B) Toxin-only accuracy by sequence-length bin (HBI vs ToxFam), +-2 bootstrap SE."""
    ax.axvspan(LEN_BINS[0] or 1, 30, color="#ededed", lw=0, zorder=0)
    ax.text(np.sqrt(9 * 30), 1.025, "$<$30 aa", ha="center", va="top", fontsize=6.5, color="#888888")
    series = {}
    for key, d in (("hbi", hbi), ("nn_combined_run", nn)):
        _, color = METHODS[key]
        cx, acc, se2, ns = _binned(d, lengths)
        ax.errorbar(cx, acc, yerr=se2, color=color, marker=METHOD_MARKER[key],
                    ls=METHOD_LINESTYLE[key], lw=1.0, ms=5, capsize=2.5,
                    elinewidth=0.7, capthick=0.7, zorder=3)
        series[key] = (cx, acc, se2, ns)
    cx0 = series["hbi"][0][0]
    ax.annotate("ToxFam", (cx0, series["nn_combined_run"][1][0]),
                xytext=(cx0 * 1.18, series["nn_combined_run"][1][0] + 0.015),
                color=LABEL_COL["nn_combined_run"], fontsize=8, fontweight="bold", va="bottom")
    ax.annotate("HBI", (cx0, series["hbi"][1][0]),
                xytext=(cx0 * 1.18, series["hbi"][1][0] - 0.02),
                color=LABEL_COL["hbi"], fontsize=8, fontweight="bold", va="top")
    ln_all, _ = _toxin_lengths(nn, lengths)
    ax.plot(ln_all, np.full_like(ln_all, 0.32), "|", color="#999999", ms=5, alpha=0.45,
            markeredgewidth=0.5, zorder=2, clip_on=True)
    cx, accn, se2n, ns = series["nn_combined_run"]
    _, acch, se2h, _ = series["hbi"]
    ylow = np.minimum(accn - se2n, acch - se2h)
    for x, yl, n in zip(cx, ylow, ns):
        ax.text(x, yl - 0.03, f"$n$={n}", ha="center", va="top", fontsize=7, color="#666666")
    ax.set_xscale("log")
    ax.set_xlim(9, 1900)
    ax.set_xticks(XTICKS)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _pos: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Sequence length (residues)")
    ax.set_ylabel("Toxin-only accuracy")
    ax.set_ylim(0.30, 1.04)
    ax.set_title("Robustness to sequence length", loc="left", pad=6, fontsize=8.5)
    panel_label(ax, "B")


def _panel_coverage(ax, hbi, nn):
    """(C) ToxFam coverage where HBI returns no hit (HBI = 0% by construction)."""
    nohit_ids = hbi.loc[hbi["predicted_label"] == NO_HIT_LABEL, "identifier"]
    nn_nh = nn[nn["identifier"].isin(nohit_ids)]
    tox_m = toxin_mask(nn_nh)
    groups = [("toxin\nno-hit", nn_nh[tox_m]), ("non-toxin\nno-hit", nn_nh[~tox_m])]
    labels, acc, se2 = [], [], []
    for gname, g in groups:
        ci = bootstrap_accuracy_ci(correctness(g))
        labels.append(f"{gname}\n($n$={len(g)})")
        acc.append(ci["point"])
        se2.append(ci["two_se"])
    x = np.arange(len(groups))
    _, orange = METHODS["nn_combined_run"]
    ax.bar(x, acc, 0.55, yerr=se2, capsize=3, color=orange, edgecolor="white",
           linewidth=0.0, error_kw={"elinewidth": 0.7, "capthick": 0.7})
    for xi, a, s in zip(x, acc, se2):
        ax.text(xi, a + s + 0.015, fmt_pm(a, s), ha="center", va="bottom", fontsize=7.5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Accuracy")
    ax.set_title("No-homolog coverage", loc="left", pad=6, fontsize=8.5)
    panel_label(ax, "C")


def main() -> None:
    apply_style()
    hbi = load_preds("test_set", "hbi")
    nn = load_preds("test_set", "nn_combined_run")
    lengths = sequence_lengths()

    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(DOUBLE_COL, 3.15),
        gridspec_kw={"width_ratios": [1.0, 1.75, 1.05]}, layout="constrained")
    _panel_mcc(axA)
    _panel_length(axB, hbi, nn, lengths)
    _panel_coverage(axC, hbi, nn)
    save_fig(fig, "figure_capability")


if __name__ == "__main__":
    main()
