"""Capability figure --- ToxFam's family-level advantage over homology, and where
it concentrates.

Merges the former Fig. 1 (multiclass-MCC bars) and Fig. 2 (homology break-down)
into one figure so the whole HBI comparison reads as a single message:
  (A) multiclass (Gorodkin) MCC for HBI, ToxFam (emb) and ToxFam (emb+tax) --- the
      imbalance-robust headline metric, +-2 bootstrap SE;
  (B) toxin-only accuracy across sequence length as a boundary-corrected local-linear
      curve (+-2 bootstrap SE band): HBI degrades progressively on the shortest toxins
      while ToxFam stays flat. A top-marginal histogram shows the length distribution
      so the reader can weigh where the toxin population actually sits;
  (C) ToxFam coverage on the proteins where HBI returns no hit (HBI = 0% there by
      construction).

Interpretable-but-ranking-redundant accuracy views (toxin-only, all-class) live in
Supplementary Fig. S1 so the main figure shows only what is needed.

Panel B is a continuous local-linear (LOESS degree-1) regression of correctness on
log-length rather than coarse bins: it corrects the boundary bias that a plain kernel
average suffers at the short end, and so faithfully renders the *graded* homology
collapse (~0.9 down to ~25 aa, then falling to ~0.4 on the shortest toxins) that a
single <30 aa bin would hide.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from paper.figures._common import (
    DOUBLE_COL, MCC_CI_N_BOOT, METHOD_DARK, METHOD_LINESTYLE, METHOD_ORDER, METHODS,
    apply_style, fmt_pm, load_preds, panel_label, save_fig, sequence_lengths,
)
from toxfam.evaluation.hbi import NO_HIT_LABEL
from paper.stats import (
    band_separation_length, bootstrap_accuracy_ci, bootstrap_label_metric_ci, correctness,
    length_support_mask, local_linear_accuracy, local_linear_band, overall_mcc, toxin_mask,
)

XTICKS = [10, 30, 50, 100, 300, 1000]
XLIM = (9, 1900)
BW = 0.16                       # local-linear bandwidth in log10 length (tuned to the data)
HIST_GREY = "#d9d9d9"
GREY_D, ORANGE_D = METHOD_DARK["hbi"], METHOD_DARK["nn_combined_run"]


def _toxin_lengths(preds, lengths):
    """Toxin sequence lengths paired with per-row correctness, in the same order.

    Toxins whose identifier has no known length are dropped from both arrays so a
    missing length cannot poison the log-scale grid or the local-linear fit.
    """
    tox = preds[toxin_mask(preds)]
    ln = lengths.reindex(tox["identifier"].to_numpy()).to_numpy(dtype=float)
    corr = correctness(tox).astype(float)
    ok = np.isfinite(ln)
    return ln[ok], corr[ok]


def _logx(ax):
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _pos: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Sequence length (residues)")


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


def _panel_length(ax, axtop, hbi, nn, lengths, rng):
    """(B) Toxin-only accuracy vs length (local-linear +-2 SE) with a length histogram."""
    lnH, corrH = _toxin_lengths(hbi, lengths)
    lnN, corrN = _toxin_lengths(nn, lengths)

    # --- top marginal: length distribution (own count axis; not overlaid on accuracy) ---
    edges = np.logspace(np.log10(lnH.min()), np.log10(lnH.max()), 24)
    counts, _, _ = axtop.hist(lnH, bins=edges, color=HIST_GREY, edgecolor="white", linewidth=0.3)
    peak = int(counts.max())
    axtop.set_xscale("log")
    axtop.set_xlim(*XLIM)
    axtop.set_ylim(0, peak * 1.18)
    axtop.set_yticks([0, peak])
    axtop.tick_params(axis="y", labelsize=6, colors="#999999", length=2)
    axtop.tick_params(axis="x", labelbottom=False, length=0)
    axtop.set_ylabel("toxins", fontsize=6.5, color="#999999", rotation=0, ha="right", va="center")
    for sp in ("top", "right"):
        axtop.spines[sp].set_visible(False)

    # --- accuracy curves: each method pairs its OWN lengths with its OWN correctness
    #     (the hbi and nn frames need not share row order), evaluated on a shared grid. ---
    grid = np.logspace(np.log10(lnH.min()), np.log10(np.percentile(lnH, 98)), 160)
    keep = length_support_mask(lnH, grid, bandwidth=BW)
    gk = grid[keep]
    series = {}
    for key, ln_m, corr_m, dark in (("hbi", lnH, corrH, GREY_D),
                                    ("nn_combined_run", lnN, corrN, ORANGE_D)):
        y = local_linear_accuracy(ln_m, corr_m, grid, bandwidth=BW)[keep]
        s = local_linear_band(ln_m, corr_m, grid, bandwidth=BW, rng=rng)[keep]
        ax.fill_between(gk, y - s, y + s, color=METHODS[key][1], alpha=0.20, lw=0, zorder=2)
        ax.plot(gk, y, color=dark, ls=METHOD_LINESTYLE[key], lw=1.7, zorder=3)
        series[key] = (y, s)
    yN, sN = series["nn_combined_run"]
    yH, sH = series["hbi"]

    # Significance boundary: the length below which the two +-2 SE bands stop overlapping
    # (ToxFam pointwise significantly more accurate than HBI); taken from the plotted
    # bands, so the guide sits exactly where they visibly separate.
    xcross = band_separation_length(gk, yH + sH, yN - sN)
    if xcross is not None:
        ax.axvline(xcross, color="#9a9a9a", ls=(0, (1, 1.6)), lw=0.8, zorder=1)
        ax.text(xcross * 1.07, 0.30, f"$\\approx${xcross:.0f} aa", fontsize=7,
                color="#8a8a8a", ha="left", va="bottom")

    # direct end-labels in the empty right region (data ends ~400 aa, axis runs to 1900)
    xr = gk[-1] * 1.25
    ax.text(xr, yN[-1] + 0.005, "ToxFam", color=ORANGE_D, fontsize=8,
            fontweight="bold", ha="left", va="center")
    ax.text(xr, yH[-1] - 0.005, "HBI", color=GREY_D, fontsize=8,
            fontweight="bold", ha="left", va="center")
    ax.annotate("homology degrades\non the shortest toxins", xy=(11, 0.45), xytext=(70, 0.58),
                fontsize=6.6, color="#777777", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.7,
                                connectionstyle="arc3,rad=-0.15"))
    _logx(ax)
    ax.set_ylim(0.28, 1.04)
    ax.set_ylabel("Toxin-only accuracy")


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


def main() -> None:
    apply_style()
    hbi = load_preds("test_set", "hbi")
    nn = load_preds("test_set", "nn_combined_run")
    lengths = sequence_lengths()
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(DOUBLE_COL, 3.35), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 6.2],
                          width_ratios=[1.0, 1.75, 1.05], hspace=0.05)
    # Every column has a top-row axes carrying its header, so all three titles and panel
    # letters align by construction. Only column B's top row is a visible marginal (the
    # length histogram); A and C use blank spacers.
    axAtop = fig.add_subplot(gs[0, 0])
    axBtop = fig.add_subplot(gs[0, 1])
    axCtop = fig.add_subplot(gs[0, 2])
    axAtop.axis("off")
    axCtop.axis("off")
    axA = fig.add_subplot(gs[1, 0])
    axB = fig.add_subplot(gs[1, 1], sharex=axBtop)
    axC = fig.add_subplot(gs[1, 2])

    _panel_mcc(axA)
    _panel_length(axB, axBtop, hbi, nn, lengths, rng)
    _panel_coverage(axC, hbi, nn)

    for axtop, letter, title in ((axAtop, "A", "Family-level performance"),
                                 (axBtop, "B", "Robustness to sequence length"),
                                 (axCtop, "C", "No-homolog coverage")):
        axtop.set_title(title, loc="left", pad=4, fontsize=8.5)
        panel_label(axtop, letter)
    save_fig(fig, "figure_capability")


if __name__ == "__main__":
    main()
