"""Figure 2 — ToxFam's advantage is concentrated where homology breaks.

Panel A: toxin-only accuracy in fixed length bins (point +- 2 bootstrap SE), HBI
vs ToxFam, with a length rug showing data density; n per bin sits under each point.
Panel B: ToxFam coverage on the proteins where homology returns no hit (HBI = 0%
by construction, shown as an annotated baseline rather than a phantom bar).

Uncertainty is +-2 bootstrap SE throughout (the capability-matrix table convention).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from analysis.figures._common import (
    DOUBLE_COL, LEN_BINS, METHOD_DARK, METHOD_LINESTYLE, METHOD_MARKER, METHODS, apply_style,
    fmt_pm, load_preds, panel_label, save_fig, sequence_lengths,
)
from toxfam.evaluation.manuscript import bootstrap_accuracy_ci, correctness, toxin_mask
from toxfam.evaluation.hbi import NO_HIT_LABEL

# LEN_BINS is imported from _common (shared with numbers_manifest) so the cited
# per-bin accuracies and the plotted bins stay keyed to identical edges.
XTICKS = [10, 30, 50, 100, 300, 1000]
LABEL_COL = {"hbi": METHOD_DARK["hbi"], "nn_combined_run": METHODS["nn_combined_run"][1]}


def _toxin_lengths(preds: pd.DataFrame, lengths: pd.Series):
    tox = preds[toxin_mask(preds)]
    ln = lengths.reindex(tox["identifier"].to_numpy()).to_numpy(dtype=float)
    return ln, correctness(tox).astype(float)


def _binned(preds: pd.DataFrame, lengths: pd.Series):
    """Per-bin toxin-only accuracy + 2 bootstrap SE, with data-driven log centres."""
    ln, correct = _toxin_lengths(preds, lengths)
    centres, acc, se2, ns = [], [], [], []
    for a, b in zip(LEN_BINS[:-1], LEN_BINS[1:]):
        m = (ln >= a) & (ln < b)
        ci = bootstrap_accuracy_ci(correct[m])
        centres.append(float(np.exp(np.log(ln[m]).mean())))  # geometric mean of bin lengths
        acc.append(ci["point"]); se2.append(ci["two_se"]); ns.append(int(m.sum()))
    return np.array(centres), np.array(acc), np.array(se2), ns


def main() -> None:
    apply_style()
    hbi = load_preds("test_set", "hbi")
    nn = load_preds("test_set", "nn_combined_run")
    lengths = sequence_lengths()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.3), layout="constrained")

    # --- Panel A: toxin-only accuracy by length bin, point + 2 bootstrap SE ---
    # very light grey band marks the alignment-hostile <30 aa region; kept much lighter
    # than the HBI marker grey (#BBBBBB) so it reads as a backdrop, not a data series
    axA.axvspan(LEN_BINS[0] or 1, 30, color="#ededed", lw=0, zorder=0)
    axA.text(np.sqrt(9 * 30), 1.025, "$<$30 aa", ha="center", va="top", fontsize=6.5,
             color="#888888")
    series = {}
    for key, d in (("hbi", hbi), ("nn_combined_run", nn)):
        label, color = METHODS[key]
        cx, acc, se2, ns = _binned(d, lengths)
        axA.errorbar(cx, acc, yerr=se2, color=color, marker=METHOD_MARKER[key],
                     ls=METHOD_LINESTYLE[key], lw=1.0, ms=5, capsize=2.5,
                     elinewidth=0.7, capthick=0.7, zorder=3)
        series[key] = (cx, acc, se2, ns)

    # Direct labels at the leftmost bin, where the two methods are most separated.
    cx0 = series["hbi"][0][0]
    axA.annotate("ToxFam", (cx0, series["nn_combined_run"][1][0]),
                 xytext=(cx0 * 1.18, series["nn_combined_run"][1][0] + 0.015),
                 color=LABEL_COL["nn_combined_run"], fontsize=8, fontweight="bold", va="bottom")
    axA.annotate("HBI", (cx0, series["hbi"][1][0]),
                 xytext=(cx0 * 1.18, series["hbi"][1][0] - 0.02),
                 color=LABEL_COL["hbi"], fontsize=8, fontweight="bold", va="top")

    # length rug (data density) along the bottom; n per bin just under the lower point
    ln_all, _ = _toxin_lengths(nn, lengths)
    axA.plot(ln_all, np.full_like(ln_all, 0.32), "|", color="#999999", ms=5, alpha=0.45,
             markeredgewidth=0.5, zorder=2, clip_on=True)
    cx, accn, se2n, ns = series["nn_combined_run"]
    _, acch, se2h, _ = series["hbi"]
    ylow = np.minimum(accn - se2n, acch - se2h)
    for x, yl, n in zip(cx, ylow, ns):
        axA.text(x, yl - 0.03, f"$n$={n}", ha="center", va="top", fontsize=7, color="#666666")

    axA.set_xscale("log")
    axA.set_xlim(9, 1900)
    axA.set_xticks(XTICKS)
    axA.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _pos: f"{v:g}"))
    axA.xaxis.set_minor_formatter(mticker.NullFormatter())
    axA.set_xlabel("Sequence length (residues)")
    axA.set_ylabel("Toxin-only accuracy")
    axA.set_ylim(0.30, 1.04)
    axA.set_title("Robustness to sequence length", loc="left", pad=8)
    panel_label(axA, "A")

    # --- Panel B: ToxFam coverage where HBI returns no hit (HBI = 0% by construction) ---
    nohit_ids = hbi.loc[hbi["predicted_label"] == NO_HIT_LABEL, "identifier"]
    nn_nh = nn[nn["identifier"].isin(nohit_ids)]
    tox_m = toxin_mask(nn_nh)
    groups = [("toxin\nno-hit", nn_nh[tox_m]), ("non-toxin\nno-hit", nn_nh[~tox_m])]
    labels, acc, se2 = [], [], []
    for gname, g in groups:
        ci = bootstrap_accuracy_ci(correctness(g))
        labels.append(f"{gname}\n($n$={len(g)})")
        acc.append(ci["point"]); se2.append(ci["two_se"])
    x = np.arange(len(groups))
    _, orange = METHODS["nn_combined_run"]
    axB.bar(x, acc, 0.55, yerr=se2, capsize=3, color=orange, edgecolor="white",
            linewidth=0.0, error_kw={"elinewidth": 0.7, "capthick": 0.7})
    for xi, a, s in zip(x, acc, se2):  # value ±2 SE clear above the error cap
        axB.text(xi, a + s + 0.015, fmt_pm(a, s), ha="center", va="bottom", fontsize=7.5)
    # HBI is 0% on these proteins by construction (it found no hit); the descriptive
    # title carries that, so no baseline line or tag is drawn.
    axB.set_xlim(-0.6, 1.6)
    axB.set_xticks(x)
    axB.set_xticklabels(labels)
    axB.set_ylim(0, 1.14)
    axB.set_ylabel("Accuracy")
    axB.set_title("ToxFam coverage where HBI finds no hit", loc="left", pad=8)
    panel_label(axB, "B")

    save_fig(fig, "figure2_homology")


if __name__ == "__main__":
    main()
