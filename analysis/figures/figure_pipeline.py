"""Dataset pipeline figure (Fig. 1) --- how the ToxFam training set is built.

Replaces the earlier hand-drawn TikZ attrition flow (a tall vertical stack of
identical rounded boxes) with a compact, proportional, two-stream funnel that
reads left-to-right. Design follows a verified figure-methodology review
(2026-07-04) and this project's house style:

  * Bars encode magnitude by LENGTH (perception judges length >> area), not an
    area/width funnel -- each per-step drop is legible at a glance.
  * Two mirrored lanes -- toxins grow up (amber, the focus, carrying the family
    vocabulary story), non-toxins grow down (grey). Each lane is scaled on its
    own axis so the ~5% toxin minority stays readable; a "true class balance"
    reference bar keeps the real imbalance honest.
  * CONSORT/PRISMA grammar -- exclusions annotated inside the tapering flow with
    count + percent-of-parent; in-place transforms (label normalisation, SignalP6
    trimming, which change labels/residues but not counts) sit in the centre gap.
  * Okabe-Ito colour-blind-safe palette; streams are labelled, never colour-only.

The counts are the frozen UniProt-snapshot numbers; they mirror
``manuscript/dataset_numbers.tex`` (the LaTeX single source of truth shared with
Supplementary Table ``tab:dataset_pipeline``). The manuscript lives in a separate
(git-ignored) repository, so the values are duplicated here with matching content;
keep the two in sync when the snapshot changes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from analysis.figures._common import DOUBLE_COL, apply_style, save_fig

# --- Okabe-Ito semantic colours (same hex as the method palette in _common) ---
TOX = "#E69F00"      # amber = toxins (the focus)
NON = "#BBBBBB"      # grey  = non-toxins (recessive majority)
TOX_DK = "#a5670a"   # darker amber for text on white
NON_DK = "#6f6f6f"
DROP = "#B0455A"     # muted red = removed (never paired with green)
OUT = "#4d6472"      # neutral slate = final ML-ready splits (not a method colour)
INK = "#1f1f1f"
MUTE = "#5f5f5f"

# Frozen pipeline counts (mirror manuscript/dataset_numbers.tex).
C = {
    "RawTox": 5927, "RawNontox": 99846,
    "DropNoFam": 360, "FamTox": 5567,
    "DropLen": 996, "LenNontox": 98850,
    "RepTox": 3416, "RepNontox": 61763, "RepTotal": 65179,
    "SplitTrain": 45621, "SplitVal": 9779, "SplitTest": 9779,
}


def _build() -> plt.Figure:
    # Four count-changing checkpoints (label-only transforms are annotations, not bars).
    levels = [
        ("UniProtKB/\nSwiss-Prot query", C["RawTox"], C["RawNontox"], "572 strings"),
        ("Family-\nannotated",           C["FamTox"], C["RawNontox"], None),
        ("Length-filtered\n& mature",    C["FamTox"], C["LenNontox"], "45 families"),
        ("Non-redundant\nreps (90% id)", C["RepTox"], C["RepNontox"], "37 → K=38"),
    ]
    tox_excl = [(0, C["DropNoFam"], "no family"), (2, C["FamTox"] - C["RepTox"], "clustered")]
    non_excl = [(1, C["DropLen"], "longest"), (2, C["LenNontox"] - C["RepNontox"], "clustered")]
    transforms = [(1, 2, "normalise\n572→45"), (2, 3, "SignalP6\ntrim")]
    splits = [("Train", C["SplitTrain"]), ("Val", C["SplitVal"]), ("Test", C["SplitTest"])]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.2))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # full-bleed (save_fig has no tight bbox)
    ax.set_xlim(0, 112); ax.set_ylim(-23.5, 26.5); ax.axis("off")

    xs = [11, 31, 51, 71]; BARW = 12; GAP = 3.0
    s_tox = 14.5 / levels[0][1]
    s_non = 14.5 / levels[0][2]
    ht = lambda v: v * s_tox
    hn = lambda v: v * s_non

    # continuity flows between bars (retained portion; the taper shows where volume is lost)
    for i in range(len(levels) - 1):
        x0, x1 = xs[i] + BARW / 2, xs[i + 1] - BARW / 2
        ax.fill([x0, x1, x1, x0], [GAP, GAP, GAP + ht(levels[i + 1][1]), GAP + ht(levels[i][1])],
                color=TOX, alpha=0.16, lw=0)
        ax.fill([x0, x1, x1, x0], [-GAP, -GAP, -GAP - hn(levels[i + 1][2]), -GAP - hn(levels[i][2])],
                color=NON, alpha=0.16, lw=0)

    for i, (lab, tox, non, vocab) in enumerate(levels):
        x = xs[i]
        ax.add_patch(plt.Rectangle((x - BARW / 2, GAP), BARW, ht(tox), fc=TOX, ec="white", lw=0.8))
        ax.text(x, GAP + ht(tox) / 2, f"{tox:,}", ha="center", va="center",
                fontsize=7.5, color="white", weight="bold")
        ax.add_patch(plt.Rectangle((x - BARW / 2, -GAP - hn(non)), BARW, hn(non), fc=NON, ec="white", lw=0.8))
        ax.text(x, -GAP - hn(non) / 2, f"{non:,}", ha="center", va="center",
                fontsize=7.5, color="white", weight="bold")
        ax.text(x, GAP + ht(tox) + 5.0, lab, ha="center", va="bottom",
                fontsize=7.5, weight="bold", color=INK, linespacing=1.05)
        if vocab:
            ax.text(x, GAP + ht(tox) + 1.4, vocab, ha="center", va="bottom",
                    fontsize=7.0, color=TOX_DK, weight="bold")

    ax.text(2.5, GAP + 7.2, "TOXINS", ha="center", va="center", fontsize=7.5, weight="bold",
            color=TOX_DK, rotation=90)
    ax.text(2.5, -GAP - 7.2, "NON-TOXINS", ha="center", va="center", fontsize=7.5, weight="bold",
            color=NON_DK, rotation=90)

    # exclusions inside the tapering flow band: count + percent-of-parent
    for (i, cnt, why) in tox_excl:
        xm = (xs[i] + xs[i + 1]) / 2
        pct = 100 * cnt / levels[i][1]
        ax.text(xm, GAP + ht(levels[i + 1][1]) / 2, f"−{cnt:,}\n{why}\n{pct:.0f}%",
                ha="center", va="center", fontsize=6.7, color=DROP, weight="bold", linespacing=1.0)
    for (i, cnt, why) in non_excl:
        xm = (xs[i] + xs[i + 1]) / 2
        pct = 100 * cnt / levels[i][2]
        ax.text(xm, -GAP - hn(levels[i + 1][2]) / 2, f"−{cnt:,}\n{why}\n{pct:.0f}%",
                ha="center", va="center", fontsize=6.7, color=DROP, weight="bold", linespacing=1.0)

    # in-place transforms (italic, centre gap)
    for (a, b, txt) in transforms:
        ax.text((xs[a] + xs[b]) / 2, 0, txt, ha="center", va="center",
                fontsize=6.8, color=MUTE, style="italic", linespacing=1.0)

    # split fan-out (neutral slate; segments proportional to split size)
    xr = 86; rep = levels[-1]; tot = C["RepTotal"]
    top = GAP + ht(rep[1]); bot = -GAP - hn(rep[2]); span = top - bot
    ax.text(xr + BARW / 2, top + 5.0, "70:15:15\nstratified", ha="center", va="bottom",
            fontsize=7.5, weight="bold", color=INK, linespacing=1.05)
    yy = bot
    for name, v in splits[::-1]:
        seg = span * v / tot
        ax.add_patch(plt.Rectangle((xr, yy), BARW, seg, fc=OUT, ec="white", lw=0.8,
                                   alpha=0.55 + 0.30 * (name == "Train")))
        ax.text(xr + BARW + 1.6, yy + seg / 2, f"{name} {v:,}", ha="left", va="center",
                fontsize=7.0, color=INK)
        yy += seg
    ax.annotate("", xy=(xr - 1.2, 0), xytext=(xs[-1] + BARW / 2 + 1.2, 0),
                arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.1))

    # true class-balance reference (honest anchor for the independently scaled lanes)
    tot_raw = levels[0][1] + levels[0][2]
    px, pw, py = 33, 32, -20.8
    ax.text(px - 1.5, py + 0.75, "true class balance", fontsize=6.8, color=MUTE, va="center", ha="right")
    ax.add_patch(plt.Rectangle((px, py), pw * levels[0][1] / tot_raw, 1.5, fc=TOX, ec="none"))
    ax.add_patch(plt.Rectangle((px + pw * levels[0][1] / tot_raw, py), pw * levels[0][2] / tot_raw, 1.5,
                               fc=NON, ec="none"))
    ax.text(px + pw + 1.6, py + 0.75,
            f"{100*levels[0][1]/tot_raw:.1f}% toxin  /  {100*levels[0][2]/tot_raw:.1f}% non-toxin",
            fontsize=6.8, color=MUTE, va="center")
    return fig


def main() -> None:
    apply_style()
    fig = _build()
    save_fig(fig, "figure_pipeline")


if __name__ == "__main__":
    main()
