"""Dataset pipeline figure (Fig. 1) --- how the ToxFam training set is built.

A compact, proportional, two-stream funnel that reads left-to-right: one
UniProtKB/Swiss-Prot query forks (on the KW-0800 toxin keyword) into a toxin
lane (amber, up) and a non-toxin lane (grey, down), each distilled through the
same three count-changing checkpoints to the final stratified splits.

Design decisions (house style + figure-methodology research, 2026-07):

  * The structure follows the ACTUAL code in ``toxfam.data.preprocessing`` (which stream
    each step touches, and the counts), staged by PURPOSE -- cleaning vs deduplication --
    rather than by strict execution order. Concretely:
      - ``load_and_prepare_raw`` drops family-less toxins AND normalises the toxin
        family vocabulary AND drops the longest 1% of non-toxins -- all at load, so
        these collapse into ONE "Curated" checkpoint (not spread across stages).
      - the longest-1% length cut is applied to NON-TOXINS ONLY (the longest toxin,
        1652 aa, is below the 2243 aa threshold), so the toxin lane shows no length
        drop -- only the missing-family drop.
      - SignalP6 trims residues but removes no sequences, so it gets no bar and is grouped
        into the "Curated" (cleaned + mature) stage; MMseqs2 clustering is what removes the
        "redundant" reps. Grey italic centre labels NAME the tool at each step (SignalP6
        count-neutral, MMseqs2 driving the red drops); red labels are the only count removals.
      - the toxin family vocabulary is a LABEL count (families + an "other" catch-all), so
        the track reads "45 labels"/"37 labels", matching the Methods 44+other / 36+other.
  * Bars encode magnitude by LENGTH (perception judges length >> area). Each lane is
    scaled on its own axis so the ~6% toxin minority stays legible; a "true class
    balance" reference bar keeps the real imbalance honest.
  * The toxin family label vocabulary is a flat top track (572 raw -> 45 -> 37),
    since it is a toxin-only concept (non-toxins are a single class); K=38 = 37 toxin
    labels + the non-toxin class is called out at the output.
  * A compact form of the literal query seeds the figure on the left; the full
    verbatim query lives in the caption/README (PRISMA-S/FAIR), keeping the figure
    uncluttered. Okabe-Ito colour-blind-safe palette; lanes are labelled, never
    colour-only.

The counts are the frozen UniProt-snapshot numbers; they mirror
``manuscript/dataset_numbers.tex`` (the LaTeX single source of truth shared with
Supplementary Table ``tab:dataset_pipeline``). The manuscript lives in a separate
(git-ignored) repository, so the values are duplicated here with matching content;
keep the two in sync when the snapshot changes.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paper.figures._common import DOUBLE_COL, apply_style, save_fig

# --- Okabe-Ito semantic colours (same hex as the method palette in _common) ---
TOX = "#E69F00"      # amber = toxins (the focus)
NON = "#BBBBBB"      # grey  = non-toxins (recessive majority)
TOX_DK = "#a5670a"   # darker amber for text on white
NON_DK = "#6f6f6f"
DROP = "#B0455A"     # muted red = removed (never paired with green)
OUT = "#4d6472"      # neutral slate = final ML-ready splits (not a method colour)
INK = "#1f1f1f"
MUTE = "#5f5f5f"
FAINT = "#9a9a9a"    # faint track/gloss notes
MONO = {"family": "monospace"}

# Frozen pipeline counts (mirror manuscript/dataset_numbers.tex).
C = {
    "RawTox": 5927, "RawNontox": 99846,
    "FamTox": 5567, "LenNontox": 98850,
    "RepTox": 3416, "RepNontox": 61763, "RepTotal": 65179,
    "SplitTrain": 45621, "SplitVal": 9779, "SplitTest": 9779,
}


def _build() -> plt.Figure:
    # Three count-changing checkpoints per lane; (header, toxin, non-toxin).
    levels = [
        ("Retrieved",     C["RawTox"], C["RawNontox"]),
        ("Curated",       C["FamTox"], C["LenNontox"]),
        ("Non-redundant", C["RepTox"], C["RepNontox"]),
    ]
    splits = [("Test", C["SplitTest"]), ("Val", C["SplitVal"]), ("Train", C["SplitTrain"])]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.4))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # full-bleed (save_fig has no tight bbox)
    ax.set_xlim(0, 124); ax.set_ylim(-23, 26); ax.axis("off")

    xs = [34, 58, 82]; BARW = 12; GAP = 3.4
    s_tox = 13.0 / levels[0][1]
    s_non = 13.0 / levels[0][2]
    ht = lambda v: v * s_tox
    hn = lambda v: v * s_non
    HEAD_TOP = GAP + 13.0 + 8.0   # common header TOP line; va="top" aligns all first lines
    VOCAB_Y = GAP + 13.0 + 2.0    # flat family-vocabulary track (reads 572 -> 45 -> 37)

    # --- query source box (left), forking into the two keyword-split streams ---
    ax.add_patch(FancyBboxPatch((1.5, -4.0), 17.5, 9.0, boxstyle="round,pad=0.2,rounding_size=0.7",
                                fc="#f6f4ef", ec="#d9d2c4", lw=0.7))
    ax.text(10.2, 3.3, "UniProtKB query", fontsize=6.9, weight="bold", color=INK, ha="center", va="center")
    for j, t in enumerate(["taxonomy:Metazoa", "AND reviewed:true", "AND fragment:false"]):
        ax.text(2.6, 1.3 - j * 1.9, t, fontsize=5.9, color="#444444", va="center", **MONO)

    # clean straight fork: one query splits on the KW-0800 keyword into two streams
    fork = (19.3, 0.5)
    tox_end = (xs[0] - BARW / 2 - 0.6, GAP + ht(levels[0][1]) / 2)
    non_end = (xs[0] - BARW / 2 - 0.6, -GAP - hn(levels[0][2]) / 2)
    ax.add_patch(FancyArrowPatch(fork, tox_end, arrowstyle="-|>", mutation_scale=10, lw=1.2, color=TOX_DK))
    ax.add_patch(FancyArrowPatch(fork, non_end, arrowstyle="-|>", mutation_scale=10, lw=1.2, color=NON_DK))
    # keyword split rides each arrow (rotated to its on-screen slope) so the KW-0800 rule
    # reads on the branch it makes. Angle uses the data->inch aspect of this axes.
    xin = DOUBLE_COL / (124 - 0)
    yin = 3.4 / (26 - (-23))
    ang = lambda p0, p1: math.degrees(math.atan2((p1[1] - p0[1]) * yin, (p1[0] - p0[0]) * xin))
    ax.text(22.6, 5.5, "AND KW-0800", fontsize=5.8, color=TOX_DK, ha="center", va="center",
            rotation=ang(fork, tox_end), rotation_mode="anchor", **MONO)
    ax.text(22.6, -5.0, "NOT KW-0800", fontsize=5.8, color=NON_DK, ha="center", va="center",
            rotation=ang(fork, non_end), rotation_mode="anchor", **MONO)
    ax.text(27.2, 12.6, "Toxins", fontsize=7.3, weight="bold", color=TOX_DK, ha="right", va="center")
    ax.text(27.2, -12.6, "Non-toxins", fontsize=7.3, weight="bold", color=NON_DK, ha="right", va="center")

    # --- continuity flows (retained portion; the taper shows where volume is lost) ---
    for i in range(len(levels) - 1):
        x0, x1 = xs[i] + BARW / 2, xs[i + 1] - BARW / 2
        ax.fill([x0, x1, x1, x0], [GAP, GAP, GAP + ht(levels[i + 1][1]), GAP + ht(levels[i][1])],
                color=TOX, alpha=0.16, lw=0)
        ax.fill([x0, x1, x1, x0], [-GAP, -GAP, -GAP - hn(levels[i + 1][2]), -GAP - hn(levels[i][2])],
                color=NON, alpha=0.16, lw=0)

    # --- bars + aligned headers ---
    for i, (lab, tox, non) in enumerate(levels):
        x = xs[i]
        ax.add_patch(plt.Rectangle((x - BARW / 2, GAP), BARW, ht(tox), fc=TOX, ec="white", lw=0.8))
        ax.text(x, GAP + ht(tox) / 2, f"{tox:,}", ha="center", va="center", fontsize=7.4,
                color="white", weight="bold")
        ax.add_patch(plt.Rectangle((x - BARW / 2, -GAP - hn(non)), BARW, hn(non), fc=NON, ec="white", lw=0.8))
        ax.text(x, -GAP - hn(non) / 2, f"{non:,}", ha="center", va="center", fontsize=7.4,
                color="white", weight="bold")
        ax.text(x, HEAD_TOP, lab, ha="center", va="top", fontsize=7.5, weight="bold", color=INK)

    # --- flat toxin family-vocabulary track (a progression, not per-bar clutter) ---
    ax.text(xs[0] - BARW / 2 - 0.5, VOCAB_Y, "toxin family labels", fontsize=5.6, color=TOX_DK,
            style="italic", ha="right", va="center")
    vocab_txt = ["572 raw", "45 labels", "37 labels"]
    for i, vt in enumerate(vocab_txt):
        ax.text(xs[i], VOCAB_Y, vt, ha="center", va="center", fontsize=6.8, color=TOX_DK, weight="bold")
        if i < len(vocab_txt) - 1:
            ax.annotate("", xy=(xs[i + 1] - BARW / 2 - 0.8, VOCAB_Y), xytext=(xs[i] + BARW / 2 + 0.8, VOCAB_Y),
                        arrowprops=dict(arrowstyle="-|>", color=TOX, lw=0.9))

    # --- transitions: per-lane removals (red) + shared centre transforms (grey italic) ---
    # each removal is vertically centred within its lane's tapering flow band at the gap
    def drop(xm, yc, cnt, reason):
        ax.text(xm, yc + 1.0, f"−{cnt:,}", ha="center", va="center", fontsize=7.0, color=DROP, weight="bold")
        ax.text(xm, yc - 1.0, reason, ha="center", va="center", fontsize=6.0, color=DROP, linespacing=0.95)

    def transform(xm, txt):
        ax.text(xm, 0, txt, ha="center", va="center", fontsize=6.2, color=MUTE, style="italic", linespacing=1.0)

    def tox_mid(i):   # centre of the toxin flow band at gap i -> i+1
        return GAP + (ht(levels[i][1]) + ht(levels[i + 1][1])) / 4
    def non_mid(i):
        return -GAP - (hn(levels[i][2]) + hn(levels[i + 1][2])) / 4

    xm0 = (xs[0] + xs[1]) / 2
    xm1 = (xs[1] + xs[2]) / 2
    drop(xm0, tox_mid(0), C["RawTox"] - C["FamTox"], "no family\nlabel")
    drop(xm0, non_mid(0), C["RawNontox"] - C["LenNontox"], "longest 1%\n(>2,243 aa)")
    transform(xm0, "SignalP6\nsignal-peptide trim")
    drop(xm1, tox_mid(1), C["FamTox"] - C["RepTox"], "redundant")
    drop(xm1, non_mid(1), C["LenNontox"] - C["RepNontox"], "redundant")
    transform(xm1, "MMseqs2\n90% id, keep reps")

    # --- split fan-out (neutral slate; proportional segments) ---
    xr = 97; rep = levels[-1]; tot = C["RepTotal"]
    top = GAP + ht(rep[1]); bot = -GAP - hn(rep[2]); span = top - bot
    ax.text(xr + BARW / 2, HEAD_TOP, "70:15:15\nstratified", ha="center", va="top", fontsize=7.5,
            weight="bold", color=INK, linespacing=1.0)
    yy = bot
    for name, v in splits:
        seg = span * v / tot
        ax.add_patch(plt.Rectangle((xr, yy), BARW, seg, fc=OUT, ec="white", lw=0.8,
                                   alpha=0.55 + 0.30 * (name == "Train")))
        ax.text(xr + BARW + 1.4, yy + seg / 2, f"{name} {v:,}", ha="left", va="center", fontsize=7.0, color=INK)
        yy += seg
    ax.annotate("", xy=(xr - 1.2, 0), xytext=(xs[-1] + BARW / 2 + 1.2, 0),
                arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.1))
    ax.text(xr + BARW / 2, top + 1.4, "K = 38 classes\n37 toxin + non-toxin", ha="center", va="bottom",
            fontsize=6.4, color=OUT, weight="bold", linespacing=1.0)

    # --- true class-balance reference (honest anchor for the independently scaled lanes) ---
    tot_raw = levels[0][1] + levels[0][2]
    px, pw, py = 40, 30, -20.8
    ax.text(px - 1.5, py + 0.7, "true class balance", fontsize=6.2, color=MUTE, va="center", ha="right")
    ax.add_patch(plt.Rectangle((px, py), pw * levels[0][1] / tot_raw, 1.4, fc=TOX, ec="none"))
    ax.add_patch(plt.Rectangle((px + pw * levels[0][1] / tot_raw, py), pw * levels[0][2] / tot_raw, 1.4,
                               fc=NON, ec="none"))
    ax.text(px + pw + 1.6, py + 0.7,
            f"{100 * levels[0][1] / tot_raw:.1f}% toxin  /  {100 * levels[0][2] / tot_raw:.1f}% non-toxin",
            fontsize=6.2, color=MUTE, va="center")
    return fig


def main() -> None:
    apply_style()
    fig = _build()
    save_fig(fig, "figure_pipeline")


if __name__ == "__main__":
    main()
