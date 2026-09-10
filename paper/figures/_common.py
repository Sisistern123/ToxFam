"""Shared loaders and matplotlib style for manuscript figures.

Style and palette follow verified Bioinformatics (OUP) figure guidelines and
colour-blind-safe palette research (see
docs/superpowers/specs/2026-06-30-figure-overhaul-design.md):

* Build at final column width (double = 178 mm = 7.008 in) -- never draw large and
  let the journal shrink it (that is what made earlier text illegible). See apply_style()
  for why the preprint class still downscales these by ~3.5%.
* Arial, white opaque background, 0.5 pt spines, fonts embedded as TrueType.
* Okabe-Ito method palette (grey/blue/orange) + Paul Tol high-contrast
  adjudication ramp (blue/amber/red, luminance-ordered, greyscale-safe).
"""

from __future__ import annotations

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

from paper._paths import figures_output_dir, model_run_dir
from toxfam._paths import (
    benchmark_dir,
    evaluation_data_dir,
    processed_dir,
)

console = Console()

# Figure PDFs/PNGs + results_numbers.{json,tex} are written here (paper/figures/output).
# Created lazily by save_fig / the manifest writer, not at import time.
FIG_DIR = figures_output_dir()

# --- Bioinformatics (OUP) column widths, inches (verified: 86 mm / 178 mm) ---
SINGLE_COL = 86 / 25.4  # 3.386 in
DOUBLE_COL = 178 / 25.4  # 7.008 in

# Two-line panel titles do not fit at SINGLE_COL in the 9 pt axes.titlesize the style
# sets, so single-column figures step down to this. Named here so the next single-column
# figure does not rediscover the number, and so it moves with the rest of the type scale.
TITLE_FS_COMPACT = 8

# Consistent, colour-blind-safe method colours/labels across all figures.
# Okabe-Ito blue/orange is the most CVD-robust contrast pair and is greyscale
# distinguishable; grey pushes the homology baseline visually behind the models.
METHODS = {
    "hbi": ("HBI", "#BBBBBB"),
    "nn_standard_run": ("ToxFam (emb)", "#0072B2"),
    "nn_combined_run": ("ToxFam (emb+tax)", "#E69F00"),
}
# Redundant (non-colour) encoding so series survive total colour loss.
METHOD_MARKER = {"hbi": "o", "nn_standard_run": "^", "nn_combined_run": "s"}
METHOD_LINESTYLE = {
    "hbi": (0, (5, 2)),
    "nn_standard_run": (0, (1, 1)),
    "nn_combined_run": "-",
}
# Canonical method order (the METHODS insertion order). Single source of truth so the
# figure scripts never re-hardcode the key list and drift from the palette.
METHOD_ORDER = list(METHODS)
# Hand-tuned darker variants of the method colours, for text labels and marker edges
# where the pale canonical fill needs more contrast. Kept beside METHODS so the shade
# and its base colour live in one place (used by figure2 labels + figure3 edges).
METHOD_DARK = {"hbi": "#6f6f6f", "nn_combined_run": "#b06a00"}

# Toxin / non-toxin CLASS colours, for data-side figures (the pipeline figure, the
# preprocessing audit). A separate namespace from METHODS -- and, since 2026-09-10, a
# DISJOINT one. It used to reuse #E69F00 and #0072B2 with a comment saying the reader must
# not confuse the two meanings; in practice they do, because every performance figure
# trains them to read amber as ToxFam and grey as HBI, and then Fig. 1 asks them to read
# amber as "toxin" and grey as "non-toxin". So: no hex appears in both dicts.
#
# Green (Okabe-Ito bluish green) is the toxin lane, slate the non-toxin majority. Both are
# clear of METHODS (#BBBBBB / #0072B2 / #E69F00) and of the removal red (#B0455A) that
# labels sit in on top of them, and they separate in greyscale by lightness.
CLASSES = {
    "toxin": "#009E73",
    "toxin_dark": "#00654a",
    "nontoxin": "#9DB4C0",
    "nontoxin_dark": "#5b7280",
    "neutral": "#BBBBBB",
    "accent": "#CC79A7",
}

# Ordered good->bad adjudication ramp (Paul Tol high-contrast). NEVER green=good/
# red=bad (the exact deuteranopia failure case); this ramp is luminance-ordered so
# it reads as good->bad even in greyscale.
ADJUDICATION = {"correct": "#004488", "partial": "#DDAA33", "incorrect": "#BB5566"}

# Toxin-only sequence-length bins, shared by figure2 and numbers_manifest so the
# plotted per-bin accuracies and the cited numbers stay keyed to identical edges.
LEN_BINS = [0, 30, 50, 75, 150, 5000]

# Bootstrap resamples for MCC confidence intervals. Shared by figure1 (the MCC panel)
# and numbers_manifest so the figure and the numbers manifest report matching CIs.
MCC_CI_N_BOOT = 2000

# Minimum share of the test split that the committed external-tool score snapshot must
# cover before its numbers are quotable. Mirrors compare.py's MIN_COVERAGE: a snapshot
# from an older split still intersects, just to a smaller and meaningless subset.
# ToxDL 2.0 sets the real floor at ~92% (proteins with no AlphaFold model score NA).
EXT_SCORES_MIN_COVERAGE = 0.90


def load_preds(dataset: str, method: str) -> pd.DataFrame:
    """Load a labelled `toxfam eval` benchmark, in canonical identifier order.

    The sort is load-bearing, not cosmetic. `evaluation.runner` writes
    predictions.csv in whatever order the search or dataloader produced, and every
    case-bootstrap downstream (`bootstrap_accuracy_ci`,
    `paired_bootstrap_accuracy_diff`, `local_linear_band`, the inline resample in
    numbers_manifest) draws *positions* — so an unsorted frame makes the reported CI
    a function of that incidental order while the point estimate stays put, which is
    exactly what makes such a drift hard to notice. Canonicalising here fixes the
    whole class at the boundary, the same way `multilabel_stratified_splits` sorts
    before selecting positionally.
    """
    path = benchmark_dir() / dataset / method / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"predictions not found: {path}\n"
            f"Regenerate it first, e.g. 'uv run toxfam eval <method> {dataset}', "
            f"to produce benchmark/{dataset}/{method}/predictions.csv."
        )
    return pd.read_csv(path).sort_values("identifier", ignore_index=True)


def load_predict(dataset: str) -> pd.DataFrame:
    """Load a `toxfam predict` run: top-k families + p_toxic, no ground-truth labels.

    Distinct from load_preds(), which loads a labelled `toxfam eval` benchmark. The
    non-metazoan and unreviewed sets are scored through predict because neither is a
    benchmark: predict builds the taxonomy vectors from each set's own organism IDs,
    so the combined model's taxonomy branch is live rather than zero-filled.

    Reads the single-model output name. `predict --model-dir <combined>` writes the
    -o path verbatim; only the two-model form (--standard-model-dir) suffixes it
    with _combined/_standard. Both sets have an organism ID for every protein, so
    the single-model form covers all of them and needs no standard fallback.

    Returned in canonical identifier order, for the reason given on load_preds().
    """
    path = benchmark_dir() / dataset / "predict" / "predictions.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"predict output not found: {path}\n"
            f"Regenerate it with 'uv run toxfam predict {dataset} "
            f"--model-dir model/model_output/combined_run "
            f"-o benchmark/{dataset}/predict/predictions.tsv'."
        )
    return pd.read_csv(path, sep="\t").sort_values("identifier", ignore_index=True)


def unreviewed_families() -> pd.Series:
    """Raw UniProt "Protein families" for the unreviewed set, indexed by identifier.

    Deliberately raw: the caller collapses to the model's vocabulary via
    paper.stats, so the normalization rules live in one place.
    """
    path = evaluation_data_dir() / "unreviewed" / "unreviewed.tsv"
    df = pd.read_csv(path, sep="\t")
    return df.set_index("Entry")["Protein families"]


def model_vocab() -> set[str]:
    """The combined model's family label space, from its class_indices.json."""
    import json

    path = model_run_dir() / "class_indices.json"
    return set(json.loads(path.read_text()).values())


def deployed_binary_threshold(run: str = "combined_run") -> float:
    """The deployed binary operating point t*, from the run's Platt calibrator.

    ``toxfam predict`` writes a CALIBRATED p_toxic and thresholds it with this same
    value (``toxfam.prediction._read_optimized_threshold``), so 0.5 is NOT the decision
    threshold -- scoring the non-metazoan set at 0.5 understated recall by an order of
    magnitude (13/812 against 218/812). Read at call time, like :func:`model_vocab`, so
    importing a figure module on a checkout without ``model_output/`` still works.
    """
    import json

    path = model_run_dir(run) / "models" / "binary_calibrator.json"
    if not path.exists():
        raise FileNotFoundError(
            f"deployed binary calibrator not found: {path}\n"
            "Fetch the published checkpoints with 'uv run toxfam download-models', or "
            f"deploy one with 'uv run toxfam eval binary model/model_output/{run} "
            "--deploy'."
        )
    return float(json.loads(path.read_text())["threshold"])


def test_set_class_list() -> list[str]:
    """The 38-class label space = sorted unique actual labels on the test set."""
    df = load_preds("test_set", "nn_combined_run")
    return sorted(df["actual_label"].unique().tolist())


def sequence_lengths() -> pd.Series:
    df = pd.read_csv(processed_dir() / "training_data.csv")
    return pd.Series(df["Sequence"].str.len().values, index=df["identifier"].values)


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save vector PDF (primary, for the manuscript) + 600 dpi PNG (preview).

    PDFs are copied into manuscript/Fig/ separately, only after visual verification,
    so a broken render never lands in the manuscript automatically.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Both saves inherit savefig.dpi from apply_style(); see the note there for why a
    # vector PDF needs a dpi at all. Kept as one policy so the deliverable and the
    # preview cannot be given different resolutions.
    fig.savefig(FIG_DIR / f"{name}.pdf")  # vector, fonts embedded (rcParams)
    fig.savefig(FIG_DIR / f"{name}.png")  # raster preview
    plt.close(fig)
    console.print(f"saved {name}.pdf / .png")


def apply_style() -> None:
    """Publication rcParams for Bioinformatics (OUP), built at final column size.

    Font floor is 7 pt at final width (OUP minimum); body 8 pt.

    Figures are built at the JOURNAL's spec (86 mm single / 178 mm double column), which
    is what the standalone files handed to OUP production must satisfy. Note that this is
    ~3.5% wider than the preprint class's own measure (\\textwidth = 488.5 pt = 171.7 mm
    against the 178 mm build), so \\includegraphics[width=\\textwidth] downscales every
    inclusion by 0.965 in main.pdf. Consequence: a 7 pt built label prints at 6.75 pt in
    the preprint, just under the OUP floor. Do NOT "fix" this by retargeting the widths to
    the class -- that would make the production files off-spec. If the floor has to hold in
    the preprint too, raise the built sizes here instead (7 -> 7.5) and re-check every
    figure for label collisions.
    """
    mpl.rcParams.update(
        {
            # fonts (>= 7 pt floor at final size; OUP/Nature minimum)
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            # Arial has no monospace member, so name the mono stack explicitly --
            # otherwise family="monospace" silently falls back to DejaVu Sans Mono.
            "font.monospace": ["Courier New", "DejaVu Sans Mono"],
            # mathtext defaults to the "dejavusans" fontset regardless of font.family,
            # so every $...$ label (the ($n$=63) idiom, the $\approx$18 aa marker) was
            # being set in DejaVu inside an otherwise-Arial figure. pdffonts showed
            # DejaVuSans + DejaVuSans-Oblique embedded in 5 of 8 figures, including
            # main-text Figs. 2 and 3. "custom" routes mathtext through the faces below.
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 9,
            # lines / ticks (OUP 0.35-1.5 pt; no hairlines)
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "grid.linewidth": 0.4,
            "patch.linewidth": 0.5,
            # chartjunk off
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            # export: embed TrueType (not Type-3), keep text as text
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            # white opaque background, vector-first. NB: NO savefig.bbox="tight" -- a tight
            # crop changes the saved width away from the exact figsize (overhanging labels
            # expand it), so \includegraphics[width=\columnwidth] then rescales the figure and
            # the journal shrinks the fonts below the 7 pt floor. layout="constrained" already
            # fits decorations inside the canvas, so saving at the built width keeps fonts at spec.
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 150,
            # dpi matters even for a vector PDF: any rasterized=True layer (the jitter
            # clouds in figure_capability and figure_confidence_curation) is embedded at
            # this resolution. Set here as export policy rather than per-savefig, so the
            # deliverable PDF and the preview PNG cannot be handed different values --
            # the PDF used to inherit figure.dpi=150 while the PNG got 600.
            "savefig.dpi": 600,
            "legend.frameon": False,
        }
    )


def panel_label(ax, letter, *, dx=-0.06, dy=1.02):
    """Bold panel label in axes-fraction coords (Bioinformatics/OUP style).

    Placed just outside the top-left of the axes. ``letter`` is the bare letter and is
    rendered verbatim, so pass the case the caption uses -- every caller and every
    manuscript caption uses uppercase (``"A"``), which is what the captions' ``(A)``
    tags refer to.
    """
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def fmt_pm(value, unc, *, sep=" ± "):
    """Format ``value ± uncertainty`` rounded per GUM/NIST (the ±2 SE rule).

    ``unc`` is the symmetric uncertainty (here ±2 bootstrap SE). It is rounded to
    two significant figures and the value to the same decimal place, e.g.
    (0.9459, 0.052) -> "0.946 ± 0.052" and (0.90, 0.13) -> "0.90 ± 0.13". Pass
    ``sep="\\n±"`` for a two-line label.
    """
    if unc is None or not np.isfinite(unc) or unc <= 0:
        return f"{value:.3f}"
    ndec = max(0, -(math.floor(math.log10(unc)) - 1))  # decimals for 2 sig figs of unc
    return f"{value:.{ndec}f}{sep}{unc:.{ndec}f}"
