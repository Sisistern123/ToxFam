"""Shared loaders and matplotlib style for manuscript figures.

Style and palette follow verified Bioinformatics (OUP) figure guidelines and
colour-blind-safe palette research (see
docs/superpowers/specs/2026-06-30-figure-overhaul-design.md):

* Build at final column width (double = 178 mm = 7.008 in) -- never draw large and
  let the journal shrink it (that is what made earlier text illegible).
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
    path = benchmark_dir() / dataset / method / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"predictions not found: {path}\n"
            f"Regenerate it first, e.g. 'uv run toxfam eval <method> {dataset}', "
            f"to produce benchmark/{dataset}/{method}/predictions.csv."
        )
    return pd.read_csv(path)


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
    """
    path = benchmark_dir() / dataset / "predict" / "predictions.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"predict output not found: {path}\n"
            f"Regenerate it with 'uv run toxfam predict {dataset} "
            f"--model-dir model/model_output/combined_run "
            f"-o benchmark/{dataset}/predict/predictions.tsv'."
        )
    return pd.read_csv(path, sep="\t")


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
    fig.savefig(FIG_DIR / f"{name}.pdf")  # vector, fonts embedded (rcParams)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=600)  # raster preview
    plt.close(fig)
    console.print(f"saved {name}.pdf / .png")


def apply_style() -> None:
    """Publication rcParams for Bioinformatics (OUP), built at final column size.

    Font floor is 7 pt at final width (OUP minimum); body 8 pt. Built at true
    column width so nothing is shrunk afterwards, keeping every label legible.
    """
    mpl.rcParams.update(
        {
            # fonts (>= 7 pt floor at final size; OUP/Nature minimum)
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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
            "legend.frameon": False,
        }
    )


def panel_label(ax, letter, *, dx=-0.06, dy=1.02):
    """Lowercase bold panel label in axes-fraction coords (Bioinformatics/OUP style).

    Placed just outside the top-left of the axes; ``letter`` should be the bare
    letter (``"a"``), rendered as a bold lowercase tag.
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
