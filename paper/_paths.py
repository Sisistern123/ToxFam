"""Central path helpers for the ``paper`` tree.

Repo-only companion to :mod:`toxfam._paths`: keeps every paper-side path in one
place so the figure scripts never hardcode ``analysis/...`` segments again. All
paths resolve from the project root (located by :func:`toxfam._paths.get_project_root`),
so they are stable regardless of the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from toxfam._paths import get_project_root


def paper_root() -> Path:
    """The ``paper/`` tree at the repository root."""
    return get_project_root() / "paper"


def figures_output_dir() -> Path:
    """Where figure PDFs/PNGs + ``results_numbers.{json,tex}`` are written.

    Not created on import; callers that write here should ``mkdir(parents=True,
    exist_ok=True)`` at write time.
    """
    return paper_root() / "figures" / "output"


def paper_data_dir() -> Path:
    """Curated, hand-maintained paper inputs (tracked in git)."""
    return paper_root() / "data"


def model_run_dir(run: str = "combined_run") -> Path:
    """A training run's output directory (gitignored; produced by ``toxfam train``).

    Same rationale as :func:`curated_verdicts_tsv`: the figure scripts resolve run
    artifacts through here rather than each spelling out ``model/model_output/...``.
    """
    return get_project_root() / "model" / "model_output" / run


def curation_dir() -> Path:
    """Blind confident-error curation sheet, its key, and the returned verdicts."""
    return paper_data_dir() / "curation"


def curated_verdicts_tsv() -> Path:
    """Expert verdicts for the confident errors (the sheet, returned filled in).

    Supersedes the earlier ``model_test_wrong_conf_annotated.csv`` (n=63), which was
    adjudicated against a split that has since been shown to be contaminated. This
    sheet was generated blind from the pinned manifest by
    ``scripts/build_confident_error_curation.py``; ``confident_errors_key.tsv``
    un-blinds it (split + calibrated confidence).
    """
    return curation_dir() / "confident_errors_curated.tsv"


def curation_key_tsv() -> Path:
    """Un-blinding key for :func:`curated_verdicts_tsv`: split + confidence."""
    return curation_dir() / "confident_errors_key.tsv"


def protspace_dir() -> Path:
    """Root of the ProtSpace working tree (gitignored; built by ``make protspace``).

    Holds the two ``protspace prepare`` output directories plus the annotation CSVs
    and toxin-only HDF5 they are built from. Gitignored like ``benchmark/``: every
    artifact is reproducible from ``data/processed``.
    """
    return get_project_root() / "protspace"


def protspace_bundle_dir(subset: str = "all") -> Path:
    """Output directory of one ``protspace prepare`` run.

    ``subset="all"`` projects every representative (figure panel A); ``subset="toxin"``
    is refitted on the toxins alone (panel B). They are separate runs because a UMAP
    fitted on a 95%-non-toxin set resolves no family structure among the toxins --
    see :mod:`paper.protspace_bundles`.
    """
    return protspace_dir() / f"out_{subset}"


def manuscript_tex_target() -> Path | None:
    """Optional target for auto-syncing ``results_numbers.tex`` into the manuscript.

    Preserves the historical behaviour of writing the generated LaTeX macros
    straight into the separate manuscript checkout when it is present, but routes
    it through one place and makes the location overridable via the
    ``TOXFAM_MANUSCRIPT_DIR`` environment variable. Returns ``None`` when no
    manuscript checkout is available, so callers simply skip the sync.
    """
    override = os.environ.get("TOXFAM_MANUSCRIPT_DIR")
    if override:
        base = Path(override)
        if not base.is_absolute():
            # Anchor a relative override to the project root, not the CWD, so the
            # result stays stable regardless of where the process was launched.
            base = get_project_root() / base
    else:
        base = get_project_root() / "manuscript"
    return (base / "results_numbers.tex") if base.is_dir() else None
