"""Repo-only ``paper`` tree: manuscript figures, statistics, and curated inputs.

This package is intentionally NOT part of the installable ``toxfam`` wheel
(``[tool.hatch.build.targets.wheel] packages = ["src/toxfam"]``). It holds the
one-off analysis / figure-generation code and the paper-specific statistics that
back the manuscript. Dependency direction is strictly one-way: ``paper`` imports
``toxfam``, never the reverse.

Run figure scripts from the repository root, e.g. ``uv run python -m
paper.figures.figure_pipeline`` (or via the ``Makefile`` targets).
"""
