# Repository Restructuring — Plan of Record

**Status:** executed on branch `refactor/paper-tree` (2026-07). Commits C1–C6 landed and
verified; C7 = adversarial review + PR. This document is the durable record of *what*
changed and *why*, so the work is resumable across sessions.

## Goal

Separate the reusable, installable library (`src/toxfam/`) from one-off manuscript /
analysis / figure code, kill hardcoded file paths, and make figure regeneration a single
deterministic command — without breaking the `toxfam` CLI, package imports, or the tests.

The `manuscript/` directory is a **separate git repo** (Overleaf mirror), gitignored here
and never touched by this work; only the figure-copy convention matters.

## Research basis (authoritative Python packaging + reproducible-research guides)

- **Keep src-layout + `[build-system]` + a `[project.scripts]` console-script** for a
  distributable Typer CLI (PyPA, uv, pytest, pyOpenSci). ToxFam already did this correctly.
- **Reusable code stays in the package; one-off analysis/figure/notebook code lives in a
  separate tree**, orchestrated by a make-like tool so figures regenerate deterministically.
- **Version-control source + config, not regenerated artifacts** — tempered here by the
  repo's existing deliberate policy (track the manuscript-feeding PDFs + `results_numbers`).
- **Centralize paths** (a path helper / config) instead of hardcoding them.
- Tests in a top-level `tests/` sibling of `src/`; `importlib` import mode for new suites.

## Decisions (signed off)

| # | Decision | Choice |
|---|----------|--------|
| Q1 | How far to separate paper code | **Full split** — figures **and** `manuscript.py` (+ its test) move to `paper/` |
| Q2 | Figure regeneration mechanism | **Makefile** targets (`make figures`), not a CLI subcommand or Snakemake/DVC |
| Q3 | Committed-artifact policy | **Keep selective tracking** — track figure PDFs + `results_numbers.{json,tex}`; gitignore PNGs + big CSVs |
| Q4 | Notebook / cruft cleanup | **Consolidate + prune** — keep the demo, prune stale notebooks + superseded rasters |

## Before → after

```
BEFORE                              AFTER
analysis/figures/*.py           →   paper/figures/*.py
analysis/manuscript_figures/    →   paper/figures/output/   (PDFs+numbers tracked, PNGs ignored)
src/toxfam/evaluation/manuscript.py → paper/stats.py        (leaves the wheel)
tests/test_manuscript.py        →   paper/tests/test_stats.py
analysis/model_test_wrong_conf_annotated.csv → paper/data/
analysis/curation/confident_errors_key.tsv   → paper/data/curation/
notebooks/ToxFam_predict.ipynb  →   examples/
notes.md                        →   docs/notes.md
(new)                           →   paper/_paths.py, Makefile, docs/reorg_plan.md
analysis/{*.ipynb, plots/, Pfams*.csv, ...}   → PRUNED (git history)
```

## Path / config changes (hardcoded → centralized)

All paper-side paths now resolve through **`paper/_paths.py`**:
- `figures_output_dir()` — replaces the hardcoded `analysis/manuscript_figures` (`_common.FIG_DIR`).
- `adjudication_csv()` — single source for the former `ADJ_CSV`, which was duplicated
  verbatim in `numbers_manifest.py` and `figure_confidence_curation.py`.
- `manuscript_tex_target()` — the `results_numbers.tex` auto-sync into the manuscript repo;
  behaviour preserved, now overridable via `TOXFAM_MANUSCRIPT_DIR`.
- `import-mode = importlib` + `pythonpath = ["."]` + `testpaths = ["tests", "paper/tests"]`
  added to `[tool.pytest.ini_options]`.

## Regenerating figures

```bash
make figures          # rebuild all figures + results_numbers
make fig-pipeline     # a single figure
```

**Reproducibility caveat:** figures read the gitignored `model/model_output/` and
`benchmark/` trees, so a clean checkout must first produce them:

```bash
uv run toxfam train configs/standard.yaml
uv run toxfam train configs/combined.yaml
uv run toxfam eval hbi test_set && uv run toxfam eval eat test_set \
  && uv run toxfam eval model test_set --model-dir model/model_output/combined_run
make figures
```

(The `Makefile` header documents this chain.)

## Commit sequence (each verified: `uv run pytest` green + `toxfam --help` intact)

1. **C1** — scaffold `paper/` + pytest importlib config (152 → 157 tests).
2. **C2** — move `analysis/figures/` → `paper/figures/`; repoint imports + output dir.
3. **C3** — kill hardcoded paths (single-source `ADJ_CSV`, curated data → `paper/data/`).
4. **C4** — full split: `manuscript.py` → `paper/stats.py` (+ test); wheel now excludes it.
5. **C5** — prune cruft + consolidate + remove `analysis/`.
6. **C6** — docs (CLAUDE.md `paper/` section, this file), `Makefile`, `test_cli` eat/predict.
7. **C7** — multi-agent adversarial verification of the full diff → PR.

## Deferred (intentionally out of scope)

- `figure_pipeline.py` hardcodes the frozen dataset counts (mirrors
  `manuscript/dataset_numbers.tex`) — a manuscript-numeric sync issue, not a structural one.
- `src/toxfam/visualization/taxonomy_sunburst.py` still writes to the top-level `figures/`
  (a library-CLI output, left as-is).
- `tests/test_model_config.py`'s `skipif`-masked model paths (pre-existing).
