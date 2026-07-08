# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ToxFam is a research project for classifying animal toxin protein sequences into families using MLP neural networks trained on ProtT5 sequence embeddings with optional taxonomy features. It is structured as an installable Python package (`toxfam`) with a unified Typer CLI.

## Setup & Dependencies

- Python >=3.11, managed with [uv](https://github.com/astral-sh/uv)
- Install: `uv sync`
- Key deps: PyTorch, transformers (ProtT5), scikit-learn, h5py, pymmseqs, protspace, iterative-stratification, taxopy, pydantic, typer
- SignalP6 only needed if re-running signal peptide removal (setup: `docs/signalp6_setup.md`); the cache is included in `toxfam download-data`
- Large processed data files (HDF5, CSV) are distributed via GitHub Releases; download with `uv run toxfam download-data`

## Common Commands

All commands are run via the `toxfam` CLI using `uv run`:

### Download Processed Data
```bash
uv run toxfam download-data
```
Downloads raw data to `data/raw/`, training splits, ProtT5 embeddings, and HBI reference data to `data/processed/`, evaluation datasets to `data/evaluation/`, and the SignalP6 cache to `data/intermediate/sp6/`. Taxonomy vectors are not included — regenerate with `toxfam taxonomy`. Use `--force` to re-download existing files.

To upload/update the release, use the developer script: `uv run scripts/upload_data.py`

### Data Preprocessing Pipeline
```bash
uv run toxfam preprocess [--min-seq-id 0.9]
```

### Generate ProtT5 Embeddings
```bash
uv run toxfam embed -i <input.fasta> -o <output.h5>
```

### Taxonomy Binary Vectors
```bash
uv run toxfam taxonomy [--input-csv <csv>] [--input-h5 <h5>] [--output-h5 <h5>]
```

### Train a Model
```bash
uv run toxfam train configs/standard.yaml
uv run toxfam train configs/combined.yaml
```
The config YAML selects the training strategy. Available configs in `configs/`:
- `standard.yaml` — embeddings-only MLP (38-class family prediction)
- `binary.yaml` — direct binary toxic/non-toxic MLP
- `combined.yaml` — two-branch MLP (embeddings + taxonomy)
- `example.yaml` — annotated reference config

### Predict (inference on arbitrary proteins)

```bash
# Combined model (input TSV needs identifier + Organism (ID); + Sequence if embeddings must be generated)
uv run toxfam predict input.tsv --model-dir model/model_output/combined_run --embeddings data/processed/embeddings.h5

# Combined + standard fallback (organism-less proteins → standard model; writes two TSVs)
uv run toxfam predict input.tsv --model-dir model/model_output/combined_run --standard-model-dir model/model_output/standard_run --embeddings <emb.h5>

# Standard model (organism IDs ignored, all proteins predicted)
uv run toxfam predict input.tsv --model-dir model/model_output/standard_run --embeddings <emb.h5>
```

The input may be a TSV path or a registered dataset name (`non_metazoan`, `unreviewed`, `test_set`, `val_set` — same registry as `eval`; a name auto-selects its embeddings H5). Outputs a TSV with top-K family predictions (`pred_1..K`, `conf_1..K`), `p_toxic`, and `predicted_toxic`. Always uses the calibrated model (you pass a model directory, not a checkpoint). Missing embeddings are generated from the `Sequence` column. Use `--toxicity-only` to output just the binary call (`identifier`, `p_toxic`, `predicted_toxic`) in any mode. Combined runs write a `*_unresolved_organisms.tsv` sidecar listing organism IDs that yielded no taxonomy signal (unresolvable taxon ID, or organism not among the model's 50 taxa). See `docs/predict.md` for the full contract.

### Evaluation / Benchmarking

Each method is evaluated independently; results are compared separately.

```bash
# Run HBI baseline on test set
uv run toxfam eval hbi test_set

# Run EAT baseline (embedding-based annotation transfer; embedding-space analog of HBI)
uv run toxfam eval eat test_set

# Run a trained model on test set
uv run toxfam eval model test_set --model-dir model/model_output/combined_run

# Compare all methods that have been run
uv run toxfam eval compare test_set

# Re-compute binary toxic/nontoxin metrics from a trained model
uv run toxfam eval binary model/model_output/standard_run
```

Available datasets: `test_set`, `val_set`, `non_metazoan`, `unreviewed`. Results go to `benchmark/{dataset}/{method}/`.

### Visualization
```bash
uv run toxfam plot taxonomy
```
Generates interactive sunburst plots of taxonomic distribution for toxin and non-toxin proteins.

## Architecture

### Package Structure

```
src/toxfam/
├── cli.py                    # Typer app: unified CLI entry point
├── config.py                 # Pydantic TrainConfig model
├── device.py                 # Canonical get_device() (cuda > mps > cpu)
├── _paths.py                 # get_project_root() and directory helpers
├── data/                     # Data loading, preprocessing, feature generation
│   ├── _fasta.py             # parse_fasta, read_fasta_as_dict, write_fasta
│   ├── dataset.py            # ToxDataset (embeddings + optional taxonomy)
│   ├── registry.py           # Dataset registry + loaders (shared by eval AND predict; light imports)
│   ├── preprocessing.py      # Full pipeline: normalize, SignalP6, cluster, stratified splits
│   ├── embedding.py          # ProtT5 embedding generation
│   ├── taxonomy.py           # Taxonomy retrieval + multi-hot vector generation
│   └── normalization.py      # normalize_protein_families (shared)
├── model/                    # Neural network architectures
│   ├── architectures.py      # ModularMLP, MultiInputMLP
│   ├── forward.py            # Shared forward pass (torch-only leaf; keeps model/ dep-free of training/)
│   ├── calibration.py        # ModelWithTemperature
│   ├── model_config.py       # ModelConfig for deterministic architecture reconstruction
│   └── inference.py          # Model loading + inference for evaluation
├── training/                 # Training loop, strategies, orchestration
│   ├── trainer.py            # train_model, evaluate_model, FocalLoss, get_class_weights
│   ├── strategies.py         # DataSelector, run_{standard,binary,combined}_strategy
│   └── orchestrator.py       # run_training(config) + binary metrics pipeline
├── prediction.py             # toxfam predict: label-free inference (top-K family + binary toxicity)
├── evaluation/               # Benchmark evaluation
│   ├── runner.py             # run_{hbi,eat,model}_evaluation, compare_methods
│   ├── hbi.py                # MMseqs2 HBI search (run_hbi_search, HBIResult)
│   ├── eat.py                # Embedding 1-NN annotation transfer (run_eat_search, EATResult)
│   ├── metrics.py            # MetricsResult + binary score metrics + threshold optimization
│   └── binary.py             # Score-based binary evaluation (P(toxic), ROC-AUC, threshold opt)
└── visualization/            # Plotting utilities
    ├── plots.py              # plot_loss_curve, plot_confusion_matrix
    ├── analysis.py           # label distribution, ROC curves, binary ROC/PR
    └── taxonomy_sunburst.py  # Plotly sunburst plots for taxonomic distribution
```

### Paper / Analysis Tree (`paper/`, repo-only — NOT in the wheel)

One-off manuscript code lives in `paper/`, deliberately kept out of the installable
`toxfam` wheel (`[tool.hatch.build.targets.wheel] packages = ["src/toxfam"]`).
Dependency direction is strictly one-way: `paper` imports `toxfam`, never the reverse.

```
paper/
├── _paths.py          # central path helpers (figure output, curated data, manuscript sync)
├── stats.py           # manuscript statistics (mcnemar, bootstrap CIs, per-family F1, ...) — unit-tested
├── figures/           # thin matplotlib render scripts (run via the Makefile, not by hand)
│   ├── _common.py     # shared style + loaders; save_fig() -> paper/figures/output/
│   ├── figure_*.py    # one module per figure (each exposes main())
│   ├── numbers_manifest.py   # emits paper/figures/output/results_numbers.{json,tex}
│   └── output/        # rendered artifacts (PDFs + results_numbers tracked; PNGs gitignored)
├── data/              # hand-curated inputs (adjudication CSV, curation key)
└── tests/             # tests for paper.stats + paper._paths (collected via testpaths)
```

- **Regenerate figures** with the `Makefile`: `make figures` (all) or `make fig-<name>`.
  Do NOT invoke `python -m paper.figures.<name>` by hand.
- Figures read the gitignored `model/model_output/` + `benchmark/` trees, so a clean
  checkout must first run `toxfam train` + `toxfam eval` — see the `Makefile` header for
  the full `train → eval → figures` chain.
- The manuscript statistics formerly in `src/toxfam/evaluation/manuscript.py` now live
  in `paper/stats.py` (imported as `from paper.stats import ...`).
- Hardcoded figure paths (the old `ADJ_CSV`, figure-output dir, manuscript `.tex` sync)
  are centralized in `paper/_paths.py`; the manuscript sync target is overridable via
  the `TOXFAM_MANUSCRIPT_DIR` env var.

### Training Strategies (the central design axis)

The system supports three training strategies, selected via `training_strategy` in the YAML config:

1. **`standard`** — `ModularMLP` fed with ProtT5 embeddings only (1024-dim), 38-class family prediction
2. **`binary`** — `ModularMLP` with 2 output classes, direct toxic/non-toxic prediction (recommended for binary task)
3. **`combined`** — `MultiInputMLP` with two branches: one for embeddings, one for multi-hot taxonomy vectors, concatenated before a joint head

All strategies automatically compute **binary toxic/non-toxic metrics** (ROC-AUC, PR-AUC, F1, MCC) on the test set with both default and optimized thresholds.

### Config

Training config is a Pydantic `TrainConfig` model (`src/toxfam/config.py`) loaded from YAML. Every function that needs config receives it as a `config: TrainConfig` parameter. Extra fields in YAML are silently ignored (`model_config = {"extra": "ignore"}`).

Key config fields:
- `use_focal_loss` / `focal_loss_gamma`: focal loss for class imbalance
- `lr_scheduler` / `warmup_epochs`: cosine annealing with warmup
- `early_stopping_metric`: `"loss"` or `"mcc"`

### Data Directory Layout

```
data/
├── raw/                        # Frozen UniProt TSV inputs (downloaded via `toxfam download-data`)
│   ├── 0800.tsv
│   └── nontox.tsv
├── intermediate/               # Pipeline-generated intermediates (gitignored, reproducible)
│   ├── fasta/                  # tox.fasta, nontox.fasta, *_noSP.fasta
│   ├── mmseqs/                 # All MMseqs2-related files
│   │   ├── {family}/           # Per-family: input.fasta + cluster output
│   │   └── representatives/    # Post-clustering rep seqs (CSV + FASTA)
│   └── sp6/                    # SignalP6 output + per-sequence cache (downloaded via `toxfam download-data`)
├── processed/                  # All training inputs (gitignored, via GitHub Releases)
│   ├── training_data.csv       # Train/val/test split CSV (Split column: train/val/test)
│   ├── embeddings.h5           # ProtT5 embeddings
│   ├── taxonomy_vectors.h5     # Multi-hot taxonomy vectors (50-dim)
│   ├── hbi_train_all.csv       # All cluster members of training reps (for HBI search)
│   └── hbi_train_all.fasta     # Same in FASTA format (MMseqs2 target database)
└── evaluation/                 # Evaluation-specific input data (git-tracked)
    ├── non_metazoan/           # Non-metazoan reviewed protein data
    └── unreviewed/             # Unreviewed metazoan protein data

benchmark/                      # Evaluation results only (gitignored, regenerated by eval commands)
├── {dataset}/                  # e.g. test_set, non_metazoan, unreviewed
│   ├── hbi/                    # HBI method results (predictions.csv, metrics.json, ...)
│   ├── nn_{model_name}/        # NN model results
│   └── comparison/             # Cross-method comparison (metric_comparison.csv)
```

### Data Flow

1. **Raw data** (`data/raw/`) — UniProt TSVs of toxin/non-toxin proteins
2. **Preprocessing** (`toxfam.data.preprocessing`) — normalizes family labels, runs SignalP6 signal peptide removal (per-sequence MD5-based caching in `sp6_cache.json`), clusters per-family with MMseqs2 at 90% identity, creates multilabel-stratified train/val/test splits; intermediates go to `data/intermediate/`, final split CSV + HBI reference to `data/processed/`
3. **Feature generation**:
   - `toxfam.data.embedding` — ProtT5 per-protein embeddings → `data/processed/embeddings.h5`
   - `toxfam.data.taxonomy` — reads `Organism (ID)` from training CSV → taxopy lineage → multi-hot vectors over 50 predefined taxa → `data/processed/taxonomy_vectors.h5`
4. **Training** (`toxfam.training.orchestrator`) — loads split CSV, embeddings, and optionally taxonomy vectors from `data/processed/`; dispatches to strategy, trains with early stopping, applies temperature scaling calibration, evaluates on val/test sets
5. **Outputs** (configured via `output_dir` in YAML) — `best_model.pt`, `best_model_calibrated.pt`, confusion matrices, ROC curves, predictions CSV, metrics JSON

### Key Module Relationships

- `toxfam.config` — Pydantic `TrainConfig` model, loaded via `TrainConfig.from_yaml(path)`
- `toxfam.data.dataset` — `ToxDataset` reads embeddings from multiple HDF5 files with LRU caching; optionally loads taxonomy vectors from a separate HDF5
- `toxfam.training.strategies` — `DataSelector` wraps DataLoaders to route the correct inputs per strategy
- `toxfam.training.orchestrator` — `run_training(config)` orchestrates the full training → evaluation → calibration pipeline
- `toxfam.model.calibration` — `ModelWithTemperature` wraps trained model with learned temperature scaling
- `toxfam.model.architectures` — `ModularMLP` (projector + backbone), `MultiInputMLP` (two-branch)
- `toxfam.model.inference` — loads calibrated models from training output, runs inference for evaluation
- `toxfam.data.registry` — named dataset registry (`DATASETS`, `list_datasets()`, `load_dataset()`, `resolve_embeddings_h5()`); lives in the light `data` layer so `toxfam predict` resolves dataset names without importing the eval/plotting stack
- `toxfam.evaluation.runner` — `run_hbi_evaluation()`, `run_eat_evaluation()`, `run_model_evaluation()`, `run_binary_evaluation_from_dir()`, `compare_methods()` (consumes `data.registry`); each writes standard outputs (predictions.csv, metrics.json, run_metadata.json) to `benchmark/{dataset}/{method}/`
- `toxfam.evaluation.hbi` — MMseqs2 search wrapper (`run_hbi_search()` → `HBIResult`)
- `toxfam.evaluation.eat` — embedding 1-NN annotation transfer (`run_eat_search()` → `EATResult`); reference = training split, transfers nearest ProtT5 neighbour's family + distance-margin `p_toxic`
- `toxfam.evaluation.metrics` — unified metrics (`calculate_metrics()` → `MetricsResult`)

### Data Format Conventions

- Protein IDs are in the `identifier` column (renamed from UniProt `Entry`)
- Family labels are in the `Protein families` column
- Split assignments are in the `Split` column (`train`/`val`/`test`)
- HDF5 files are keyed by protein identifier, each entry is a 1D float array

## Important Details

- All imports use fully-qualified package paths: `from toxfam.model.architectures import ModularMLP`
- All commands can be run from the project root via `uv run toxfam <command>`
- Families with <10 members are collapsed into an `"other"` class during preprocessing
- The taxonomy multi-hot vectors encode membership in 50 predefined animal taxa (from Porifera to Soricidae), defined in `toxfam.data.taxonomy.TAXA`
- Path resolution uses `toxfam._paths.get_project_root()` which finds the project root by walking up to find `pyproject.toml`
- The project uses `rich` for all CLI output (progress bars via `rich.progress.Progress`, styled messages via `console.print()`) — not `tqdm` or raw `print()`
