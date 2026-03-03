# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ToxFam is a research project for classifying animal toxin protein sequences into families using MLP neural networks trained on ProtT5 sequence embeddings with optional taxonomy features. It is structured as an installable Python package (`toxfam`) with a unified Typer CLI.

## Setup & Dependencies

- Python >=3.11, managed with [uv](https://github.com/astral-sh/uv)
- Install: `uv sync`
- Key deps: PyTorch, transformers (ProtT5), scikit-learn, h5py, pymmseqs, protspace, iterative-stratification, taxopy, pydantic, typer
- SignalP6 required for preprocessing (setup: `docs/signalp6_setup.md`)
- Large processed data files (HDF5, CSV) are distributed via GitHub Releases; download with `uv run toxfam download-data`

## Common Commands

All commands are run via the `toxfam` CLI using `uv run`:

### Download Processed Data
```bash
uv run toxfam download-data
```

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
- `standard.yaml` — embeddings-only MLP
- `combined.yaml` — two-branch MLP (embeddings + taxonomy)
- `example.yaml` — annotated reference config

### Evaluation / Benchmarking
```bash
# Evaluate on test set (compares NN vs HBI sequence similarity)
uv run toxfam eval-test [--model-dir <path>]

# Evaluate on non-metazoan reviewed proteins
uv run toxfam eval-nonmetazoan --h5-path <h5> --model-path <pt> --class-map <json>

# Evaluate on unreviewed metazoan proteins
uv run toxfam eval-unreviewed --input-tsv <tsv> --input-fasta <fasta> --input-h5 <h5>
```

## Architecture

### Package Structure

```
src/toxfam/
├── cli.py                    # Typer app: unified CLI entry point
├── config.py                 # Pydantic TrainConfig model
├── _paths.py                 # get_project_root() utility
├── data/                     # Data loading, preprocessing, feature generation
│   ├── dataset.py            # ToxDataset, analyze_data_splits
│   ├── preprocessing.py      # Full preprocessing pipeline
│   ├── embedding.py          # ProtT5 embedding generation
│   ├── taxonomy.py           # Taxonomy retrieval + binary vector generation
│   └── signalp.py            # SignalP6 signal peptide removal
├── model/                    # Neural network architectures
│   ├── architectures.py      # ModularMLP, MultiInputMLP
│   └── calibration.py        # ModelWithTemperature
├── training/                 # Training loop, strategies, orchestration
│   ├── trainer.py            # train_model, evaluate_model, get_class_weights
│   ├── strategies.py         # DataSelector, run_*_strategy, evaluate_label_on_dataset
│   └── orchestrator.py       # run_training(config) — main pipeline
├── evaluation/               # Benchmark evaluation scripts
│   ├── eval_test_set.py      # Test set evaluation (HBI vs NN)
│   ├── eval_nonmetazoan.py   # Non-metazoan binary classification
│   └── eval_unreviewed.py    # Unreviewed metazoan evaluation
└── visualization/            # Plotting utilities
    ├── plots.py              # plot_loss_curve, plot_confusion_matrix
    └── analysis.py           # label distribution, ROC curves
```

### Training Strategies (the central design axis)

The system supports two training strategies, selected via `training_strategy` in the YAML config:

1. **`standard`** — `ModularMLP` fed with ProtT5 embeddings only (1024-dim)
2. **`combined`** — `MultiInputMLP` with two branches: one for embeddings, one for binary taxonomy vectors (56-dim), concatenated before a joint head

### Config

Training config is a Pydantic `TrainConfig` model (`src/toxfam/config.py`) loaded from YAML. It replaces the old global `CONFIG` dict. Every function that needs config receives it as a `config: TrainConfig` parameter.

### Data Directory Layout

```
data/
├── raw/                        # Manually obtained inputs (committed to git)
│   ├── 0800.tsv
│   └── nontox.tsv
├── intermediate/               # All pipeline-generated intermediates (gitignored)
│   ├── fasta/                  # tox.fasta, nontox.fasta, *_noSP.fasta
│   ├── mmseqs/                 # All MMseqs2-related files
│   │   ├── {family}/           # Per-family: input.fasta + cluster output
│   │   └── representatives/    # Post-clustering rep seqs (CSV + FASTA)
│   ├── sp6/                    # SignalP6 output (tox/, nontox/)
│   └── taxonomy/               # Binary taxonomy vectors
│       └── binary_taxonomy_vectors.h5
├── processed/                  # Expensive outputs (gitignored, via GitHub Releases)
│   ├── training_data.csv       # Train/val/test split CSV
│   └── embeddings.h5           # ProtT5 embeddings
```

### Data Flow

1. **Raw data** (`data/raw/`) — UniProt TSVs of toxin/non-toxin proteins
2. **Preprocessing** (`toxfam.data.preprocessing`) — normalizes family labels, runs SignalP6 signal peptide removal, clusters per-family with MMseqs2 at 90% identity, creates multilabel-stratified train/val/test splits; intermediates go to `data/intermediate/`, final split CSV to `data/processed/`
3. **Feature generation**:
   - `toxfam.data.embedding` — ProtT5 per-protein embeddings → `data/processed/embeddings.h5`
   - `toxfam.data.taxonomy` — reads `Organism (ID)` from training CSV → taxopy lineage → binary (one-hot) vectors over 56 predefined taxa → `data/intermediate/taxonomy/`
4. **Training** (`toxfam.training.orchestrator`) — loads split CSV + embeddings from `data/processed/` and optionally taxonomy vectors from `data/intermediate/taxonomy/`; dispatches to strategy, trains with early stopping, applies temperature scaling calibration, evaluates on val/test sets
5. **Outputs** (configured via `output_dir` in YAML) — `best_model.pt`, `best_model_calibrated.pt`, confusion matrices, ROC curves, predictions CSV, metrics JSON

### Key Module Relationships

- `toxfam.config` — Pydantic `TrainConfig` model, loaded via `TrainConfig.from_yaml(path)`
- `toxfam.data.dataset` — `ToxDataset` reads embeddings from multiple HDF5 files with LRU caching; optionally loads taxonomy vectors from a separate HDF5
- `toxfam.training.strategies` — `DataSelector` wraps DataLoaders to route the correct inputs per strategy
- `toxfam.training.orchestrator` — `run_training(config)` orchestrates the full training → evaluation → calibration pipeline
- `toxfam.model.calibration` — `ModelWithTemperature` wraps trained model with learned temperature scaling
- `toxfam.model.architectures` — `ModularMLP` (projector + backbone), `MultiInputMLP` (two-branch)

### Data Format Conventions

- Protein IDs are in the `identifier` column (renamed from UniProt `Entry`)
- Family labels are in the `Protein families` column
- Split assignments are in the `Split` column (`train`/`val`/`test`)
- HDF5 files are keyed by protein identifier, each entry is a 1D float array

## Important Details

- All imports use fully-qualified package paths: `from toxfam.model.architectures import ModularMLP`
- All commands can be run from the project root via `uv run toxfam <command>`
- Families with <10 members are collapsed into an `"other"` class during preprocessing
- The taxonomy binary vectors encode membership in 56 predefined animal taxa (from Porifera to Soricidae), defined in `toxfam.data.taxonomy.TAXA`
- Path resolution uses `toxfam._paths.get_project_root()` which finds the project root by walking up to find `pyproject.toml`
