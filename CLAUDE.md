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
- `standard.yaml` — embeddings-only MLP
- `combined.yaml` — two-branch MLP (embeddings + taxonomy)
- `example.yaml` — annotated reference config

### Evaluation / Benchmarking

Each method is evaluated independently; results are compared separately.

```bash
# Run HBI baseline on test set
uv run toxfam eval hbi test_set

# Run a trained model on test set
uv run toxfam eval model test_set --model-dir model/model_output/combined_run

# Compare all methods that have been run
uv run toxfam eval compare test_set
```

Available datasets: `test_set`, `val_set`, `non_metazoan`, `unreviewed`. Results go to `benchmark/{dataset}/{method}/`.

## Architecture

### Package Structure

```
src/toxfam/
├── cli.py                    # Typer app: unified CLI entry point
├── config.py                 # Pydantic TrainConfig model (effective_embedding_dim property)
├── device.py                 # Canonical get_device() (cuda > mps > cpu)
├── _paths.py                 # get_project_root() and directory helpers
├── data/                     # Data loading, preprocessing, feature generation
│   ├── _fasta.py             # parse_fasta, read_fasta_as_dict, write_fasta
│   ├── dataset.py            # ToxDataset (embeddings + optional CPP/HBI/taxonomy)
│   ├── preprocessing.py      # Full pipeline incl. identity-aware splits
│   ├── embedding.py          # ProtT5 embedding generation
│   ├── taxonomy.py           # Taxonomy retrieval + multi-hot vector generation
│   ├── signalp.py            # SignalP6 signal peptide removal
│   ├── normalization.py      # normalize_protein_families (shared)
│   ├── xml_parser.py         # Parse UniProt XML → DataFrame
│   ├── cpp_features.py       # CPP physicochemical profiling via AAanalysis
│   └── hbi_features.py       # HBI sequence similarity features via MMseqs2
├── model/                    # Neural network architectures
│   ├── architectures.py      # ModularMLP, MultiInputMLP
│   ├── calibration.py        # ModelWithTemperature
│   ├── model_config.py       # ModelConfig for deterministic architecture reconstruction
│   └── inference.py          # Model loading + inference for evaluation
├── training/                 # Training loop, strategies, orchestration
│   ├── trainer.py            # train_model, evaluate_model, FocalLoss, get_class_weights
│   ├── strategies.py         # DataSelector, run_{standard,binary,combined}_strategy
│   └── orchestrator.py       # run_training(config) + binary metrics pipeline
├── evaluation/               # Benchmark evaluation
│   ├── runner.py             # run_hbi_evaluation, run_model_evaluation, compare_methods
│   ├── hbi.py                # MMseqs2 HBI search (run_hbi_search, HBIResult)
│   ├── metrics.py            # MetricsResult + binary score metrics + threshold optimization
│   ├── ensemble.py           # Ensemble model evaluation
│   └── data_quality.py       # Training data profiling for bias detection
└── visualization/            # Plotting utilities
    ├── plots.py              # plot_loss_curve, plot_confusion_matrix
    └── analysis.py           # label distribution, ROC curves, binary ROC/PR
```

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
- `cpp_h5_path` / `cpp_dim` / `hbi_h5_path` / `hbi_dim`: auxiliary feature files
- `include_length` / `include_venom_indicator`: scalar features
- `split_seq_id`: identity threshold for identity-aware splitting

Important property: `config.effective_embedding_dim` returns `embedding_dim + cpp_dim + hbi_dim + ...` when auxiliary features are enabled. All model construction uses this property.

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
- `toxfam.evaluation.runner` — dataset registry, `run_hbi_evaluation()`, `run_model_evaluation()`, `compare_methods()`; each writes standard outputs (predictions.csv, metrics.json, run_metadata.json) to `benchmark/{dataset}/{method}/`
- `toxfam.evaluation.hbi` — MMseqs2 search wrapper (`run_hbi_search()` → `HBIResult`)
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
