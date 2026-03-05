# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ToxFam is a research project for classifying animal toxin protein sequences into families — and predicting **toxic vs non-toxic** — using MLP neural networks trained on ProtT5 sequence embeddings with optional taxonomy and CPP features. It is structured as an installable Python package (`toxfam`) with a unified Typer CLI.

## Setup & Dependencies

- Python >=3.11, managed with [uv](https://github.com/astral-sh/uv)
- Install: `uv sync`
- Key deps: PyTorch, transformers (ProtT5), scikit-learn, h5py, pymmseqs, protspace, aaanalysis, iterative-stratification, taxopy, pydantic, typer
- Dev deps: pytest, ruff
- `aaanalysis` is installed as an editable local dependency from SpeciesEmbedding tools (provides CPP features)
- wandb is **optional** — training works without it; install separately if needed
- SignalP6 only needed if re-running signal peptide removal (setup: `docs/signalp6_setup.md`); the cache is included in `toxfam download-data`
- Large processed data files (HDF5, CSV) are distributed via GitHub Releases; download with `uv run toxfam download-data`

## Common Commands

All commands are run via the `toxfam` CLI using `uv run`:

### Download Processed Data
```bash
uv run toxfam download-data            # skip existing files
uv run toxfam download-data --force    # re-download everything
```
Downloads raw data to `data/raw/`, ProtT5 embeddings and training splits to `data/processed/`, and the SignalP6 cache to `data/intermediate/sp6/`. Use `--force` to re-download existing files.

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

### CPP Features (Comparative Physicochemical Profiling)
```bash
uv run toxfam cpp --training-csv <csv> [--output <h5>] [--n-filter 100]
```

### Train a Model
```bash
uv run toxfam train configs/standard.yaml             # 38-class family classification
uv run toxfam train configs/binary.yaml                # direct binary toxic/non-toxic
uv run toxfam train configs/binary_cpp.yaml            # binary with CPP features
uv run toxfam train configs/combined.yaml              # embeddings + taxonomy branches
uv run toxfam train configs/hierarchical_existing.yaml # two-stage: family → binary (frozen)
uv run toxfam train configs/hierarchical_unfrozen.yaml # two-stage with fine-tuned backbone
uv run toxfam train configs/multitask.yaml             # joint family + binary heads
```
The config YAML selects the training strategy. See `configs/readme.md` for full details.

Set `n_folds: 5` in config YAML to enable k-fold cross-validation.

### Evaluation / Benchmarking
```bash
# Re-compute binary metrics without retraining
uv run toxfam eval-binary <model_dir>

# Ensemble evaluation (average predictions from multiple models)
uv run toxfam eval-ensemble <model_dir1> <model_dir2> ...

# Evaluate on test set (compares NN vs HBI sequence similarity)
uv run toxfam eval-test [--model-dir <path>]

# Evaluate on non-metazoan reviewed proteins
uv run toxfam eval-nonmetazoan --h5-path <h5> --model-path <pt> --class-map <json>

# Evaluate on unreviewed metazoan proteins
uv run toxfam eval-unreviewed --input-tsv <tsv> --input-fasta <fasta> --input-h5 <h5>

# Profile training data for biases
uv run toxfam profile-data --input-csv <csv> [--h5-path <h5>]
```

All strategies automatically compute **binary toxic/non-toxic metrics** (ROC-AUC, PR-AUC, F1, MCC) on the test set with both default and optimized thresholds.

### Testing
```bash
uv run pytest               # run all tests (96 tests)
uv run pytest tests/ -v     # verbose output
```

### Linting
```bash
uv run ruff check src/toxfam/
```

## Architecture

### Package Structure

```
src/toxfam/
├── cli.py                    # Typer app: unified CLI entry point
├── config.py                 # Pydantic TrainConfig model (with effective_embedding_dim property)
├── device.py                 # Canonical get_device() (cuda > mps > cpu)
├── _paths.py                 # get_project_root() and directory helpers
├── data/                     # Data loading, preprocessing, feature generation
│   ├── _fasta.py             # parse_fasta, read_fasta_as_dict, write_fasta
│   ├── cpp_features.py       # CPP feature generation via AAanalysis
│   ├── dataset.py            # ToxDataset, analyze_data_splits
│   ├── normalization.py      # normalize_protein_families (shared)
│   ├── preprocessing.py      # Full preprocessing pipeline (incl. SignalP6, identity-aware splits, rebalancing)
│   ├── hierarchical_preprocessing.py  # Hierarchical data assembly
│   ├── embedding.py          # ProtT5 embedding generation
│   ├── taxonomy.py           # Taxonomy retrieval + binary vector generation
│   ├── xml_parser.py         # Parse UniProt XML → DataFrame
│   └── label_validation.py   # MMseqs2-based family label validation
├── model/                    # Neural network architectures
│   ├── architectures.py      # ModularMLP, MultiInputMLP, HierarchicalMLP, MultiTaskMLP
│   ├── calibration.py        # ModelWithTemperature
│   └── losses.py             # FocalLoss
├── training/                 # Training loop, strategies, orchestration
│   ├── trainer.py            # train_model, evaluate_model, get_class_weights, _build_loss_fn
│   ├── strategies.py         # DataSelector, run_*_strategy, _MultiTaskFamilyWrapper, _MultiTaskBinaryWrapper
│   ├── hierarchical.py       # Two-stage hierarchical training (family → tox/nontox)
│   ├── orchestrator.py       # run_training(config) — main pipeline with threshold optimization
│   └── cross_validation.py   # k-Fold CV at cluster level
├── evaluation/               # Benchmark evaluation scripts
│   ├── metrics.py            # calculate_binary_metrics, calculate_multiclass_metrics, find_optimal_threshold
│   ├── ensemble.py           # Ensemble model evaluation
│   ├── data_quality.py       # Data profiling for bias detection
│   ├── eval_test_set.py      # Test set evaluation (HBI vs NN)
│   ├── eval_nonmetazoan.py   # Non-metazoan binary classification
│   └── eval_unreviewed.py    # Unreviewed metazoan evaluation
└── visualization/            # Plotting utilities
    ├── plots.py              # plot_loss_curve, plot_confusion_matrix
    └── analysis.py           # label distribution, ROC curves, binary ROC, PR curves

tests/                        # pytest test suite (96 tests)
├── conftest.py               # Shared fixtures
├── test_device.py
├── test_fasta.py
├── test_normalization.py
├── test_metrics.py           # Binary metrics, threshold optimization, to_binary_class
├── test_architectures.py     # ModularMLP, MultiInputMLP, MultiTaskMLP forward shapes
├── test_calibration.py
├── test_config.py            # Config fields, effective_embedding_dim, n_folds
├── test_dataset.py
├── test_cli.py
├── test_paths.py
├── test_xml_parser.py
├── test_hierarchical.py
├── test_hierarchical_preprocessing.py
├── test_losses.py            # FocalLoss tests
├── test_identity_splits.py   # Identity-aware split + rebalancing tests
├── test_cross_validation.py  # k-Fold CV aggregation tests
└── test_ensemble.py          # Ensemble module import tests
```

### Shared Modules (deduplication)

- **`toxfam.device`** — Single `get_device()` function used by embedding, training, evaluation
- **`toxfam.data._fasta`** — `parse_fasta`, `read_fasta_as_dict`, `write_fasta` (with MD5 skip and parameterized column names)
- **`toxfam.data.normalization`** — `normalize_protein_families()` used by preprocessing and evaluation
- **`toxfam.evaluation.metrics`** — `calculate_binary_metrics()`, `calculate_multiclass_metrics()`, `calculate_binary_metrics_with_scores()`, `find_optimal_threshold()`, `to_binary_class()`, `NONTOXIN_LABELS`

### Training Strategies (the central design axis)

The system supports five training strategies, selected via `training_strategy` in the YAML config:

1. **`standard`** — `ModularMLP` fed with ProtT5 embeddings only (1024-dim), 38-class family prediction
2. **`binary`** — `ModularMLP` with 2 output classes, direct toxic/non-toxic prediction (recommended for binary task)
3. **`combined`** — `MultiInputMLP` with two branches: one for embeddings, one for binary taxonomy vectors (56-dim), concatenated before a joint head
4. **`hierarchical`** — Two-stage: Stage 1 trains `ModularMLP` on family classification, Stage 2 loads Stage 1's projector as frozen backbone into `HierarchicalMLP` with a binary tox/nontox head
5. **`multitask`** — `MultiTaskMLP` with shared backbone, joint family + binary heads, `loss = α*L_family + β*L_binary`

All strategies automatically compute binary toxic/non-toxic metrics after training, including threshold optimization.

### Config

Training config is a Pydantic `TrainConfig` model (`src/toxfam/config.py`) loaded from YAML. Every function that needs config receives it as a `config: TrainConfig` parameter. Extra fields in YAML are silently ignored (`model_config = {"extra": "ignore"}`).

Key config fields:
- `loss_function`: `"cross_entropy"` (default) or `"focal"`
- `focal_gamma`: gamma for focal loss (default 2.0)
- `multitask_family_weight` / `multitask_binary_weight`: loss weights for multitask strategy
- `split_seq_id`: sequence identity threshold for identity-aware splitting (default 0.3)
- `cpp_h5_path` / `cpp_dim`: CPP feature file and dimension (auto-adjusts input dim via `effective_embedding_dim`)
- `n_folds`: k-fold cross-validation folds (1 = no CV, default)

Important property: `config.effective_embedding_dim` returns `embedding_dim + cpp_dim` when CPP is enabled. All model construction uses this property instead of raw `embedding_dim`.

### Identity-Aware Splitting

The preprocessing pipeline uses `identity_aware_splits()` to prevent sequence leakage between train/val/test:

1. **Global clustering at 30% identity** — All representatives clustered with MMseqs2
2. **Cluster-level splitting** — Entire clusters assigned to train/val/test (70/15/15) using `MultilabelStratifiedShuffleSplit`
3. **Post-assignment rebalancing** — `_rebalance_splits()` moves smallest clusters to train when family representation < 50%
4. **Adaptive relaxation** — Families stuck in one cluster are re-clustered at 40%, 50%, 60%, 70% until splittable
5. **Split quality logging** — Families with <5 train samples or missing val/test are reported

### Training Pipeline

The orchestrator (`run_training()`) follows this flow:
1. Load config, save config copy to output_dir
2. Load data, compute class weights
3. Dispatch to strategy (standard/binary/combined/hierarchical/multitask)
4. Uncalibrated evaluation on val+test
5. Temperature scaling calibration on validation set
6. Calibrated evaluation on val+test
7. Binary metrics pipeline:
   - Compute on validation set → threshold optimization (Youden's J)
   - Compute on test set with default threshold (0.5)
   - Compute on test set with optimized threshold
   - Multitask: also evaluate binary head directly via `_MultiTaskBinaryWrapper`

### Data Directory Layout

```
data/
├── raw/                        # Frozen UniProt TSV inputs (downloaded via `toxfam download-data`)
│   ├── 0800.tsv
│   └── nontox.tsv
├── intermediate/               # All pipeline-generated intermediates (gitignored)
│   ├── fasta/                  # tox.fasta, nontox.fasta, *_noSP.fasta
│   ├── identity_splits/        # Global 30% clustering for identity-aware splits
│   ├── mmseqs/                 # All MMseqs2-related files
│   │   ├── {family}/           # Per-family: input.fasta + cluster output
│   │   └── representatives/    # Post-clustering rep seqs (CSV + FASTA)
│   ├── sp6/                    # SignalP6 output + per-sequence cache (downloaded via `toxfam download-data`)
│   ├── cpp/                    # CPP physicochemical features
│   └── taxonomy/               # Binary taxonomy vectors
│       └── binary_taxonomy_vectors.h5
├── processed/                  # Expensive outputs (gitignored, via GitHub Releases)
│   ├── training_data.csv       # Train/val/test split CSV (identity-aware)
│   └── embeddings.h5           # ProtT5 embeddings
```

### Data Flow

1. **Raw data** (`data/raw/`) — UniProt TSVs of toxin/non-toxin proteins
2. **Preprocessing** (`toxfam.data.preprocessing`) — normalizes family labels (via `toxfam.data.normalization`), runs SignalP6 signal peptide removal (per-sequence MD5-based caching in `sp6_cache.json`), clusters per-family with MMseqs2 at 90% identity, creates **identity-aware** train/val/test splits with post-assignment rebalancing; intermediates go to `data/intermediate/`, final split CSV to `data/processed/`
3. **Feature generation**:
   - `toxfam.data.embedding` — ProtT5 per-protein embeddings → `data/processed/embeddings.h5`
   - `toxfam.data.taxonomy` — reads `Organism (ID)` from training CSV → taxopy lineage → binary (one-hot) vectors over 56 predefined taxa → `data/intermediate/taxonomy/`
   - `toxfam.data.cpp_features` — AAanalysis CPP features (tox vs nontox physicochemical profiles) → `data/intermediate/cpp/`
4. **Training** (`toxfam.training.orchestrator`) — loads split CSV + embeddings from `data/processed/` and optionally taxonomy/CPP vectors; dispatches to strategy, trains with early stopping, applies temperature scaling calibration, optimizes threshold on validation set, evaluates on val/test sets, computes binary metrics. wandb logging is optional. Config saved to output_dir for re-evaluation.
5. **Outputs** (configured via `output_dir` in YAML) — `best_model.pt`, `best_model_calibrated.pt`, `config.yaml`, confusion matrices, ROC curves, binary ROC/PR curves, predictions CSV, metrics JSON, threshold optimization JSON

### Key Module Relationships

- `toxfam.device` — Canonical `get_device()` (cuda > mps > cpu), imported everywhere that needs a device
- `toxfam.config` — Pydantic `TrainConfig` model, loaded via `TrainConfig.from_yaml(path)`; `effective_embedding_dim` property handles CPP input dim
- `toxfam.data._fasta` — All FASTA I/O: `parse_fasta`, `read_fasta_as_dict`, `write_fasta`
- `toxfam.data.normalization` — `normalize_protein_families()` shared by preprocessing and evaluation
- `toxfam.data.dataset` — `ToxDataset` reads embeddings from multiple HDF5 files with LRU caching; optionally loads taxonomy/CPP vectors from separate HDF5 files
- `toxfam.evaluation.metrics` — Shared metrics: `calculate_binary_metrics`, `calculate_multiclass_metrics`, `calculate_binary_metrics_with_scores`, `find_optimal_threshold`, `to_binary_class`
- `toxfam.evaluation.ensemble` — Ensemble model evaluation (average softmax probabilities)
- `toxfam.evaluation.data_quality` — Data profiling for bias detection
- `toxfam.training.strategies` — `DataSelector` wraps DataLoaders; `_MultiTaskFamilyWrapper` and `_MultiTaskBinaryWrapper` wrap MultiTaskMLP heads
- `toxfam.training.orchestrator` — `run_training(config)` orchestrates training → calibration → threshold optimization → binary metrics
- `toxfam.training.cross_validation` — `run_kfold_training()` for cluster-level k-fold CV
- `toxfam.model.calibration` — `ModelWithTemperature` wraps trained model with learned temperature scaling
- `toxfam.model.architectures` — `ModularMLP` (projector + backbone), `MultiInputMLP` (two-branch), `HierarchicalMLP` (frozen backbone + head), `MultiTaskMLP` (shared backbone + dual heads)
- `toxfam.model.losses` — `FocalLoss` (configurable gamma, optional class weights)
- `toxfam.training.trainer` — `_build_loss_fn()` builds loss from config (cross-entropy or focal)

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
- Device detection uses `toxfam.device.get_device()` — never inline `torch.cuda.is_available()` checks
- wandb is fully optional — guarded by `try/except` at import time
- SignalP6 integration lives entirely in `preprocessing.py` (not in a separate `signalp.py`)
- `aaanalysis` is installed as an editable local dep from SpeciesEmbedding tools (uv source override for matplotlib version constraint)
- `NONTOXIN_LABELS = {"nontox", "nontoxic"}` — both label variants map to nontoxin in binary classification
- Model construction always uses `config.effective_embedding_dim` (not raw `embedding_dim`) to account for CPP feature concatenation
- Multitask binary head uses weighted cross-entropy (same as family head) for proper handling of class imbalance
- Training saves `config.yaml` to output_dir for later re-evaluation via `toxfam eval-binary`
