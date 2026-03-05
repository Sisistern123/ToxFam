# ToxFam

ToxFam is a Python framework for classifying animal toxin protein sequences into families — and ultimately predicting **toxic vs non-toxic** — using MLP neural networks trained on ProtT5 sequence embeddings with optional taxonomy features.

> This project is under active development and not intended for external use or contributions.

## Overview

ToxFam focuses on high-quality classification of protein toxins by reducing sequence redundancy, harmonizing labels, and incorporating rich sequence representations. The system supports multiple training strategies for both multiclass family classification and binary toxic/non-toxic prediction.

Supported workflows:

- Sequence preprocessing and redundancy reduction (MMseqs2 clustering)
- **Identity-aware train/val/test splitting** (no similar sequences leak between splits)
- ProtT5 embedding generation
- Optional taxonomy encoding from NCBI taxon IDs
- MLP-based toxin family classification (standard, combined, hierarchical, binary, multi-task)
- Binary toxic/non-toxic evaluation metrics for all strategies
- Evaluation and benchmarking

## Getting Started

### Prerequisites

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv)
- [MMseqs2](https://github.com/soedinglab/MMseqs2) (for preprocessing and identity-aware splits)
- [SignalP6](docs/signalp6_setup.md) (only needed if re-running signal peptide removal; the cache is included in `toxfam download-data`)
- GPU recommended for embedding generation

### Installation

```bash
git clone git@github.com:Sisistern123/ToxFam.git && cd ToxFam
uv sync
```

### Download Processed Data

Raw data, processed files (ProtT5 embeddings, training splits), and the SignalP6 cache are hosted as GitHub Release assets. Download them with:

```bash
uv run toxfam download-data          # skip existing files
uv run toxfam download-data --force  # re-download everything
```

This places files into `data/raw/`, `data/processed/`, and `data/intermediate/sp6/`. See [Data Directory](#data-directory) for the full layout.

## Workflow

All steps use the unified CLI via `uv run toxfam <command>`:

### 1. Preprocessing

```bash
uv run toxfam preprocess [--min-seq-id 0.9]
```

Removes signal peptides via [SignalP6](docs/signalp6_setup.md) (or uses the downloaded cache), reduces redundancy via per-family MMseqs2 clustering, and creates **identity-aware** stratified train/val/test splits. Requires raw data from `toxfam download-data`. See [docs/preprocessing.md](docs/preprocessing.md) for details.

### 2. Feature Generation

```bash
# ProtT5 embeddings (GPU recommended)
uv run toxfam embed -i <input.fasta> -o <output.h5>

# Taxonomy binary vectors (optional, for combined strategy)
uv run toxfam taxonomy [--input-csv <csv>] [--input-h5 <h5>] [--output-h5 <h5>]
```

See [docs/embedding.md](docs/embedding.md) for embedding options and [docs/taxonomy.md](docs/taxonomy.md) for how taxonomy vectors are built.

### 3. Training

```bash
uv run toxfam train configs/standard.yaml             # 38-class family classification
uv run toxfam train configs/binary.yaml                # direct binary toxic/non-toxic
uv run toxfam train configs/combined.yaml              # embeddings + taxonomy branches
uv run toxfam train configs/hierarchical_existing.yaml # two-stage: family → binary
```

See [configs/readme.md](configs/readme.md) for configuration details and architecture diagrams.

### 4. Evaluation

```bash
uv run toxfam eval-test [--model-dir <path>]
uv run toxfam eval-nonmetazoan --h5-path <h5> --model-path <pt> --class-map <json>
uv run toxfam eval-unreviewed --input-tsv <tsv> --input-fasta <fasta> --input-h5 <h5>
```

All strategies now automatically compute **binary toxic/non-toxic metrics** (ROC-AUC, PR-AUC, F1, MCC) on the test set, saved to `metrics/binary_test_calibrated_metrics.json`.

## Training Strategies

| Strategy | Config | Description |
|----------|--------|-------------|
| **standard** | `configs/standard.yaml` | 38-class family classification with `ModularMLP` |
| **binary** | `configs/binary.yaml` | Direct 2-class toxic/non-toxic with `ModularMLP` |
| **combined** | `configs/combined.yaml` | Two-branch `MultiInputMLP` (embeddings + taxonomy) |
| **hierarchical** | `configs/hierarchical_existing.yaml` | Stage 1: family classifier → Stage 2: binary head on frozen backbone |
| **multitask** | *(create config)* | Joint family + binary classification with shared backbone |

See [docs/training_strategies.md](docs/training_strategies.md) for architecture diagrams and detailed explanations.

## Benchmark Results

Results on identity-aware splits (30% sequence identity threshold between train/val/test):

| Metric | Standard (38-class) | Binary (2-class) |
|--------|-------------------|-----------------|
| Test Accuracy | 92.85% | 96.82% |
| Test MCC | 0.615 | 0.752 |
| Binary ROC-AUC | 0.980 | 0.986 |
| Binary PR-AUC | 0.709 | 0.999 |
| Binary F1 | 0.611 | 0.983 |
| Binary MCC | 0.627 | 0.752 |

**Key findings:**
- The binary strategy is strongly preferred for toxic/non-toxic prediction (PR-AUC 0.999 vs 0.709)
- The standard strategy remains useful for family-level classification (92.85% across 38 classes)
- Identity-aware splits prevent sequence leakage and produce honest, generalizable metrics

See [docs/experiments.md](docs/experiments.md) for full experimental details and comparison.

## Further Documentation

| Document                                         | Description                                               |
| ------------------------------------------------ | --------------------------------------------------------- |
| [docs/preprocessing.md](docs/preprocessing.md)   | Preprocessing pipeline with identity-aware splitting      |
| [docs/training_strategies.md](docs/training_strategies.md) | Training strategies, architectures, and loss functions |
| [docs/experiments.md](docs/experiments.md)        | Experimental results and analysis                         |
| [docs/embedding.md](docs/embedding.md)           | Embedding generation options, resume support, performance |
| [docs/taxonomy.md](docs/taxonomy.md)             | Taxonomy binary vector generation pipeline                |
| [docs/signalp6_setup.md](docs/signalp6_setup.md) | SignalP6 installation and setup guide                     |
| [configs/readme.md](configs/readme.md)           | Training configuration and architecture diagrams          |

## Data Directory

### Raw data

The raw data in `data/raw/` are frozen TSV exports from [UniProt](https://www.uniprot.org/), downloaded via `toxfam download-data`. They were exported on **2026-03-03** using the [UniProt REST API](https://rest.uniprot.org/) with a date cutoff to ensure reproducibility:

| File         | UniProt query                                                                                                          |
| ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `0800.tsv`   | `(taxonomy_id:33208) AND (reviewed:true) AND (fragment:false) AND (date_created:[* TO 2026-03-03]) AND (keyword:KW-0800)` |
| `nontox.tsv` | `(taxonomy_id:33208) AND (reviewed:true) AND (fragment:false) AND (date_created:[* TO 2026-03-03]) NOT (keyword:KW-0800)` |

Exported columns: `accession`, `protein_families`, `organism_id`, `sequence`, `ft_signal` (TSV format via `/uniprotkb/stream`).

### Directory layout

```
data/
├── raw/                    # Frozen UniProt TSV inputs (downloaded via `toxfam download-data`)
│   ├── 0800.tsv            #   toxin proteins (KW-0800)
│   └── nontox.tsv          #   non-toxin proteins
├── intermediate/           # Pipeline intermediates (gitignored, regenerated by `toxfam preprocess`)
│   ├── fasta/
│   ├── identity_splits/    #   global 30% clustering for identity-aware splits
│   ├── mmseqs/
│   │   ├── {family}/
│   │   └── representatives/
│   ├── sp6/                #   SignalP6 cache (downloaded via `toxfam download-data`)
│   └── taxonomy/
└── processed/              # Final outputs (gitignored, downloaded via `toxfam download-data`)
    ├── training_data.csv   #   train/val/test split metadata (identity-aware)
    └── embeddings.h5       #   ProtT5 per-protein embeddings
```

## Testing

```bash
uv run pytest               # run all tests (78 tests)
uv run pytest tests/ -v     # verbose output
uv run ruff check src/toxfam/  # lint
```

## Releasing Data

To update the processed data files distributed via GitHub Releases:

```bash
uv run scripts/upload_data.py
```

This deletes the old `data-v1` release and re-creates it with `0800.tsv`, `nontox.tsv`, `training_data.csv`, `embeddings.h5`, and `sp6_cache.zip`. Requires the [`gh` CLI](https://cli.github.com) to be installed and authenticated.
