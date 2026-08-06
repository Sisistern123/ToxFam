# ToxFam

ToxFam is a Python framework for classifying animal toxin protein sequences into families using MLP neural networks trained on ProtT5 sequence embeddings with optional taxonomy features.

> This project is under active development and not intended for external use or contributions.

## Predict in Google Colab

Run the trained models on your own sequences — no install required. Upload a FASTA file (or precomputed ProtT5 embeddings) and get the top-3 family predictions per sequence:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sisistern123/ToxFam/blob/main/examples/ToxFam_predict.ipynb)

## Overview

ToxFam focuses on high-quality classification of protein toxins by reducing sequence redundancy, harmonizing labels, and incorporating rich sequence representations.

Supported workflows:

- Sequence preprocessing and redundancy reduction (MMseqs2 clustering)
- ProtT5 embedding generation
- Optional taxonomy encoding from NCBI taxon IDs
- MLP-based toxin family classification
- Evaluation and benchmarking

## Getting Started

### Prerequisites

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv)
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

Taxonomy vectors are **not** in the release — regenerate them with `uv run toxfam taxonomy` (needed by the combined model). They are rebuilt against live NCBI taxonomy, so pin the dump via `PROTSPACE_TAXDB_DIR=<frozen copy>` if you need byte-identical vectors.

### Download Trained Models

```bash
uv run toxfam download-models          # -> model/model_output/{standard,combined}_run
uv run toxfam download-models --force  # re-download over existing runs
```

Use this instead of `toxfam train` when the goal is to **reproduce the published numbers**: a fresh training run produces a different checkpoint, so its metrics will not match the manuscript. The release carries the calibrated checkpoint, its architecture/class metadata, and `models/split_provenance.json` binding it to the split it trained on — but *not* `models/binary_calibrator.json`, so see [Evaluation](#4-evaluation) for the one step you must run locally.

## Workflow

All steps use the unified CLI via `uv run toxfam <command>`:

### 1. Preprocessing

```bash
uv run toxfam preprocess [--min-seq-id 0.9]
```

Removes signal peptides via [SignalP6](docs/signalp6_setup.md) (or uses the downloaded cache), reduces redundancy via per-family MMseqs2 clustering, and creates stratified train/val/test splits. Requires raw data from `toxfam download-data`. See [docs/preprocessing.md](docs/preprocessing.md) for details.

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
uv run toxfam train configs/standard.yaml    # embeddings only
uv run toxfam train configs/combined.yaml     # embeddings + taxonomy
```

See [configs/readme.md](configs/readme.md) for configuration details and architecture diagrams.

### 4. Evaluation

```bash
# Deploy the binary P(toxic) Platt calibrator — run ONCE PER MODEL, before anything
# that quotes binary numbers. Writes <run>/metrics/binary_metrics.json plus
# <run>/models/binary_calibrator.json, neither of which ships in the model release.
uv run toxfam eval binary model/model_output/combined_run --deploy
uv run toxfam eval binary model/model_output/standard_run --deploy

uv run toxfam eval hbi test_set
uv run toxfam eval eat test_set
# --model-dir selects the model; results go to benchmark/test_set/nn_<run>/, so run
# it once per model you want in the comparison.
uv run toxfam eval model test_set --model-dir model/model_output/combined_run
uv run toxfam eval compare test_set
```

Without `--deploy`, `eval binary` is a **diagnostic only** and deliberately does not touch the shipped calibrator. Manuscript numbers require the deployed one: `paper.figures.numbers_manifest` refuses a `binary_metrics.json` whose `score_space` is not `platt_calibrated`, because those metrics are computed on the raw score. See the `Makefile` header for the full `download → taxonomy → eval → figures` chain.

### 5. Prediction

Run a trained model on arbitrary proteins (no ground-truth labels needed). Outputs a TSV with the top-K family predictions, calibrated confidences, and a binary toxic/non-toxic call.

```bash
# Combined model (input needs identifier + Organism (ID); proteins without an organism ID are skipped)
uv run toxfam predict input.tsv --model-dir model/model_output/combined_run --embeddings emb.h5

# Combined + standard fallback (proteins without an organism ID are predicted by the standard model → two TSVs)
uv run toxfam predict input.tsv \
  --model-dir model/model_output/combined_run \
  --standard-model-dir model/model_output/standard_run --embeddings emb.h5

# Standard model (organism IDs ignored, all proteins predicted)
uv run toxfam predict input.tsv --model-dir model/model_output/standard_run --embeddings emb.h5

# Toxic / non-toxic only (drops the family columns; works in all modes)
uv run toxfam predict input.tsv --model-dir model/model_output/standard_run --embeddings emb.h5 --toxicity-only
```

The input is either a TSV path or a registered dataset name (`non_metazoan`, `unreviewed`, `test_set`, `val_set` — a name also auto-selects its embeddings H5). A TSV needs an `identifier` column (`Entry` is also accepted), plus `Organism (ID)` for combined models and `Sequence` only when embeddings must be generated. Omit `--embeddings` to embed every sequence from scratch. Combined runs report organism IDs with no taxonomy signal in a `*_unresolved_organisms.tsv` sidecar. See [docs/predict.md](docs/predict.md) for the full input/output contract and the three usage modes.

## Further Documentation

| Document                                         | Description                                               |
| ------------------------------------------------ | --------------------------------------------------------- |
| [docs/preprocessing.md](docs/preprocessing.md)   | Step-by-step preprocessing pipeline walkthrough           |
| [docs/embedding.md](docs/embedding.md)           | Embedding generation options, resume support, performance |
| [docs/taxonomy.md](docs/taxonomy.md)             | Taxonomy binary vector generation pipeline                |
| [docs/predict.md](docs/predict.md)               | `toxfam predict` input/output contract and usage modes    |
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
│   ├── mmseqs/
│   │   ├── {family}/
│   │   └── representatives/
│   ├── sp6/                #   SignalP6 cache (downloaded via `toxfam download-data`)
│   └── taxonomy/
└── processed/              # Final outputs (gitignored, downloaded via `toxfam download-data`)
    ├── training_data.csv   #   train/val/test split metadata
    └── embeddings.h5       #   ProtT5 per-protein embeddings
```

## Releasing Data

To update the processed data files distributed via GitHub Releases:

```bash
uv run scripts/upload_data.py
```

This deletes the old `data-v1` release and re-creates it with `0800.tsv`, `nontox.tsv`, `training_data.csv`, `embeddings.h5`, and `sp6_cache.zip`. Requires the [`gh` CLI](https://cli.github.com) to be installed and authenticated.
