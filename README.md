# ToxFam

ToxFam is a Python framework for classifying animal toxin protein sequences into families using MLP neural networks trained on ProtT5 sequence embeddings with optional taxonomy features.

> This project is under active development and not intended for external use or contributions.

## Overview

ToxFam focuses on high-quality classification of protein toxins by reducing sequence redundancy, harmonizing labels, and incorporating rich sequence representations.

Supported workflows:

- Sequence preprocessing and redundancy reduction (MMseqs2 clustering)
- ProtT5 embedding generation
- Optional taxonomy retrieval and encoding
- MLP-based toxin family classification
- Evaluation and benchmarking

## Getting Started

### Prerequisites

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv)
- GPU recommended for embedding generation

### Installation

```bash
git clone git@github.com:Sisistern123/ToxFam.git && cd ToxFam
uv sync
uv run toxfam --help
```

## Workflow

All steps use the unified CLI via `uv run toxfam <command>`:

### 1. Preprocessing

```bash
uv run toxfam preprocess [--no-signalp6] [--min-seq-id 0.9]
```

Filters sequences, reduces redundancy via MMseqs2, and creates stratified train/val/test splits. See [docs/preprocessing.md](docs/preprocessing.md) for a detailed step-by-step walkthrough.

### 2. Feature Generation

```bash
# ProtT5 embeddings
uv run toxfam embed -i <input.fasta> -o <output.h5> [--per-protein]

# Taxonomy annotation and binary vectors
uv run toxfam taxonomy --input-csv <csv> --output-csv <csv>
uv run toxfam taxonomy-vectors --tax-csv <csv> --input-h5 <h5> --output-h5 <h5>
```

### 3. Training

```bash
uv run toxfam train configs/standard.yaml    # embeddings only
uv run toxfam train configs/combined.yaml     # embeddings + taxonomy
```

### 4. Evaluation

```bash
uv run toxfam eval-test [--model-dir <path>]
uv run toxfam eval-nonmetazoan --h5-path <h5> --model-path <pt> --class-map <json>
uv run toxfam eval-unreviewed --input-tsv <tsv> --input-fasta <fasta> --input-h5 <h5>
```
