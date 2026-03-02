# ToxFam

**ToxFam** is a research-driven Python framework for classifying animal toxin protein sequences into families.
It combines sequence embeddings, optional taxonomy features, and simple neural network models to improve protein family assignments in toxin datasets.

> **Note:** This project is under active development and not intended for external cloning, reuse, or contributions at this time.

---

## 🧬 Overview

ToxFam focuses on **high-quality classification of protein toxins** by reducing sequence redundancy, harmonizing labels, and incorporating rich representations of sequences (and optionally, taxonomy).

It currently supports workflows for:

- Sequence preprocessing and filtering
- Redundancy reduction via sequence similarity
- Embedding generation (e.g. **ProtT5**, **ProtSpace**)
- Optional taxonomy retrieval and encoding
- MLP-based toxin family classification and evaluation
- Downstream analysis of confidence scores and benchmarking

The code is primarily intended for **internal research experiments**, not as a general-purpose library.

---

## ✨ Key Features

- **Research-oriented design**
  Built for iterative experimentation with toxin family classification, not for production deployments.

- **Flexible embeddings**
  Pluggable support for different embedding backends (e.g. ProtT5, ProtSpace), making it easy to compare representation choices.

- **Optional taxonomy integration**
  Retrieval and encoding of taxonomic information as additional features where available.

- **Toxin-focused preprocessing**
  Tools for filtering, deduplicating, and relabeling toxin protein sequences to improve label quality and reduce redundancy.

- **Simple neural models**
  Lightweight MLP models for toxin family prediction, emphasizing interpretability and ease of experimentation.

- **Analysis utilities**
  Scripts and notebooks for confidence score inspection and benchmarking across models or feature sets.

---

## 🚧 Project Status

This repository is part of an **internal research project**.

- The codebase changes frequently.
- Components may be **experimental**, **incomplete**, or refactored without notice.
- Some scripts and modules are intended only for one-off experiments.
- Documentation may lag behind the current code state.

Please **do not**:

- Treat this repository as a stable, versioned package.
- Use it for production workloads or strict reproducibility.
- Open external issues or pull requests.

---

## 🧩 High-Level Workflow

All steps are accessible through the unified `toxfam` CLI (run via `uv run toxfam <command>`):

1. **Data preparation & preprocessing**
   ```bash
   uv run toxfam preprocess [--run-signalp6] [--min-seq-id 0.9]
   ```
   Filters sequences, reduces redundancy via MMseqs2 clustering, and creates stratified train/val/test splits.

2. **Feature generation**
   ```bash
   # ProtT5 embeddings
   uv run toxfam embed -i <input.fasta> -o <output.h5> [--per-protein]

   # Taxonomy annotation
   uv run toxfam taxonomy --input-csv <csv> --output-csv <csv>

   # Binary taxonomy vectors
   uv run toxfam taxonomy-vectors --tax-csv <csv> --input-h5 <h5> --output-h5 <h5>
   ```

3. **Model training**
   ```bash
   uv run toxfam train configs/standard.yaml      # embeddings only
   uv run toxfam train configs/combined.yaml       # embeddings + taxonomy
   ```

4. **Evaluation & benchmarking**
   ```bash
   uv run toxfam eval-test [--model-dir <path>]
   uv run toxfam eval-nonmetazoan --h5-path <h5> --model-path <pt> --class-map <json>
   uv run toxfam eval-unreviewed --input-tsv <tsv> --input-fasta <fasta> --input-h5 <h5>
   ```

5. **Analysis**
   - Inspect confidence distributions, misclassifications, and edge cases using notebooks in `analysis/` and `benchmark/`.

---

## 🚀 Getting Started (Internal Use)

The following steps are **guidelines** for internal users of the repository.
Concrete commands or entry points may differ depending on your environment and current branch.

### Prerequisites

- Python (version as specified in `pyproject.toml`)
- [uv](https://github.com/astral-sh/uv) or a standard Python environment with `pip`
- (Optional but recommended) GPU support for large-scale embedding generation

### Installation

```bash
# Clone the repository (internal path or URL)
git clone git@github.com:Sisistern123/ToxFam.git ToxFam
cd ToxFam

# Install dependencies with uv (recommended)
uv sync

# Verify the CLI works
uv run toxfam --help
```
