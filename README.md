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

A typical ToxFam workflow looks roughly like this:

1. **Data preparation**
   - Collect toxin protein sequences and associated family labels.
   - Store raw and intermediate data under `data/` (see structure below).

2. **Preprocessing & redundancy reduction**
   - Filter sequences (e.g. by length, alphabet, or metadata).
   - Reduce redundancy using sequence similarity thresholds.
   - Optionally relabel or discard ambiguous entries.

3. **Feature generation**
   - Compute sequence embeddings using a chosen backend (e.g. ProtT5, ProtSpace).
   - Optionally retrieve taxonomy information (e.g. from external services) and encode it as features.

4. **Model training & evaluation**
   - Train MLP-based classifiers on the generated features.
   - Evaluate model performance across toxin families.
   - Export predictions, metrics, and confidence scores.

5. **Analysis & benchmarking**
   - Inspect confidence distributions, misclassifications, and edge cases.
   - Compare models, feature sets, or preprocessing settings using scripts and notebooks in `analysis/` and `benchmark/`.

> The exact entry points (scripts / notebooks) for each step may evolve as the project develops.  
> Always refer to the latest internal documentation, notebooks, or commit messages for the current workflow.

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

# Alternatively, install via pip in an active virtual environment
# (Adjust to your internal conventions as needed)
pip install -e .
