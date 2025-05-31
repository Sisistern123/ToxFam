# ToxFam

**ToxFam** is a research-driven Python framework for classifying animal toxin protein sequences into families. It combines modern sequence embeddings, machine learning models, and curated preprocessing to improve protein family assignments in toxin datasets.

> **Note:** This project is under active development and not intended for external cloning, reuse, or contributions at this time.

---

## 🧬 Overview

ToxFam focuses on the high-quality classification of protein toxins by reducing sequence redundancy and enhancing label consistency. It supports workflows for:

- Sequence preprocessing and filtering
- Redundancy reduction via sequence similarity thresholds
- Label reassignment for ambiguous or mislabeled sequences
- Embedding generation using ProtT5
- Machine learning-based toxin family classification (via MLP)

---

## 🚧 Project Status

This repository is part of an internal research project. The codebase is subject to change frequently and may contain placeholders, draft components, and experimental scripts.
**Please do not use this repository for production or reproducibility purposes.**

---

## 🗂 Directory Structure (subject to change)
```
ToxFam/
├── data/ # Input and intermediate data files
├── model/ # model architecture and training
├── model_output/ # model output
├── notebooks/ # Jupyter notebooks for data preprocessing
└── generate_embeds.py # script for generating embeddings
```

---