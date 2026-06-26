# Dataset Reference

This document keeps the current core datasets and their provenance in one place.

## Raw (tox and nontox)

These are the frozen UniProt exports used as preprocessing input.

| File | Purpose | Query | Original export cutoff date | Local file date |
| --- | --- | --- | --- | --- |
| `data/raw/0800.tsv` | Toxin proteins (KW-0800) | `(taxonomy_id:33208) AND (reviewed:true) AND (fragment:false) AND (date_created:[* TO 2026-03-03]) AND (keyword:KW-0800)` | 2026-03-03 | 2026-04-02 |
| `data/raw/nontox.tsv` | Non-toxin proteins | `(taxonomy_id:33208) AND (reviewed:true) AND (fragment:false) AND (date_created:[* TO 2026-03-03]) NOT (keyword:KW-0800)` | 2026-03-03 | 2026-04-02 |

Newest notes query variants (for overall context):

- Tox: `(keyword:KW-0800) AND (reviewed:true) AND (taxonomy_id:33208) AND (fragment:false)`
- Nontox: `(reviewed:true) AND (taxonomy_id:33208) AND (fragment:false) AND ((existence:1) OR (existence:2)) NOT (keyword:KW-0800)`

## Unreviewed

| File | Query | Original file date | Notes |
| --- | --- | --- | --- |
| `data/evaluation/unreviewed/unreviewed.tsv` | `(keyword:KW-0800) AND (reviewed:false) AND (taxonomy_id:33208) AND (fragment:false)` | 2026-04-13 | Used by `toxfam eval model unreviewed` |

## Non-metazoan

| File | Query | Original file date | Notes |
| --- | --- | --- | --- |
| `data/evaluation/non_metazoan/non_metazoan.tsv` | `(keyword:KW-0800) AND (reviewed:true) AND (date_created:[* TO 2026-03-03]) NOT (taxonomy_id:33208)` | 2026-04-13 | Used by `toxfam eval hbi non_metazoan` and `toxfam eval model non_metazoan`. Reviewed KW-0800 toxins from non-metazoan organisms. Reproduces the 812 entries in the current file (same date cutoff as the raw training data; note no `fragment:false` filter). Re-export with an added `Organism (ID)` field for combined-model `toxfam predict`. |

## Processed files (derived artifacts)

Processed files are generated or downloaded artifacts derived from the datasets above.

| File | How it is produced | What it is used for |
| --- | --- | --- |
| `data/processed/training_data.csv` | Output of `uv run toxfam preprocess` from raw tox+nontox | Train/val/test split table (`Split` column) |
| `data/processed/hbi_train_all.csv` | Output of preprocessing (train representatives expanded to all members) | HBI target table |
| `data/processed/hbi_train_all.fasta` | Output of preprocessing | HBI target FASTA |
| `data/processed/embeddings.h5` | `uv run toxfam embed` (or downloaded release artifact) | Sequence embeddings for training/eval |
| `data/processed/taxonomy_vectors.h5` | `uv run toxfam taxonomy` | Optional taxonomy input for combined model |