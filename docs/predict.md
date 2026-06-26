# `toxfam predict`

Pure inference: predict the toxin family and a binary toxic/non-toxic call for
arbitrary proteins, using a trained model. No ground-truth labels, no metrics,
no `benchmark/` writes — just predictions.

```bash
uv run toxfam predict INPUT.tsv --model-dir <model_dir> [options]
```

## Input

The input argument is either a **path to a tab-separated file** or a **registered
dataset name** (the same names `toxfam eval` accepts: `non_metazoan`,
`unreviewed`, `test_set`, `val_set`). A dataset name auto-resolves its TSV and
its default embeddings H5 (override with `--embeddings`). Unlike `eval`, rows are
*not* dropped for missing family labels — every protein is predicted.

When a file path is given, it is a **tab-separated** file with these columns:

| Column          | Required?                                    | Used for                                  |
| --------------- | -------------------------------------------- | ----------------------------------------- |
| `identifier`    | Always (`Entry` is accepted and renamed)     | Key into embeddings + output row key      |
| `Organism (ID)` | Combined models only                         | NCBI taxon ID → multi-hot taxonomy vector |
| `Sequence`      | Only when an embedding must be generated     | ProtT5 embedding generation               |

If you pass `--embeddings` and it already contains every `identifier`, the
`Sequence` column is never read and may be omitted. Otherwise the missing
sequences are embedded on the fly (the `Sequence` column is then required).

## Models and calibration

You pass a model **directory** (`--model-dir`), not a checkpoint path. `predict`
always loads the **calibrated** model (`models/best_model_calibrated.pt`) and
applies the learned temperature, so the reported confidences and `p_toxic` are
temperature-scaled. The architecture (combined `MultiInputMLP` vs. standard
`ModularMLP`) is auto-detected from `model_config.json`.

## Usage modes

The mode is chosen automatically from the primary model and whether a standard
fallback is supplied:

| Mode | Invocation | Proteins without an organism ID |
| --- | --- | --- |
| **Combined only** | `--model-dir <combined>` | **Excluded** (logged) |
| **Combined + standard** | `--model-dir <combined> --standard-model-dir <standard>` | Predicted by the standard model |
| **Standard only** | `--model-dir <standard>` | N/A — organism IDs ignored, all predicted |

A protein "has an organism ID" when `Organism (ID)` parses as a number.
`--standard-model-dir` must point at a standard (`ModularMLP`) model; it is
ignored (with a warning) when the primary model is already standard.

## Embeddings

ProtT5 embeddings are reused from `--embeddings` (an H5 keyed by `identifier`)
when present, and generated otherwise via the same pipeline as `toxfam embed`:

- All identifiers present in `--embeddings` → used as-is, no embedding step.
- Some missing → the supplied H5 is **copied** (non-destructive) and only the
  missing sequences are embedded into the copy.
- No `--embeddings` → everything is embedded from `Sequence` into a temporary H5.

Generating ProtT5 embeddings downloads the model from HuggingFace on first use
and is GPU-accelerated when available (CUDA > MPS > CPU). Supplying precomputed
embeddings is much faster for large inputs.

## Taxonomy (combined models)

For combined models, a per-run multi-hot taxonomy H5 is built from each pool's
`Organism (ID)` using the **same** pipeline as `toxfam taxonomy` (50 fixed taxa,
fixed order), guaranteeing the encoding matches training. The taxopy database is
downloaded automatically on first use.

### Taxonomy coverage warnings

A protein can end up with an **all-zero** taxonomy vector — meaning the combined
model has no taxonomy signal for it and the prediction relies on the embedding
alone. `predict` detects this, prints a summary, and writes a
`<output>_unresolved_organisms.tsv` sidecar (columns: `identifier`,
`Organism (ID)`, `reason`). There are two reasons:

- **`unresolvable taxon id`** — taxopy could not look the ID up at all. This
  happens when the taxon ID is obsolete/merged/deleted in NCBI taxonomy, is
  malformed, or the local taxopy database is stale (refresh it by regenerating
  taxonomy, or it auto-refreshes periodically).
- **`organism not among model's 50 taxa`** — the lineage resolved fine, but none
  of the 50 predefined `TAXA` appear in it (e.g. a bacterium, or a metazoan
  lineage outside the trained taxa). The model simply has no feature for it.

These proteins are still predicted; the warning just flags that their taxonomy
branch contributed nothing.

## Output

A TSV with the top-K family predictions and the binary call:

```
identifier  pred_1  conf_1  pred_2  conf_2  pred_3  conf_3  p_toxic  predicted_toxic
```

- `pred_1..K` / `conf_1..K` — the top-K families by calibrated probability
  (`--top-k`, default 3).
- `p_toxic` — score-based toxicity probability, `1 − Σ P(nontoxin classes)`.
  Always included alongside the family columns (use `--toxicity-only` to get
  just the binary call).
- `predicted_toxic` — `p_toxic ≥ threshold`, where `threshold` is the model's
  own `optimized_threshold` from `metrics/binary_metrics.json` (falls back to
  0.5 if absent).

**Combined + standard mode writes two disjoint files** instead of one:
`<output>_combined.tsv` (proteins with an organism ID) and
`<output>_standard.tsv` (proteins without). Each pool uses its own model's
calibrated threshold. Because the two models can have different class label
sets, compare families within a file, not across.

### Toxicity-only output

With `--toxicity-only`, the per-family columns are dropped and the output is just
the binary call — in every mode:

```
identifier  p_toxic  predicted_toxic
```

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--model-dir` | *(required)* | Primary model directory |
| `--standard-model-dir` | `None` | Standard fallback for organism-less proteins |
| `--embeddings` | `None` | Precomputed ProtT5 embeddings H5 |
| `-o`, `--output` | `predictions.tsv` | Output TSV path (suffixed per pool in combined+standard mode) |
| `--top-k` | `3` | Number of top family predictions |
| `--toxicity-only` | off | Only predict toxic/non-toxic; drop the family columns |
| `--max-residues` | `4000` | Max residues per embedding batch |
| `--max-batch` | `100` | Max sequences per embedding batch |

## Examples

```bash
# Combined model, embeddings already computed
uv run toxfam predict proteins.tsv \
  --model-dir model/model_output/combined_run \
  --embeddings data/processed/embeddings.h5 \
  -o out/preds.tsv

# Combined + standard fallback → out/preds_combined.tsv and out/preds_standard.tsv
uv run toxfam predict proteins.tsv \
  --model-dir model/model_output/combined_run \
  --standard-model-dir model/model_output/standard_run \
  --embeddings data/processed/embeddings.h5 \
  -o out/preds.tsv

# Standard model, embed sequences from scratch (no --embeddings)
uv run toxfam predict proteins.tsv \
  --model-dir model/model_output/standard_run \
  -o out/preds.tsv
```
