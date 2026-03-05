# Training Strategies

ToxFam supports five training strategies, selected via `training_strategy` in the YAML config. All strategies automatically produce binary toxic/non-toxic metrics alongside their primary task metrics.

## Strategy Overview

| Strategy | Classes | Architecture | Use Case |
|----------|---------|-------------|----------|
| `standard` | 38 families | `ModularMLP` | Family-level classification |
| `binary` | 2 (toxic/nontoxic) | `ModularMLP` | Direct binary prediction |
| `combined` | 38 families | `MultiInputMLP` | Family classification with taxonomy features |
| `hierarchical` | Stage 1: 38, Stage 2: 2 | `ModularMLP` → `HierarchicalMLP` | Transfer learning: family knowledge → binary |
| `multitask` | 38 + 2 (joint) | `MultiTaskMLP` | Simultaneous family + binary learning |

## Standard Strategy

**Config:** `training_strategy: "standard"`

Baseline multiclass family classifier. Learns to predict which of 38 protein families a sequence belongs to.

```
[ ProtT5 Embedding ] (1024)
         │
         ▼
[   Projector   ] Linear(1024 → 256) + ReLU + Dropout
         │
         ▼
[   Backbone    ] Linear(256 → 256) + ReLU + Dropout
         │
         ▼
[  Family Head  ] Linear(256 → 38)
```

Binary toxic/non-toxic probability is derived post-hoc as `p_toxic = 1 - softmax[nontox_idx]`.

## Binary Strategy

**Config:** `training_strategy: "binary"`

Direct 2-class toxic/non-toxic classifier. Uses the same `ModularMLP` architecture but with only 2 output classes. Family labels are mapped to binary labels at runtime: any toxic family → `"toxic"`, `"nontox"` → `"nontoxic"`.

```
[ ProtT5 Embedding ] (1024)
         │
         ▼
[   Projector   ] Linear(1024 → 256) + ReLU + Dropout
         │
         ▼
[   Backbone    ] Linear(256 → 256) + ReLU + Dropout
         │
         ▼
[ Binary Head   ] Linear(256 → 2)
```

This is the **recommended strategy for toxic/non-toxic prediction** — it achieves PR-AUC 0.999 vs 0.709 for the standard approach.

## Combined Strategy

**Config:** `training_strategy: "combined"`

Two-branch architecture that processes ProtT5 embeddings and taxonomy binary vectors separately before concatenation.

```
[ Embeddings ] (1024)        [ Taxonomy ] (56)
       │                           │
       ▼                           ▼
[ Embed Branch ]            [ Tax Branch ]
  (passthrough)             Linear(56 → 8) + ReLU
       │                           │
       └──────────┬────────────────┘
                  │
                  ▼
          [ Concatenation ] (1024 + 8 = 1032)
                  │
                  ▼
          [  Joint Backbone  ]
          Linear → 256 → 256
                  │
                  ▼
          [ Family Head ] (38 classes)
```

Requires taxonomy vectors (`tax_h5_path` in config).

## Hierarchical Strategy

**Config:** `training_strategy: "hierarchical"`

Two-stage training that transfers family-level knowledge to binary classification:

**Stage 1 — Family Classification:**
Trains a standard `ModularMLP` on all 38 families (including nontox). This teaches the model family-level structural/functional features.

**Stage 2 — Binary Classification:**
Extracts Stage 1's projector as a frozen backbone, adds a new binary classification head (`HierarchicalMLP`), and trains only the head on toxic/nontoxic labels.

```
Stage 1:                          Stage 2:
[ Embedding ] (1024)              [ Embedding ] (1024)
       │                                 │
       ▼                                 ▼
[  Projector  ] ─── trained ──→  [ Frozen Projector ] (256)
       │                                 │
       ▼                                 ▼
[  Backbone   ]                  [  Binary Head  ]
       │                         Linear(256 → 64) + ReLU
       ▼                                 │
[ Family Head ] (38)                     ▼
                                 Linear(64 → 2)
```

Key config options:
- `stage2_freeze_backbone: true` — freeze projector weights in Stage 2
- `stage2_learning_rate: 0.00001` — lower LR for Stage 2
- `stage2_hidden_dim: 64` — binary head hidden dimension

Use `configs/hierarchical_existing.yaml` to run on the standard training data (derives `is_toxic` from family labels at runtime).

## Multi-Task Strategy

**Config:** `training_strategy: "multitask"`

Joint training with shared backbone producing both family and binary predictions simultaneously.

```
[ ProtT5 Embedding ] (1024)
         │
         ▼
[   Projector   ] Linear(1024 → 256) + ReLU + Dropout
         │
         ▼
[   Backbone    ] Linear(256 → 256) + ReLU + Dropout
         │
    ┌────┴────┐
    ▼         ▼
[ Family ] [ Binary ]
  (38)      (2)
```

Loss: `L = α * L_family + β * L_binary`

Config options:
- `multitask_family_weight: 1.0` — weight for family classification loss (α)
- `multitask_binary_weight: 1.0` — weight for binary classification loss (β)

## Loss Functions

All strategies support configurable loss functions:

| Loss | Config | Description |
|------|--------|-------------|
| Cross-Entropy | `loss_function: "cross_entropy"` | Standard weighted CE (default) |
| Focal Loss | `loss_function: "focal"` | Down-weights easy examples: `FL(p_t) = -α_t(1-p_t)^γ log(p_t)` |

Focal loss config: `focal_gamma: 2.0` (higher γ = more focus on hard examples). With `γ=0`, focal loss reduces to weighted cross-entropy.

## Binary Evaluation

All strategies automatically compute binary toxic/non-toxic metrics on the test set after training:

- **ROC-AUC** — Area under receiver operating characteristic curve
- **PR-AUC** — Area under precision-recall curve (important for imbalanced data)
- **F1 Score** — Harmonic mean of precision and recall
- **MCC** — Matthews correlation coefficient
- **Accuracy** — Overall classification accuracy

For multiclass strategies (standard, combined), binary probability is derived as `p_toxic = 1 - softmax[nontox_idx]`. For binary/hierarchical strategies, the toxic class probability is used directly.

Results are saved to `metrics/binary_test_calibrated_metrics.json` with ROC and PR curve plots.

## Threshold Optimization

After calibration, the pipeline automatically optimizes the binary classification threshold on the validation set using Youden's J statistic (maximizing TPR - FPR). Both default (0.5) and optimized threshold metrics are computed on the test set.

Available methods via `find_optimal_threshold()`:
- **`youden`** — Maximize Youden's J (TPR - FPR)
- **`f1`** — Maximize F1 score
- **`target_precision`** — Find threshold achieving target precision with maximum recall

Results are saved to `metrics/threshold_optimization.json`.

## k-Fold Cross-Validation

Set `n_folds > 1` in the training config to enable cluster-level k-fold CV:

```yaml
n_folds: 5  # default: 1 (no CV)
```

Design:
- **Test set is fixed** across all folds (the identity-aware test set)
- Only train/val are re-split per fold at the cluster level
- Each fold trains from scratch (no warm-starting)
- Results saved to `output_dir/kfold/fold_N/` per fold + `output_dir/kfold/summary.json`

## Ensemble Evaluation

Evaluate multiple trained models as an ensemble:

```bash
uv run toxfam eval-ensemble model/model_output/run1 model/model_output/run2
```

Methods:
- **`mean`** (default) — Average softmax probabilities across models
- **`vote`** — Majority vote on predicted classes

## Re-evaluation Without Retraining

Re-compute binary metrics for a saved model:

```bash
uv run toxfam eval-binary model/model_output/binary_run
```

This loads the calibrated model and config from the model directory and re-runs the binary evaluation pipeline.

## Data Quality Profiling

Profile training data for potential biases:

```bash
uv run toxfam profile-data --input-csv data/processed/training_data.csv
```

Reports class distribution, organism diversity, sequence length distributions, and optional embedding similarity analysis.
