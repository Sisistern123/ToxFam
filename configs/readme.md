# Configuration & Architectures

Training is configured via YAML files. The **`training_strategy`** parameter determines which neural network architecture and training procedure is used.

## Available Configs

| Config File | Strategy | Description |
|------------|----------|-------------|
| `standard.yaml` | `standard` | 38-class family classification |
| `binary.yaml` | `binary` | Direct binary toxic/non-toxic |
| `binary_cpp.yaml` | `binary` | Binary with CPP physicochemical features |
| `combined.yaml` | `combined` | Two-branch (embeddings + taxonomy) |
| `hierarchical_existing.yaml` | `hierarchical` | Two-stage: family → binary (frozen backbone) |
| `hierarchical_unfrozen.yaml` | `hierarchical` | Two-stage with fine-tuned backbone |
| `multitask.yaml` | `multitask` | Joint family + binary classification |

---

## Global Parameters

Parameters shared across all strategies.

```yaml
# Data paths
input_csv: "data/processed/training_data.csv"
h5_paths_glob: "data/processed/embeddings.h5"
tax_h5_path: null                    # Path to taxonomy HDF5 (combined strategy only)
cpp_h5_path: null                    # Path to CPP features HDF5 (optional)
output_dir: "model/model_output/experiment_name"

# Model architecture
hidden_dims: [256, 256]              # Hidden layer sizes
dropout: 0.5                         # Dropout rate
embedding_dim: 1024                  # ProtT5 embedding dimension
tax_dim: 56                          # Taxonomy vector dimension (combined only)
cpp_dim: 100                         # CPP feature dimension (optional)

# Training
batch_size: 64
num_epochs: 200
learning_rate: 0.0001
early_stopping_patience: 10

# Loss function
loss_function: "cross_entropy"       # "cross_entropy" or "focal"
focal_gamma: 2.0                     # Focal loss gamma (only when loss_function: "focal")
```

---

## Strategies & Architectures

### Standard (`training_strategy: "standard"`)

Baseline 38-class family classifier using ProtT5 embeddings only.

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

### Binary (`training_strategy: "binary"`)

Direct 2-class toxic/non-toxic classifier. Same `ModularMLP` architecture but with only 2 output classes. Family labels are mapped to binary labels at runtime.

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

**Recommended for toxic/non-toxic prediction.**

### Combined (`training_strategy: "combined"`)

Two-branch architecture processing ProtT5 embeddings and taxonomy vectors separately before concatenation. Requires `tax_h5_path`.

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

### Hierarchical (`training_strategy: "hierarchical"`)

Two-stage training that transfers family-level knowledge to binary classification.

**Stage 1:** Train `ModularMLP` on all 38 families (including nontox).
**Stage 2:** Extract projector as frozen backbone, add binary head (`HierarchicalMLP`).

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

Extra parameters:

```yaml
stage2_freeze_backbone: true      # Freeze projector weights in Stage 2
stage2_learning_rate: 0.00001     # Lower LR for Stage 2
stage2_hidden_dim: 64             # Binary head hidden dimension
stage1_model_path: null           # Optional: skip Stage 1 by providing pretrained model
```

### Multi-Task (`training_strategy: "multitask"`)

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

Loss: `L = alpha * L_family + beta * L_binary`

Extra parameters:

```yaml
multitask_family_weight: 1.0      # Weight for family classification loss (alpha)
multitask_binary_weight: 1.0      # Weight for binary classification loss (beta)
```

---

## Loss Functions

| Loss | Config Value | Description |
|------|-------------|-------------|
| Cross-Entropy | `loss_function: "cross_entropy"` | Standard weighted CE (default) |
| Focal Loss | `loss_function: "focal"` | Down-weights easy examples: `FL(p_t) = -alpha_t(1-p_t)^gamma log(p_t)` |

All strategies use inverse-frequency class weighting to handle the 18:1 nontox:toxic imbalance.

---

## Full Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **`training_strategy`** | String | `"standard"` | `"standard"`, `"binary"`, `"combined"`, `"hierarchical"`, `"multitask"` |
| `input_csv` | String | — | Path to training CSV with `Split` column |
| `h5_paths_glob` | String | — | Glob pattern for embedding HDF5 files |
| `tax_h5_path` | String | `null` | Path to taxonomy HDF5 (required for combined) |
| `cpp_h5_path` | String | `null` | Path to CPP features HDF5 (optional) |
| `output_dir` | String | — | Output directory for model, plots, metrics |
| `hidden_dims` | List[int] | `[256, 256]` | Hidden layer sizes |
| `dropout` | Float | `0.5` | Dropout rate |
| `embedding_dim` | Int | `1024` | Input embedding dimension |
| `tax_dim` | Int | `56` | Taxonomy vector dimension |
| `cpp_dim` | Int | `100` | CPP feature dimension |
| `batch_size` | Int | `64` | Training batch size |
| `num_epochs` | Int | `200` | Maximum training epochs |
| `learning_rate` | Float | `0.0001` | Learning rate |
| `early_stopping_patience` | Int | `10` | Epochs without improvement before stopping |
| `loss_function` | String | `"cross_entropy"` | `"cross_entropy"` or `"focal"` |
| `focal_gamma` | Float | `2.0` | Focal loss gamma parameter |
| `stage2_freeze_backbone` | Bool | `true` | Freeze projector in hierarchical Stage 2 |
| `stage2_learning_rate` | Float | `null` | Stage 2 LR (defaults to `learning_rate / 10`) |
| `stage2_hidden_dim` | Int | `64` | Hierarchical Stage 2 head hidden dim |
| `stage1_model_path` | String | `null` | Path to pretrained Stage 1 model |
| `multitask_family_weight` | Float | `1.0` | Multi-task family loss weight (alpha) |
| `multitask_binary_weight` | Float | `1.0` | Multi-task binary loss weight (beta) |
| `n_folds` | Int | `1` | k-Fold CV folds (1 = no CV, >1 = k-fold) |
