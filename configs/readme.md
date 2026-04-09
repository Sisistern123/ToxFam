# Configuration & Architectures

This project uses YAML configs to control training. The **`training_strategy`** parameter determines which neural network architecture is built and how data flows through it.

## 1. Global Parameters

```yaml
# DATA PATHS
input_csv: "data/processed/training_data.csv"
h5_paths_glob: "data/processed/embeddings.h5"
tax_h5_path: "data/processed/taxonomy_vectors.h5"
output_dir: "model/model_output/experiment_name"

# MODEL
hidden_dims: [256, 256]
dropout: 0.3
embedding_dim: 1024       # ProtT5 input size
tax_dim: 50                # Taxonomy vector size

# TRAINING
batch_size: 64
num_epochs: 200
learning_rate: 0.0001
early_stopping_patience: 10
early_stopping_metric: "mcc"  # "loss" or "mcc"
seed: 42

# OPTIMIZER
optimizer: "adamw"         # "adam" or "adamw"
weight_decay: 0.01

# LR SCHEDULER
lr_scheduler: "cosine"     # "none" or "cosine"
warmup_epochs: 5

# LOSS
use_focal_loss: false
focal_loss_gamma: 2.0
label_smoothing: 0.0
max_grad_norm: 1.0
```

---

## 2. Strategies & Architectures

### Strategy A: Standard

**Key:** `training_strategy: "standard"`

Baseline model predicting protein families from ProtT5 embeddings only.

```text
[ Embeddings ] (1024)
       │
       ▼
[ Projector ] ── maps 1024 → 256
       │
       ▼
[ Backbone ] (256 → 256, ReLU + Dropout)
       │
       ▼
[ Classification Head ] ── N classes
```

### Strategy B: Binary

**Key:** `training_strategy: "binary"`

Direct binary toxic/non-toxic prediction. Same `ModularMLP` architecture as standard but with 2 output classes.

### Strategy C: Combined

**Key:** `training_strategy: "combined"`

Two-branch model processing embeddings and taxonomy vectors separately, then concatenating them.

```text
[ Embeddings ] (1024)       [ Taxonomy ] (50)
       │                          │
       ▼                          ▼
[ Embed Branch ]           [ Tax Branch ]
  (Linear → 256)            (Linear → 8)
       │                          │
       └──────────┬───────────────┘
                  │
                  ▼
          [ Concatenation ] (264)
                  │
                  ▼
          [ Joint Head ] ── N classes
```

---

## 3. Parameter Reference

| Parameter | Type | Description |
|---|---|---|
| `training_strategy` | String | `"standard"`, `"binary"`, or `"combined"` |
| `input_csv` | String | Path to metadata CSV with `Split` column |
| `h5_paths_glob` | String | Glob pattern for embedding HDF5 files |
| `tax_h5_path` | String | Path to taxonomy HDF5 (required for combined) |
| `hidden_dims` | list[int] | Hidden layer sizes, e.g. `[256, 256]` |
| `dropout` | Float | Dropout rate, `[0, 1]` |
| `use_focal_loss` | Bool | `true` = Focal Loss, `false` = CrossEntropy |
| `focal_loss_gamma` | Float | Focal Loss strength (default `2.0`) |
| `label_smoothing` | Float | Label smoothing factor, `[0, 1)` |
| `optimizer` | String | `"adam"` or `"adamw"` |
| `weight_decay` | Float | Weight decay (default `0.01`) |
| `lr_scheduler` | String | `"none"` or `"cosine"` with warmup |
| `warmup_epochs` | Int | LR warmup epochs for cosine scheduler |
| `early_stopping_metric` | String | `"loss"` or `"mcc"` |
| `max_grad_norm` | Float/null | Gradient clipping threshold |
| `seed` | Int/null | Random seed for reproducibility |
