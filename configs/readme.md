# Configuration & Architectures

This project uses a central `config.yaml` to control training logic. The **`training_strategy`** parameter determines which neural network architecture is built and how data flows through it.

## 1\. Global Parameters

Parameters applied to all models.

```yaml
# 1. DATA PATHS
input_csv: "data/processed/training_data.csv"
h5_paths_glob: "data/processed/embeddings/training_data.h5"
tax_h5_path: "data/processed/taxonomy/binary_taxonomy_vectors.h5"
output_dir: "model/model_output/experiment_name"

# 2. MODEL SPECS
hidden_dims: [256, 256]    # Size of hidden backbone layers
dropout: 0.5
embedding_dim: 1024        # ProtT5/ESM input size
tax_dim: 56                # Taxonomy vector input size

# 3. TRAINING
use_focal_loss: true       # True = Focal Loss, False = CrossEntropy
focal_loss_gamma: 2.0
batch_size: 64
num_epochs: 200
learning_rate: 0.0001
early_stopping_patience: 10
```

-----

## 2\. Strategies & Architectures

### Strategy A: Standard

**Key:** `training_strategy: "standard"`

The baseline model. It learns to predict protein families purely from protein language model embeddings. Taxonomy data is ignored.

```yaml
training_strategy: "standard"
```

**Architecture Diagram:**

```text
[ Input: Embeddings ] (1024)
         │
         ▼
[   Projector Layer  ] ── maps 1024 → 256
         │
         ▼
[   Backbone Layer 1 ] (256 + ReLU + Dropout)
         │
         ▼
[   Backbone Layer 2 ] (256 + ReLU + Dropout)
         │
         ▼
[  Classification Head ] ── Output: N Classes
```

-----

### Strategy B: Combined

**Key:** `training_strategy: "combined"`

A multi-modal "Branched" model. Both Embeddings and Taxonomy are fed in simultaneously. They are processed by separate branches and then concatenated. This usually yields the highest accuracy but requires taxonomy data during inference.

```yaml
training_strategy: "combined"
```

**Architecture Diagram:**

```text
[Input: Embeddings] (1024)       [Input: Taxonomy] (56)
         │                                │
         ▼                                ▼
[   Embed Branch   ]             [   Tax Branch   ]
(Linear → 256)                   (Linear → 8)
         │                                │
         └───────────────┬────────────────┘
                         │
                         ▼
                 [ Concatenation ] (256 + 8 = 264)
                         │
                         ▼
                 [   Joint Head   ]
                 (Linear → 256 → Classes)
```

-----

## 3\. Parameter Reference

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`training_strategy`** | String | Options: `"standard"`, `"combined"`. |
| `input_csv` | String | Path to metadata CSV. Must contain `Split` column. |
| `h5_paths_glob` | String | Glob pattern for embedding HDF5 files. |
| `tax_h5_path` | String | Path to taxonomy HDF5 file (Required for Combined). |
| `hidden_dims` | List[int] | Hidden layer sizes, e.g., `[256, 256]`. |
| `use_focal_loss` | Bool | `true` uses Focal Loss (for imbalance), `false` uses CrossEntropy. |
| `focal_loss_gamma`| Float | Strength of Focal Loss. Default `2.0`. |
| `num_epochs` | Int | Max epochs for main training phase. |
| `early_stopping_patience`| Int | Stop if validation metric stalls for N epochs. |
