# Configuration & Architectures

This project uses a central `config.yaml` to control training logic. The **`training_strategy`** parameter determines which neural network architecture is built and how data flows through it.

## 1\. Global Parameters

Parameters applied to all models.

```yaml
# 1. DATA PATHS
input_csv: "data/interm/training_data.csv"
h5_paths_glob: "data/protspace/training_data.h5"
tax_h5_path: "data/tax/binary_taxonomy_vectors.h5" 
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

### Strategy C: Pretrain & Finetune (Curriculum Learning)

**Key:** `training_strategy: "pretrain_finetune"`

A two-stage transfer learning approach.

1.  **Stage 1:** Train the backbone using **only Taxonomy** (Clean signal, easy task).
2.  **Stage 2:** Transfer the backbone weights to a new model. Swap the input layer for **Embeddings**. Finetune the model to map noisy embeddings to the learned structure.

<!-- end list -->

```yaml
training_strategy: "pretrain_finetune"
freeze_backbone: false      # If true, Backbone is locked in Stage 2
tax_epochs: 50              # Stage 1 duration
tax_lr: 0.001               # Stage 1 learning rate
```

**Architecture Diagram:**

**Step 1: Pre-training (Taxonomy)**

```text
[ Input: Taxonomy ] (56)
         │
         ▼
[   Projector A    ] ── maps 56 → 256  (Learned & Discarded)
         │
         ▼
┌──────────────────────┐
│  Shared Backbone     │ ◄─── Model learns valid family
│  (Layer 1 & 2)       │      groupings here.
└──────────────────────┘
         │
         ▼
[  Classification  ]
```

**Step 2: Fine-tuning (Embeddings)**

```text
[ Input: Embeddings ] (1024)
         │
         ▼
[   Projector B    ] ── maps 1024 → 256 (Learned from scratch)
         │
         ▼
┌──────────────────────┐
│  Shared Backbone     │ ◄─── WEIGHTS TRANSFERRED FROM STEP 1
│  (Layer 1 & 2)       │      (Can be Frozen or Tuned)
└──────────────────────┘
         │
         ▼
[  Classification  ]
```

-----

## 3\. Parameter Reference

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`training_strategy`** | String | Options: `"standard"`, `"combined"`, `"pretrain_finetune"`. |
| `input_csv` | String | Path to metadata CSV. Must contain `Split` column. |
| `h5_paths_glob` | String | Glob pattern for embedding HDF5 files. |
| `tax_h5_path` | String | Path to taxonomy HDF5 file (Required for Combined/Pretrain). |
| `hidden_dims` | List[int] | Hidden layer sizes, e.g., `[256, 256]`. |
| `use_focal_loss` | Bool | `true` uses Focal Loss (for imbalance), `false` uses CrossEntropy. |
| `focal_loss_gamma`| Float | Strength of Focal Loss. Default `2.0`. |
| `num_epochs` | Int | Max epochs for main training phase. |
| `early_stopping_patience`| Int | Stop if validation metric stalls for N epochs. |
| **Pretrain Specific** | | |
| `tax_epochs` | Int | Epochs for Stage 1 (Taxonomy). |
| `tax_lr` | Float | Learning rate for Stage 1. |
| `freeze_backbone` | Bool | If `true`, Stage 2 only trains the Projector layer. |