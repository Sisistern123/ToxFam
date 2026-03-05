# ToxFam Architecture Overview

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ DATA PIPELINE                                                       │
│                                                                     │
│  Raw TSVs (data/raw/)                                              │
│      │                                                              │
│      ▼                                                              │
│  toxfam preprocess                                                  │
│  ├── Normalize family labels                                        │
│  ├── SignalP6 signal peptide removal (cached)                      │
│  ├── Per-family MMseqs2 clustering (90% identity)                  │
│  └── Identity-aware train/val/test splitting (30% identity)        │
│      ├── Cluster-level stratified assignment                        │
│      ├── Post-assignment rebalancing                                │
│      └── Adaptive relaxation for under-represented families        │
│      │                                                              │
│      ▼                                                              │
│  data/processed/training_data.csv                                  │
│  data/intermediate/ (FASTA, clusters, caches)                      │
│                                                                     │
│  toxfam embed                                                       │
│      │                                                              │
│      ▼                                                              │
│  data/processed/embeddings.h5  (ProtT5, 1024-dim per protein)     │
│                                                                     │
│  toxfam taxonomy (optional)                                         │
│      ▼                                                              │
│  data/intermediate/taxonomy/binary_taxonomy_vectors.h5 (56-dim)    │
│                                                                     │
│  toxfam cpp (optional)                                              │
│      ▼                                                              │
│  data/intermediate/cpp/cpp_features.h5 (100-dim)                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ TRAINING PIPELINE                                                   │
│                                                                     │
│  toxfam train configs/<strategy>.yaml                              │
│      │                                                              │
│      ├── Load config (TrainConfig from YAML)                       │
│      ├── Save config copy to output_dir                            │
│      ├── Load data splits + embeddings (+ taxonomy/CPP if set)     │
│      ├── Compute class weights (inverse-frequency)                 │
│      ├── Dispatch to strategy:                                      │
│      │   ├── standard  → ModularMLP (38 classes)                   │
│      │   ├── binary    → ModularMLP (2 classes)                    │
│      │   ├── combined  → MultiInputMLP (embed + taxonomy)          │
│      │   ├── hierarchical → Stage1 family + Stage2 binary          │
│      │   └── multitask → MultiTaskMLP (family + binary heads)      │
│      │                                                              │
│      ├── Train with early stopping + wandb logging (optional)      │
│      ├── Uncalibrated evaluation (val + test)                      │
│      ├── Temperature scaling calibration on val set                │
│      ├── Calibrated evaluation (val + test)                        │
│      │                                                              │
│      ├── Binary metrics pipeline:                                   │
│      │   ├── Compute on validation set                             │
│      │   ├── Optimize threshold (Youden's J on val)                │
│      │   ├── Compute on test set (default threshold)               │
│      │   ├── Compute on test set (optimized threshold)             │
│      │   └── Multitask: evaluate binary head directly              │
│      │                                                              │
│      └── Save: model, metrics, plots, predictions, config          │
│                                                                     │
│  k-Fold CV (if n_folds > 1):                                      │
│      ├── Fixed test set across all folds                           │
│      ├── Cluster-level train/val re-splitting per fold             │
│      ├── Full training pipeline per fold                           │
│      └── Aggregate metrics (mean ± std)                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ EVALUATION PIPELINE                                                 │
│                                                                     │
│  toxfam eval-binary <model_dir>                                    │
│      └── Re-compute binary metrics without retraining              │
│                                                                     │
│  toxfam eval-ensemble <model_dir1> <model_dir2> ...                │
│      └── Average softmax probabilities → binary metrics            │
│                                                                     │
│  toxfam eval-test [--model-dir <path>]                             │
│      └── Compare NN vs HBI sequence similarity                     │
│                                                                     │
│  toxfam profile-data --input-csv <csv> [--h5-path <h5>]           │
│      └── Data quality profiling for bias detection                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Model Architectures

### Standard / Binary — `ModularMLP`

```
Input (1024 or 1024+CPP)
    │
    ▼
Projector: Linear → ReLU → Dropout
    │
    ▼
Backbone:  Linear → ReLU → Dropout (repeated)
    │
    ▼
Head:      Linear → N classes (38 for standard, 2 for binary)
```

### Combined — `MultiInputMLP`

```
Embeddings (1024)          Taxonomy (56)
    │                          │
    │                    Tax Branch:
    │                    Linear(56→8) → ReLU → Dropout
    │                          │
    └──────────┬───────────────┘
               │
         Concatenation (1032)
               │
         Joint Backbone
               │
         Family Head (38)
```

### Hierarchical — `HierarchicalMLP`

```
Stage 1 (Training):              Stage 2 (Transfer):
  Embedding (1024)                 Embedding (1024)
       │                                │
   Projector ──── weights ────→  Frozen Projector (256)
       │                                │
   Backbone                        Binary Head
       │                         Linear(256→64) → ReLU
   Family Head (38)                     │
                                   Linear(64→2)
```

### Multitask — `MultiTaskMLP`

```
Embedding (1024)
    │
Projector: Linear → ReLU → Dropout
    │
Backbone:  Linear → ReLU → Dropout
    │
 ┌──┴──┐
 │     │
 ▼     ▼
Family Binary
(38)   (2)

Loss = α·L_family + β·L_binary
Both heads use weighted cross-entropy.
```

## Key Design Decisions

1. **Identity-aware splitting** prevents sequence leakage between train/test at 30% identity
2. **Post-split rebalancing** ensures families with ≥10 members have ≥50% in training
3. **Temperature scaling** calibrates model confidence on the validation set
4. **Threshold optimization** finds the optimal binary classification threshold via Youden's J
5. **All strategies** automatically compute binary toxic/non-toxic metrics
6. **Config-driven** — a single YAML file controls the entire pipeline
7. **Cluster-level k-fold CV** respects identity constraints across folds
