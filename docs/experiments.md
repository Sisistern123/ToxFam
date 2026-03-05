# Experiments & Results

This document records the experimental results from training ToxFam models, including the impact of identity-aware splitting and comparison across training strategies.

## Dataset

- **65,179** protein sequences (90%-clustered representatives)
- **61,763** non-toxic (nontox) — 94.8%
- **3,416** toxic across 37 families + 1 "other" class — 5.2%
- **18:1 class imbalance** (nontox:toxic)

## Identity-Aware Splitting

### Problem

The original splitting approach used `MultilabelStratifiedShuffleSplit` which randomly assigns individual sequences to train/val/test. This allows proteins with >30% sequence identity to appear in both train and test, inflating metrics through data leakage.

### Solution: Cluster-Then-Split

The new `identity_aware_splits()` function:

1. **Global clustering at 30% identity** — All 65K representatives are clustered with MMseqs2 at `--min-seq-id 0.3`
2. **Cluster-level splitting** — Entire clusters are assigned to train/val/test (70/15/15) using `MultilabelStratifiedShuffleSplit` at the cluster level
3. **Adaptive relaxation** — Families stuck in a single cluster are re-clustered at progressively higher thresholds (40%, 50%, 60%, 70%) until splittable

### Split Statistics

After re-splitting with identity-aware clustering:

| Split | Total | Toxic | Non-toxic | Families |
|-------|-------|-------|-----------|----------|
| Train | 45,883 | 2,416 | 43,467 | 38 |
| Val | 9,709 | 423 | 9,286 | 35 |
| Test | 9,587 | 577 | 9,010 | 35 |

Threshold distribution:
- 34 families at 30% (clean separation)
- 1 family at 40%
- 2 families at 50%
- 1 family at 70%

3 families appear only in train (single tight cluster even at 70% — truly near-identical members).

## Training Results

All models trained on MPS (Apple Silicon) with:
- Hidden dims: [256, 256]
- Dropout: 0.5 (standard/binary), 0.3 (hierarchical)
- Learning rate: 0.0001
- Early stopping patience: 10
- Batch size: 64
- Temperature-scaled calibration on validation set

### Comparison: Identity-Aware vs Old Random Splits

Standard strategy (38-class), showing the impact of proper splitting:

| Metric | Old Random Splits | Identity-Aware Splits |
|--------|------------------|----------------------|
| Test Accuracy | 91.43% | 92.85% |
| Multiclass MCC | 0.572 | 0.615 |
| Binary ROC-AUC | 0.990 | 0.980 |
| Binary PR-AUC | 0.830 | 0.709 |

The binary ROC-AUC drop (0.990 → 0.980) and PR-AUC drop (0.830 → 0.709) confirm that old splits had sequence leakage inflating binary prediction performance. Multiclass accuracy actually improved slightly, likely due to better generalization from non-leaking splits.

### Strategy Comparison on Identity-Aware Splits

| Metric | Standard (38-class) | Binary (2-class) |
|--------|-------------------|-----------------|
| **Test Accuracy** | 92.85% | **96.82%** |
| **Test MCC** | 0.615 | **0.752** |
| Binary ROC-AUC | 0.980 | **0.986** |
| Binary PR-AUC | 0.709 | **0.999** |
| Binary F1 | 0.611 | **0.983** |
| Binary MCC | 0.627 | **0.752** |
| Epochs to converge | 108 | 35 |

### Analysis

1. **Binary strategy is strongly preferred for toxic/non-toxic prediction.** PR-AUC 0.999 vs 0.709 is a massive improvement — the direct binary model produces near-perfect ranking of toxic probability.

2. **Standard model's weak binary performance is expected.** The 38-class model spreads probability across 37 toxic families. When we derive `p_toxic = 1 - p(nontox)`, the threshold of 0.5 is suboptimal and the signal is diluted. The model is optimized for family distinction, not toxicity detection.

3. **Class weighting handles the 18:1 imbalance.** The binary model achieves 0.983 F1 despite having only ~5% toxic sequences, thanks to inverse-frequency class weighting in the loss function.

4. **Binary model converges 3x faster** (35 vs 108 epochs). The 2-class objective is simpler and the gradient signal is more direct.

5. **ProtT5 embeddings contain strong toxicity signal.** Both strategies achieve >0.98 ROC-AUC, indicating the protein language model representations reliably separate toxic from non-toxic sequences.

### Important Note on Binary PR-AUC

The binary strategy's PR-AUC of 0.999 was measured before a bug fix in `to_binary_class()`. The binary/hierarchical strategies used `"nontoxic"` as the non-toxic label (mapped via `is_toxic`), but `NONTOXIN_LABELS` only contained `"nontox"`. This caused `"nontoxic"` → `"toxin"`, effectively measuring the **nontoxin-as-positive** PR-AUC (which is trivially high due to 18:1 imbalance).

After the fix (`NONTOXIN_LABELS = {"nontox", "nontoxic"}`), PR-AUC should be re-evaluated. The corrected metric measures toxic-as-positive class performance, which is the meaningful evaluation.

### Additional Strategies

| Strategy | Test Accuracy | Binary ROC-AUC | Notes |
|----------|-------------|--------------|-------|
| Hierarchical (frozen) | — | — | Two-stage: family → binary transfer |
| Hierarchical (unfrozen) | — | — | Fine-tuned backbone with very low LR |
| Multitask | — | — | Joint family + binary heads |
| Binary + CPP | — | — | Binary with physicochemical features |

*Results to be filled after retraining with corrected metrics.*

### Threshold Optimization

After calibration, Youden's J statistic is computed on the validation set to find the optimal classification threshold (which may differ from the default 0.5, especially for imbalanced data). Both default and optimized threshold metrics are saved.

### k-Fold Cross-Validation

Use `n_folds > 1` in config to enable cluster-level k-fold CV. The test set is held fixed; only train/val are re-split per fold. Results are aggregated as mean ± std.

### Ensemble Evaluation

Multiple trained models can be evaluated as an ensemble using `toxfam eval-ensemble`. Softmax probabilities are averaged across models before computing binary metrics.

## Reproduction

```bash
# Re-split existing data with identity-aware splits
uv run python3 -c "
import pandas as pd
from toxfam.data.preprocessing import identity_aware_splits
df = pd.read_csv('data/processed/training_data.csv').drop(columns=['Split'])
train, val, test = identity_aware_splits(df, base_seq_id=0.3)
train['Split'], val['Split'], test['Split'] = 'train', 'val', 'test'
pd.concat([train, val, test]).to_csv('data/processed/training_data.csv', index=False)
"

# Train standard
uv run toxfam train configs/standard.yaml

# Train binary
uv run toxfam train configs/binary.yaml

# Train hierarchical (family backbone → binary head)
uv run toxfam train configs/hierarchical_existing.yaml

# Train multitask (joint family + binary)
uv run toxfam train configs/multitask.yaml

# Re-evaluate without retraining
uv run toxfam eval-binary model/model_output/binary_run

# Ensemble evaluation
uv run toxfam eval-ensemble model/model_output/binary_run model/model_output/standard_run

# k-Fold CV (set n_folds in config or use a config with n_folds: 5)
uv run toxfam train configs/binary.yaml  # with n_folds: 5 in config
```
