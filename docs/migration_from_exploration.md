# Migration from Exploration Branch

This document describes the migration of features from the `Exploration` branch onto `main`'s refactored codebase, completed on 2026-03-31.

## Summary

All mature research features from Exploration were ported onto main's refactored foundation (6 commits: taxonomy overhaul, training pipeline modernization, CLI improvements, shared modules). Main's structure and patterns were preserved; Exploration's features were adapted to fit.

## What Was Ported

### PR 1: Binary Metrics + Evaluation Pipeline
**What:** Score-based binary toxic/nontoxin metrics with threshold optimization.
**Why:** Main could only report multiclass family metrics. Binary classification is the primary research question.

- `calculate_binary_metrics_with_scores()` — ROC-AUC, PR-AUC, F1, MCC from probability scores
- `find_optimal_threshold()` — Youden's J and F1 methods
- `NONTOXIN_LABELS` expanded to `{"nontox", "nontoxic"}`
- Binary metrics pipeline in orchestrator (auto-runs after every training)
- `plot_binary_roc()` and `plot_binary_pr()` plotting functions
- `eval-binary` CLI command for post-hoc binary evaluation

**Tests:** 11 (TestToBinaryClass, TestNontoxinLabels, TestCalculateBinaryMetricsWithScores, TestFindOptimalThreshold)

### PR 2: Binary Strategy + Config Expansion
**What:** Direct 2-class toxic/nontoxin MLP training strategy.
**Why:** Direct binary classification achieves higher PR-AUC than derived-from-family metrics because the model optimizes directly for the binary task.

- `TrainConfig` expanded: 5 strategies (standard, combined, binary, hierarchical, multitask)
- `effective_embedding_dim` property for auxiliary feature composition
- Config fields: hierarchical (stage1/2), multitask weights, CPP/HBI, length, venom indicator, n_folds, split_seq_id
- `run_binary_strategy()` in strategies.py
- Orchestrator handles binary label derivation and strategy dispatch
- `binary.yaml` config

**Tests:** 23 (TestStrategyTypes, TestExtraFieldsIgnored, TestFocalLoss, TestHierarchicalFields, TestMultitaskFields, TestCrossValidation, TestEffectiveEmbeddingDim, TestFromYaml, TestFieldValidation)

### PR 3: Hierarchical + Multitask Strategies + k-Fold CV
**What:** Two advanced training approaches + cross-validation.
**Why:** Hierarchical leverages family-level knowledge for binary prediction via transfer learning. Multitask trains both tasks simultaneously. k-fold CV provides robust performance estimates.

- `HierarchicalMLP` — frozen/unfrozen projector from Stage 1 + binary head
- `MultiTaskMLP` — shared backbone with dual family + binary heads
- `_MultiTaskFamilyWrapper` / `_MultiTaskBinaryWrapper` for evaluation
- `training/hierarchical.py` — two-stage training logic
- `training/cross_validation.py` — k-fold with cluster-level stratified splitting
- `hierarchical_existing.yaml`, `hierarchical_unfrozen.yaml`, `multitask.yaml` configs

**Tests:** 19 (TestHierarchicalMLP: shapes, gradient freezing, weight loading, backward pass; TestMultiTaskMLP: shapes, shared backbone; TestAggregateFoldMetrics)

### PR 4: Auxiliary Features (CPP, HBI, Counterparts)
**What:** Framework for concatenating auxiliary features to ProtT5 embeddings.
**Why:** Complementary features boost binary classification. Best combined result on Exploration: MCC 0.774, ROC-AUC 0.990.

- `ToxDataset` extended: CPP, HBI, length, venom indicator concatenation
- `_extra_dataset_kwargs()` in orchestrator wires config → dataset
- `cpp_features.py` — CPP generation via AAanalysis
- `hbi_features.py` — HBI feature computation via MMseqs2
- `cpp` CLI command
- `binary_cpp.yaml` config (when CPP features are available)

### PR 5: Identity-Aware Splits
**What:** MMseqs2-based 30% identity clustering for train/val/test splitting.
**Why:** Prevents sequence leakage between splits — critical for honest evaluation.

- `identity_aware_splits()` — global clustering → cluster-level stratified split → rebalancing
- `_rebalance_splits()` — moves smallest clusters to ensure family representation
- `split_seq_id` config field

**Tests:** 4 (TestRebalanceSplits: returns three sets, no overlap, preserves all clusters, never empties split)

### PR 6: Evaluation Toolkit
**What:** Ensemble evaluation, data quality profiling, HBI binary baseline.
**Why:** Completes the evaluation framework for publication.

- `ensemble.py` — average softmax across multiple calibrated models
- `data_quality.py` — class distribution, organism diversity, sequence length, embedding similarity analysis
- `hbi_binary_baseline.py` — HBI binary evaluation baseline
- `eval-ensemble` and `profile-data` CLI commands

**Tests:** 1 (ensemble module import)

## What Was NOT Ported

- **Handcrafted features** (Atchley factors, cysteine patterns) — proven redundant with ProtT5 on Exploration
- **`hierarchical_preprocessing.py`** — Phase 2 data assembly from new XML sources, not needed for current dataset
- **External tool integrations** (ToxinPred2/3, TOXIFY) — require external dependencies, separate concern

## Architecture Decisions

1. **Main wins on structure:** Kept trainer.py's FocalLoss (has label_smoothing), optimizer/scheduler patterns, wandb integration (made optional). Exploration's separate losses.py not needed.
2. **effective_embedding_dim property:** Auto-calculates input dimension from base embedding + all auxiliary features. All model construction uses this property.
3. **Binary metrics auto-run:** Every strategy automatically computes binary metrics after training. No separate step needed.
4. **DataSelector preserved:** Main's pattern of wrapping DataLoaders with mode-specific selectors works for all 5 strategies.

## Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_architectures.py | 15 | All 4 model classes: shapes, gradients, transfer |
| test_binary_metrics.py | 11 | Binary metrics, threshold optimization, label mapping |
| test_config.py | 23 | All strategies, validation, effective_embedding_dim |
| test_cross_validation.py | 4 | Metric aggregation across folds |
| test_ensemble.py | 1 | Module import |
| test_identity_splits.py | 4 | Rebalancing logic |
| **Total** | **58** | |

## Benchmark Evidence

*To be filled after running benchmarks on the ported codebase:*

| Strategy | ROC-AUC | PR-AUC | F1 | MCC |
|----------|---------|--------|-----|-----|
| standard (baseline) | — | — | — | — |
| binary | — | — | — | — |
| hierarchical (frozen) | — | — | — | — |
| multitask | — | — | — | — |
| binary + CPP | — | — | — | — |

*Prior Exploration results (for reference):*

| Method | ROC-AUC | PR-AUC | MCC |
|--------|---------|--------|-----|
| NN augmented+CP | 0.990 | 0.892 | 0.774 |
| NN binary (vanilla) | 0.986 | 0.838 | 0.758 |
| HBI best-hit | 0.860 | 0.615 | 0.760 |
