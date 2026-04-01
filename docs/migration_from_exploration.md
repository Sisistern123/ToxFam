# Migration from Exploration Branch

This document describes the migration of features from the `Exploration` branch onto `main`'s refactored codebase, completed 2026-04-01.

## Summary

All mature research features from Exploration were ported onto main's refactored foundation. Main's structure was preserved — including the recent additions of `ModelConfig` for deterministic model reconstruction, `runner.py` for unified evaluation, and the `eval hbi/model/compare` CLI pattern. Exploration's features were adapted to fit alongside these.

**Metrics API:** Both APIs coexist — `MetricsResult` dataclass for multiclass metrics (used by `runner.py`), and dict-based returns for score-based binary metrics and threshold optimization (new).

## What Was Ported

### PR 1: Foundation — wandb optional, shared modules, test infrastructure
**What:** Make wandb optional, port shared utility modules, establish test suite.
**Why:** Training crashed without wandb. Shared modules are dependencies for all subsequent PRs. Test suite goes from 0 to ~55 tests.

- wandb optional across training pipeline (try/except import, graceful login failure)
- `device.py` — canonical `get_device()` (cuda > mps > cpu), unified across all modules
- `normalization.py` — shared family label normalization
- `_fasta.py` — `read_fasta_as_dict` + `write_fasta` with MD5 skip
- `xml_parser.py` — UniProt XML parsing
- `label_validation.py` — MMseqs2-based family validation
- Test infrastructure: `conftest.py` shared fixtures + tests for device, fasta, normalization, paths, xml_parser, calibration, dataset, losses
- Gitignore cleanup

### PR 2: Binary metrics + binary training strategy
**What:** Score-based binary metrics pipeline + direct 2-class MLP.
**Why:** Main could only report multiclass family metrics. Binary toxic/nontox is the primary research question. Direct binary achieves **+5% MCC** over standard.

- `calculate_binary_metrics_with_scores()` — ROC-AUC, PR-AUC, F1, MCC from probability scores (dict return, coexists with `MetricsResult`)
- `find_optimal_threshold()` — Youden's J, F1, target precision methods (dict return)
- `NONTOXIN_LABELS` expanded to `{"nontox", "nontoxic", "nontoxin"}`
- Binary metrics pipeline in orchestrator (auto-runs after every strategy)
- `eval-binary` CLI command (coexists with main's `eval hbi/model/compare` subcommands)
- Binary ROC and PR curve plotting
- `TrainConfig` expanded: 5 strategies, `effective_embedding_dim` property, all config fields
- `run_binary_strategy()` + binary label derivation in orchestrator
- `binary.yaml` config

**Benchmark evidence:**

| Strategy | Test ROC-AUC | Test PR-AUC | Test MCC (t=0.5) |
|----------|-------------|------------|-----------------|
| Standard (baseline) | 0.983 | 0.894 | 0.780 |
| **Binary (new)** | **0.987** | 0.884 | **0.818** |

### PR 3: Hierarchical + multitask strategies + k-fold CV
**What:** Two advanced training approaches + cross-validation.
**Why:** Hierarchical leverages family-level knowledge for binary prediction. Multitask trains both tasks simultaneously. k-fold CV provides robust estimates.

- `HierarchicalMLP` — frozen/unfrozen projector from Stage 1 + binary head
- `MultiTaskMLP` — shared backbone with dual family + binary heads
- `_MultiTaskFamilyWrapper`/`_MultiTaskBinaryWrapper` for evaluation
- `training/hierarchical.py` — two-stage training (family → binary)
- `training/cross_validation.py` — k-fold with cluster-level stratified splitting
- `FocalLoss` gets `reduction` parameter (none/sum/mean)
- Configs: `hierarchical_existing.yaml`, `hierarchical_unfrozen.yaml`, `multitask.yaml`

**Note:** `ModelConfig` (from main's recent refactoring) currently supports `ModularMLP`/`MultiInputMLP`. Extending it for `HierarchicalMLP`/`MultiTaskMLP` is a follow-up.

### PR 4: Auxiliary features + identity-aware splits
**What:** Feature concatenation framework + leakage-free splitting.
**Why:** Complementary features boost binary classification. Identity-aware splits prevent sequence leakage — critical for publication.

- `ToxDataset` extended: CPP, HBI, length, venom indicator concatenation
- `_extra_dataset_kwargs()` helper (single source of truth, imported by orchestrator, strategies, hierarchical, ensemble)
- `cpp_features.py` — CPP generation via AAanalysis
- `hbi_features.py` — HBI feature computation via MMseqs2
- `counterpart_acquisition.py` — non-toxic structural counterpart fetching
- Identity-aware splits: MMseqs2 30% clustering → cluster-level stratified split → rebalancing
- `cpp` CLI command

### PR 5: Evaluation toolkit + publication figures
**What:** Ensemble evaluation, data profiling, comparison pipeline, publication figures.
**Why:** Completes the evaluation framework. Ensemble reduces variance. Publication figures ready for paper.

- `ensemble.py` — average softmax across multiple calibrated models
- `data_quality.py` — training data profiling (bias detection, embedding similarity)
- `hbi_binary_baseline.py` — HBI binary evaluation baseline
- `comparison.py` — full method comparison pipeline
- `confidence_routing.py` — model confidence-based routing
- `per_family_eval.py` — per-family breakdown
- `external_benchmarks.py` — ToxinPred2/3 integration
- `publication.py` — 10 publication-quality figures
- CLI commands: `eval-ensemble`, `profile-data`

**Note:** These tools coexist with main's `runner.py` evaluation pattern. `runner.py` handles standard `eval hbi/model/compare` workflows; our tools handle binary-specific evaluation, ensemble, and publication analysis.

### PR 6: Documentation
**What:** Migration documentation + CLAUDE.md update.
**Why:** Keeps project documentation in sync with the expanded codebase.

- `docs/migration_from_exploration.md` — this document
- `CLAUDE.md` updated with 5 strategies, config fields, binary metrics pipeline

## What Was NOT Ported

- **Handcrafted features** (Atchley factors, cysteine patterns) — proven redundant with ProtT5
- **`hierarchical_preprocessing.py`** — Phase 2 data assembly, not needed
- **`toxify_benchmark.py`** — cannot run on macOS ARM64 (TF 1.8 AVX crash)

## Architecture Decisions

1. **Main wins on structure.** Kept `MetricsResult` dataclass, `runner.py` evaluation pattern, `ModelConfig` for inference, `eval hbi/model/compare` CLI, optimizer/scheduler/FocalLoss in `trainer.py`, wandb integration (made optional).
2. **Both metrics APIs coexist.** `MetricsResult` for multiclass (used by `runner.py`), dict returns for binary score-based metrics (ROC-AUC, PR-AUC, threshold optimization). No conflict.
3. **Canonical `get_device()`** lives in `toxfam.device` (cuda > mps > cpu). All modules import from there; `trainer.py` and `embedding.py` re-export for backward compatibility.
4. **`_extra_dataset_kwargs`** is the single source of truth for auxiliary feature kwargs. Defined once in `orchestrator.py`, imported by `strategies.py`, `hierarchical.py`, and `ensemble.py`.
5. **Binary metrics auto-run** after every strategy via `_run_binary_metrics_pipeline`.

## Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_architectures.py | 8 | ModularMLP, MultiInputMLP, MultiTaskMLP shapes |
| test_calibration.py | 3 | ModelWithTemperature |
| test_cli.py | 1 | CLI command registration |
| test_config.py | 23 | All 5 strategies, validators, effective_embedding_dim |
| test_cross_validation.py | 4 | Metric aggregation |
| test_dataset.py | 6 | ToxDataset loading |
| test_device.py | 1 | Device detection |
| test_ensemble.py | 1 | Module import |
| test_fasta.py | 7 | Parse, read_as_dict, write, MD5 skip |
| test_hierarchical.py | 10 | HierarchicalMLP: shapes, gradients, weight loading, config |
| test_identity_splits.py | 4 | Rebalancing logic |
| test_losses.py | 6 | FocalLoss: shape, gamma, weights, reduction |
| test_metrics.py | 17 | MetricsResult + binary scores + threshold optimization |
| test_model_config.py | 8 | ModelConfig (from main's recent refactoring) |
| test_normalization.py | 8 | Family label normalization |
| test_paths.py | 3 | Path helpers |
| test_xml_parser.py | 3 | XML parsing |
| **Total** | **113** | |

## Benchmark Evidence

*Benchmarked on ported codebase (65,179 proteins, MPS device, seed=42):*

| Strategy | Val MCC (best) | Test ROC-AUC | Test PR-AUC | Test MCC (t=0.5) |
|----------|---------------|-------------|------------|-----------------|
| standard (38-class) | 0.784 | 0.983 | 0.894 | 0.780 |
| **binary (2-class)** | **0.881** | **0.987** | 0.884 | **0.818** |
| hierarchical (frozen) | — | — | — | — |
| multitask | — | — | — | — |
| binary + CPP | — | — | — | — |

**Key finding:** Binary strategy achieves +0.039 test MCC over standard, confirming direct binary optimization outperforms derived-from-family metrics.

## Follow-Up Items

- Extend `ModelConfig` to support `HierarchicalMLP`, `MultiTaskMLP`, and `effective_embedding_dim`
- Run hierarchical/multitask/CPP benchmarks
- One Embedding integration (per-residue ProtT5 → codec → 3072d protein vectors)
