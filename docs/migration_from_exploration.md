# Migration from Exploration Branch

This document describes the migration of features from the `Exploration` branch onto `main`'s refactored codebase, completed 2026-04-01.

## Summary

Selected mature research features from Exploration were ported onto main's refactored foundation. Main's structure was preserved — including `ModelConfig` for deterministic model reconstruction, `runner.py` for unified evaluation, and the `eval hbi/model/compare` CLI pattern. Exploration's features were adapted to fit alongside these.

**Metrics API:** Both APIs coexist — `MetricsResult` dataclass for multiclass metrics (used by `runner.py`), and dict-based returns for score-based binary metrics and threshold optimization (new).

## What Was Ported

### Foundation — wandb optional, shared modules, test infrastructure
- wandb optional across training pipeline (try/except import, graceful login failure)
- `device.py` — canonical `get_device()` (cuda > mps > cpu), unified across all modules
- `normalization.py` — shared family label normalization (extracted from preprocessing)
- `_fasta.py` — `read_fasta_as_dict` + `write_fasta` with MD5 skip
- Test infrastructure: `conftest.py` shared fixtures + tests for device, fasta, normalization, paths, calibration, dataset, losses
- Gitignore cleanup

### Binary metrics + binary training strategy
- `calculate_binary_metrics_with_scores()` — ROC-AUC, PR-AUC, F1, MCC from probability scores
- `find_optimal_threshold()` — Youden's J, F1, target precision methods
- `NONTOXIN_LABELS` expanded to `{"nontox", "nontoxic", "nontoxin"}`
- Binary metrics pipeline in orchestrator (auto-runs after every strategy)
- `eval-binary` CLI command
- Binary ROC and PR curve plotting
- `run_binary_strategy()` + binary label derivation in orchestrator
- `binary.yaml` config

**Benchmark evidence:**

| Strategy | Test ROC-AUC | Test PR-AUC | Test MCC (t=0.5) |
|----------|-------------|------------|-----------------|
| Standard (baseline) | 0.983 | 0.894 | 0.780 |
| **Binary (new)** | **0.987** | 0.884 | **0.818** |

**Key finding:** Binary strategy achieves +0.039 test MCC over standard, confirming direct binary optimization outperforms derived-from-family metrics.

### Code quality improvements
- `FocalLoss` gets `reduction` parameter (none/sum/mean)
- `torch.load(..., weights_only=True)` for security
- Modern type hints throughout (Dict→dict, List→list, Union→|)
- Narrowed exception handling in preprocessing

## What Was NOT Ported

Features from Exploration that were removed during review as unused or unvalidated:

- **CPP features** (`cpp_features.py`) — physicochemical profiling via AAanalysis. Never run, AAanalysis not in dependencies.
- **HBI training features** (`hbi_features.py`) — 4-dim homology features for training augmentation. Never run, no CLI command. (Note: `evaluation/hbi.py` for benchmarking is retained.)
- **Identity-aware splitting** — cluster-based splitting to prevent sequence leakage. Implemented but never wired into the pipeline.
- **Ensemble evaluation** (`ensemble.py`) — multi-model averaging. Never used.
- **Data quality profiling** (`data_quality.py`) — bias detection. Never used.
- **XML parser** (`xml_parser.py`) — UniProt XML parsing. Never imported by production code.
- **Hierarchical strategy** (`HierarchicalMLP`) — frozen/unfrozen projector with binary head. Removed.
- **Multitask strategy** (`MultiTaskMLP`) — shared backbone with dual heads. Removed.
- **k-fold cross-validation** — cluster-level stratified splitting. Removed.
- **Handcrafted features** (Atchley factors, cysteine patterns) — proven redundant with ProtT5.
- **`toxify_benchmark.py`** — cannot run on macOS ARM64.

## Architecture Decisions

1. **Main wins on structure.** Kept `MetricsResult` dataclass, `runner.py` evaluation pattern, `ModelConfig` for inference, `eval hbi/model/compare` CLI, optimizer/scheduler/FocalLoss in `trainer.py`, wandb integration (made optional).
2. **Both metrics APIs coexist.** `MetricsResult` for multiclass (used by `runner.py`), dict returns for binary score-based metrics. No conflict.
3. **Canonical `get_device()`** lives in `toxfam.device`. All modules import from there; `trainer.py` and `embedding.py` re-export for backward compatibility.
4. **Binary metrics auto-run** after every strategy via `_run_binary_metrics_pipeline`.

## Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_architectures.py | 6 | ModularMLP, MultiInputMLP shapes |
| test_calibration.py | 3 | ModelWithTemperature |
| test_cli.py | 1 | CLI command registration |
| test_config.py | 14 | Strategies, validators, effective_embedding_dim |
| test_dataset.py | 6 | ToxDataset loading |
| test_device.py | 2 | Device detection |
| test_fasta.py | 7 | Parse, read_as_dict, write, MD5 skip |
| test_losses.py | 6 | FocalLoss: shape, gamma, weights, reduction |
| test_metrics.py | 17 | MetricsResult + binary scores + threshold optimization |
| test_model_config.py | 8 | ModelConfig round-trip + checkpoint loading |
| test_normalization.py | 8 | Family label normalization |
| test_paths.py | 3 | Path helpers |
| **Total** | **79** | |
