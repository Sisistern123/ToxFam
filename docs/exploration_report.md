# ToxFam Phase 2: Exploration Report

This document covers all exploratory work conducted on the `Exploration` branch — from initial hierarchical training strategies through external tool benchmarking and feature engineering experiments. It serves as a comprehensive record of what was tried, what worked, and why, so that future collaborators do not repeat dead ends.

## 1. Starting Point

At the start of Phase 2, ToxFam had:
- A functional pipeline: preprocessing, ProtT5 embedding, training (5 strategies), evaluation
- **65,179 sequences** (3,416 toxic from 37 families + 61,763 non-toxic)
- Identity-aware splitting at 30% sequence identity
- Best binary model: NN binary (ProtT5 only), MCC ~0.615 via standard 38-class prediction

The core question: **Can we beat simple homology-based inference (HBI) for toxin prediction?**

## 2. Timeline of Experiments

This timeline shows every experiment we ran, in chronological order. Checkmarks indicate what was kept in the final system; crosses indicate what was tried and discarded.

| # | Experiment | Result | Verdict |
|---|-----------|--------|---------|
| 1 | **Standard 38-class family classification** | MCC 0.615 (binary via p(nontox)) | Baseline — superseded by direct binary |
| 2 | **Direct binary (ProtT5 only)** | MCC 0.758 | ✅ Kept as vanilla NN baseline |
| 3 | **Hierarchical two-stage** (family→binary, frozen backbone) | ~MCC 0.62 | ❌ No improvement over direct binary |
| 4 | **Hierarchical unfrozen** (fine-tuned backbone) | ~MCC 0.63 | ❌ Marginal, not worth complexity |
| 5 | **Multi-task** (joint family + binary heads) | ~MCC 0.65 | ❌ Comparable to standard, more complex |
| 6 | **Ensemble** (binary + hierarchical average) | ~MCC 0.65 | ❌ No meaningful gain from averaging weak models |
| 7 | **Nontox contamination removal** | Removed 375 mislabeled entries | ✅ Kept — cleaner data |
| 8 | **HBI baseline** (MMseqs2 label transfer) | MCC 0.760 | ✅ Kept as comparison baseline |
| 9 | **HBI as NN features** (4-dim, leave-one-out) | MCC 0.769 | ✅ Kept — key improvement |
| 10 | **Length + venom indicator features** | MCC 0.769 (with HBI) | ✅ Kept — cheap, helpful |
| 11 | **Non-toxic counterpart expansion** (14→24 groups, 752 fetched) | MCC 0.774, FN 73→41 | ✅ Kept — biggest single improvement |
| 12 | **Confidence routing** (HBI when confident, NN fallback) | MCC 0.764 | ❌ Inferior to integrating HBI as features |
| 13 | **Handcrafted features** (Atchley factors + cysteine patterns, 15-dim) | MCC 0.776, PR-AUC 0.878 | ❌ Redundant with ProtT5 (PR-AUC dropped) |
| 14 | **CPP features** (100-dim physicochemical profiles via AAanalysis) | ROC-AUC 0.986, PR-AUC 0.839, MCC 0.746 | ❌ Redundant with ProtT5 (no improvement) |
| 15 | **ToxinPred2 benchmark** (AAC-RF, ONNX) | MCC 0.500 | ✅ Kept as external comparison |
| 16 | **ToxinPred3 benchmark** (AAC+DPC ET, isolated sklearn env) | MCC 0.604 | ✅ Kept as external comparison |
| 17 | **TOXIFY reimplementation** (Atchley GRU, PyTorch) | MCC 0.561 | ✅ Kept as external comparison |

### Discarded model outputs

The following training runs were deleted as they are superseded by the final models. Their results are recorded in this document:

- `standard_run/` — 38-class family classification (experiment #1)
- `hierarchical_existing_run/` — frozen backbone two-stage (experiment #3)
- `hierarchical_unfrozen_run/` — unfrozen backbone two-stage (experiment #4)
- `multitask_run/` — joint family + binary heads (experiment #5)
- `ensemble_all/`, `ensemble_binary_hier/` — ensemble averaging (experiment #6)
- `binary_augmented_run/` — HBI features without counterparts (superseded by counterparts version)
- `binary_augmented_handcrafted_run/` — handcrafted features experiment (experiment #13)
- `confidence_routing_vanilla/` — duplicate routing output

### Retained model outputs

- `binary_run/` — vanilla NN binary baseline (experiment #2)
- `binary_augmented_counterparts_run/` — **best model** (experiment #11)
- `binary_cpp_run/` — CPP features experiment (experiment #14, redundant with ProtT5)
- `comparison/` — publication figures and metrics summary
- `external_benchmarks/` — ToxinPred2/3/TOXIFY metrics
- `hbi_baselines/` — HBI baseline metrics
- `confidence_routing/` — routing experiment metrics

## 3. Detailed Experiment Notes

### 3.1 Hierarchical and Multi-task Strategies (experiments #3–6)

**Hierarchical Training (Two-Stage)**:
Stage 1 trains a 38-class family classifier, Stage 2 loads the projector as a frozen (or fine-tunable) backbone into a binary tox/nontox head.

Result: Did not improve over direct binary training. The family classification backbone learned to separate 38+ classes but this didn't transfer well to the binary boundary. The unfrozen variant was marginally better but not worth the added training time and complexity.

**Multi-task Training (Joint Heads)**:
Shared backbone with two heads — family (38-class) and binary (2-class) — trained jointly with weighted losses (α·L_family + β·L_binary).

Result: Comparable to standard training. The binary head benefits from shared features but the family objective doesn't specifically optimize for tox/nontox discrimination.

**Ensemble**:
Average softmax probabilities from binary + hierarchical models.

Result: No meaningful gain — averaging a weak model with a decent one doesn't help when the decent one alone is better.

**Conclusion**: Direct binary training is simpler and equally effective. Architecture complexity is not the bottleneck — data quality and feature engineering are.

### 3.2 Nontox Contamination Removal (experiment #7)

375 entries in `nontox.tsv` had "venom" or "toxin" in their UniProt family names — actual venom proteins mislabeled as non-toxic (e.g., Venom Kunitz-type from venomous snakes).

Added `_remove_nontox_contamination()` to preprocessing. Impact: cleaner training signal, especially for Kunitz-type and phospholipase families.

### 3.3 HBI Features (experiments #8–9)

**HBI Baseline**: MMseqs2 search of each test sequence against the training set, transfer label of best hit. MCC=0.760, Accuracy=97.4%. This set the bar.

**HBI as NN Features** (4-dim per sequence):
1. `best_hit_fident` — fractional identity of best hit
2. `best_hit_is_toxic` — binary label of best hit
3. `top5_frac_toxic` — fraction of top-5 hits that are toxic
4. `neg_log_evalue` — -log10(best hit e-value), normalized

Key design: Leave-one-out for training data (exclude self-hits). 93.8% of sequences have at least one hit.

Result: MCC 0.758 → 0.769. The NN learns to use homology evidence when available and fall back on embedding-based classification otherwise.

### 3.4 Non-Toxic Counterpart Expansion (experiment #11)

Many toxin families had poor recall because the model saw no non-toxic structural homologs during training. For example, Snaclec (C-type lectin-like) had 25/46 false negatives — the only C-type lectins in training were toxic.

Fetched 752 reviewed Swiss-Prot proteins across 24 counterpart categories (KW-0800 excluded, signal peptides trimmed). 287 were new to the training set. See `docs/current_state.md` Section 7 for the full counterpart table.

Impact — dramatic per-family recall improvement:

| Family | FN Before | FN After |
|--------|-----------|----------|
| Snaclec | 25 | **0** |
| Neurotoxin | 1 | **0** |
| Short scorpion toxin | 1 | **0** |
| Natriuretic/BPP | 1 | **0** |

Total false negatives: 73 → 41 (44% reduction). This was the single biggest improvement.

**Not covered**: Conotoxin (unique to *Conus*) and Neurotoxin (overlaps with 3FTx, covered by Ly6/uPAR counterparts).

### 3.5 Handcrafted Feature Experiment (experiment #13)

Tested whether features from the literature complement ProtT5:
- Atchley factor statistics (10-dim): mean + std of 5 physicochemical factors
- Cysteine patterns (5-dim): count, fraction, potential disulfide bonds, spacing CV, venom framework

| Model | Input Dim | ROC-AUC | PR-AUC | MCC |
|-------|-----------|---------|--------|-----|
| ProtT5+HBI+length+venom | 1030 | 0.990 | 0.892 | 0.774 |
| + Atchley + cysteine | 1045 | 0.989 | 0.878 | 0.776 |

**Verdict**: Redundant. MCC barely changed but PR-AUC actually dropped. ProtT5 already captures these properties implicitly.

### 3.6 CPP Features (experiment #14)

CPP (Comparative Physicochemical Profiling) generates 100-dim features via AAanalysis by comparing physicochemical profiles of toxic vs non-toxic proteins across 586 physicochemical scales.

**Memory challenge**: CPP creates ~580K candidate features (586 scales × ~990 split positions for median sequence length 363). With the full dataset (63K sequences), the internal matrix would require 100+ GB RAM. Solution: subsample to 1,000 toxic + 1,000 nontoxic for feature selection (statistically sufficient for identifying discriminative features), then compute the final 100-feature matrix on all 63K sequences.

**Settings**: `vectorized=False`, `n_batches=5`, `n_jobs=1`, ~9.5 GB peak memory, ~70 min runtime.

| Model | Input Dim | ROC-AUC | PR-AUC | MCC |
|-------|-----------|---------|--------|-----|
| NN binary (ProtT5 only) | 1024 | 0.986 | 0.838 | 0.758 |
| NN binary+CPP | 1124 | 0.986 | 0.839 | 0.746 |

**Verdict**: Redundant. Nearly identical to vanilla binary — CPP physicochemical profiles add no information beyond what ProtT5 already captures. This confirms the broader finding that **all sequence-derived physicochemical features** (AAC, DPC, Atchley factors, cysteine patterns, CPP) are redundant with ProtT5 embeddings.

### 3.7 Confidence Routing (experiment #12)

Use HBI predictions when confident (e-value < 0.1), fall back to NN otherwise.

Result: MCC=0.764 — better than vanilla NN (0.758) but worse than the augmented model (0.774) which integrates HBI features directly into the NN.

**Lesson**: Integrating evidence as features is better than routing between separate prediction systems.

## 4. External Tool Benchmarking

We evaluated three external toxin prediction tools on our test set (9,587 sequences, 577 toxic):

### 4.1 ToxinPred2 (Sharma et al. 2022)
- **Features**: 20-dim AAC, Random Forest (ONNX)
- **How**: Direct ONNX inference bypassing buggy CLI
- **Result**: ROC-AUC=0.970, PR-AUC=0.652, MCC=0.500

### 4.2 ToxinPred3 (Sharma et al. 2024)
- **Features**: AAC (20) + DPC (400) = 420-dim, Extra Trees
- **Challenge**: Model pickled with sklearn 1.2.2; incompatible with modern sklearn
- **Solution**: Isolated Python 3.10 venv with sklearn 1.0.2
- **Result**: ROC-AUC=0.916, PR-AUC=0.533, MCC=0.604

### 4.3 TOXIFY (Cole & Bhatt 2019) — Reimplemented
- **Features**: Atchley factors (5-dim/AA), single GRU(270)
- **Challenge**: Original requires TF 1.8 — cannot run on ARM64 (AVX not supported)
- **Solution**: Full PyTorch reimplementation, trained on original repo data (6,133 venom + 50,000 non-venom)
- **Result**: ROC-AUC=0.959, PR-AUC=0.610, MCC=0.561

### 4.4 ToxDL 2.0 (Jiang et al. 2025) — Documented for Future
- **Architecture**: ESM-2 per-residue + GCN on AlphaFold contact maps + domain Skip-gram
- **Status**: Not benchmarked. Requires AlphaFold PDB download + ESM-2 GPU inference + PyTorch Geometric
- **Feasibility**: Documented in `docs/external_tools_exploration.md` with code snippets and setup instructions

### 4.5 Feature Redundancy Insight

Features **redundant** with ProtT5 (tested or inferred):
- AAC (amino acid composition) — used by ToxinPred2/3
- DPC (dipeptide composition) — used by ToxinPred3
- Atchley factors — used by TOXIFY, tested by us
- Cysteine patterns — tested by us
- CPP physicochemical profiles (100-dim) — tested by us, confirms redundancy across 586 scales

Features **complementary** to ProtT5:
- HBI features (explicit database search results)
- Counterpart training data (curated non-toxic homologs)
- Organism metadata (venom indicator)
- 3D structural topology (contact maps — ToxDL 2.0, untested)
- Domain co-occurrence embeddings (ToxDL 2.0, untested)

## 5. Final Comparison

### Overall Binary Metrics (Test Set, 9,587 sequences)

| Method | ROC-AUC | PR-AUC | F1 | MCC | Notes |
|--------|---------|--------|-----|-----|-------|
| Length baseline | 0.908 | 0.336 | 0.504 | 0.478 | Sigmoid at 73aa |
| ToxinPred2 | 0.970 | 0.652 | 0.464 | 0.500 | AAC + RF |
| ToxinPred3 | 0.916 | 0.533 | 0.617 | 0.604 | AAC + DPC + ET |
| TOXIFY (reimpl.) | 0.959 | 0.610 | 0.552 | 0.561 | Atchley GRU |
| NN binary | 0.986 | 0.838 | 0.769 | 0.758 | ProtT5 only |
| NN binary+CPP | 0.986 | 0.839 | 0.755 | 0.746 | ProtT5 + CPP (100-dim) |
| HBI best-hit | 0.860 | 0.615 | 0.773 | 0.760 | MMseqs2 label transfer |
| Confidence routing | 0.926 | 0.774 | 0.778 | 0.764 | HBI + NN fallback |
| **NN augmented+CP** | **0.990** | **0.892** | **0.780** | **0.774** | **ProtT5+HBI+len+venom+counterparts** |

## 6. Publication Figures

All figures generated by `uv run toxfam eval-comparison`, saved in `model/model_output/comparison/figures/`:

| Figure | Description |
|--------|-------------|
| `fig1_overall_metrics.png` | Grouped bar chart: ROC-AUC, PR-AUC, F1, MCC per method |
| `fig2_per_family_mcc.png` | Per-family MCC breakdown across methods |
| `fig3_dataset_composition.png` | Dataset composition: families, splits, counterparts |
| `fig4_confusion_matrices.png` | Binary confusion matrices for all methods |
| `fig5_per_family_confusion.png` | Per-family confusion for problematic families |
| `fig6_length_distribution.png` | Sequence length distributions: toxic vs non-toxic |
| `fig7_roc_curves.png` | Overlay ROC curves for all methods |
| `fig8_pr_curves.png` | Overlay Precision-Recall curves |
| `fig9_error_venn.png` | Error overlap: Length baseline vs NN binary |
| `fig9b_error_venn_aug_vs_hbi.png` | Error overlap: NN augmented+CP vs HBI |

## 7. Conclusions

1. **ProtT5 embeddings are a strong foundation** — a simple MLP on 1024-dim mean-pooled embeddings achieves MCC 0.758 out of the box.

2. **External knowledge is the key differentiator** — HBI features, counterpart training data, and organism metadata are the only features that measurably improve upon ProtT5. All sequence-derived features (AAC, DPC, Atchley factors, cysteine patterns) are redundant.

3. **Training data quality matters more than model complexity** — counterpart expansion reduced false negatives by 44% and was more impactful than any architectural change (hierarchical, multi-task, ensemble).

4. **All existing external tools significantly underperform** on our test set (ToxinPred2 MCC=0.500, TOXIFY MCC=0.561, ToxinPred3 MCC=0.604) compared to our approach (MCC=0.774).

5. **Structural features are the next frontier** — ToxDL 2.0's use of AlphaFold structures with GCN captures 3D spatial relationships that no sequence-only model can learn. This is the most promising direction for further improvement.

## 8. Future Work

1. **AlphaFold structural features**: Download PDB structures, extract pLDDT, contact density, radius of gyration as auxiliary inputs
3. **ESM-2 per-residue embeddings**: Test whether per-residue representations improve over ProtT5 per-protein means
4. **Full ToxDL 2.0 benchmark**: With AlphaFold structures from UniProt, run on our test set
5. **Expanded counterparts**: Cover remaining families (Conotoxin, Scoloptoxin) if suitable homologs found
