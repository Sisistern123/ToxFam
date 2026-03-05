# ToxFam — Current State

## 1. System Architecture

```
data/raw/ (UniProt TSVs: 0800.tsv, nontox.tsv)
    |
    v
[Preprocessing] — normalization, nontox contamination removal,
    |              SignalP6 signal peptide removal,
    |              MMseqs2 clustering (90%), identity-aware splits (30%)
    v
data/processed/training_data.csv + embeddings.h5
    |
    +---> [ProtT5 Embeddings]  (1024-dim per protein)
    +---> [CPP Features]       (100-dim physicochemical profiles)
    +---> [HBI Features]       (4-dim MMseqs2-derived, leave-one-out for train)
    +---> [Taxonomy Vectors]   (56-dim binary, animal taxa membership)
    +---> [Length Feature]      (1-dim, log2(sequence length))
    +---> [Venom Indicator]    (1-dim, organism in venomous taxon)
    +---> [Handcrafted]        (15-dim, Atchley factors + cysteine patterns)
    |
    v
[Training] — MLP (ModularMLP / MultiInputMLP / HierarchicalMLP / MultiTaskMLP)
    |         5 strategies: standard, binary, combined, hierarchical, multitask
    v
[Evaluation] — binary metrics (ROC-AUC, PR-AUC, F1, MCC)
               per-family breakdown, confidence routing, external benchmarks
               publication-quality figures (10 plots)
```

## 2. Dataset

- **65,466 sequences** (with expanded counterparts): ~62,050 nontoxic + ~3,416 toxic (18:1 imbalance)
  - Base dataset: 65,179 sequences (61,763 nontoxic + 3,416 toxic)
  - Non-toxic counterparts: 752 fetched from 24 UniProt query groups, 287 new sequences added to training set
- **37 toxin families** + "other" (546 families with <10 members) + "nontox"
- **Identity-aware splitting**: 30% sequence identity via MMseqs2 clustering, adaptive relaxation for small families
- **Nontox contamination removed**: 375 entries with venom/toxin in UniProt family name were removed from nontox set during preprocessing
- **Signal peptide removal**: All sequences processed through SignalP6; training and evaluation on mature sequences
- **Test set**: 9,587 sequences (577 toxic, 9,010 nontoxic) — unchanged by counterpart addition

### Per-Family Test Set Distribution (Top Families)

| Family | Total | Train | Val | Test |
|--------|-------|-------|-----|------|
| nontox | 61,763 | 43,189 | 9,126 | 9,448 |
| Conotoxin family | 753 | 527 | 113 | 113 |
| Neurotoxin family | 399 | 280 | 60 | 59 |
| Three-finger toxin family | 240 | 168 | 36 | 36 |
| Long (4 C-C) scorpion toxin superfamily | 190 | 133 | 29 | 28 |
| Short scorpion toxin superfamily | 185 | 130 | 28 | 27 |
| Phospholipase family | 175 | 123 | 26 | 26 |
| Scoloptoxin family | 121 | 85 | 18 | 18 |
| Venom metalloproteinase (M12B) family | 123 | 86 | 18 | 19 |
| Snaclec family | 96 | 67 | 15 | 14 |

## 3. Feature Types

| Feature | Dim | Source | Description |
|---------|-----|--------|-------------|
| ProtT5 embeddings | 1024 | ProtT5-XL-U50 | Per-protein mean-pooled transformer embeddings |
| CPP profiles | 100 | AAanalysis | Comparative physicochemical profiling (tox vs nontox) |
| HBI features | 4 | MMseqs2 | best_hit_fident, best_hit_is_toxic, top5_frac_toxic, neg_log_evalue |
| Taxonomy vectors | 56 | taxopy | Binary membership in 56 predefined animal taxa |
| Handcrafted | 15 | Sequence | Atchley factor stats (10) + cysteine patterns (5) |
| Length feature | 1 | Sequence | log2(sequence length) |
| Venom indicator | 1 | Organism ID | Binary: organism in known venomous taxon |

**HBI features**: For training data, leave-one-out search (exclude self-hits) prevents leakage. For val/test, standard search against full training set. 93.8% of sequences have at least one hit. Features per sequence:
1. `best_hit_fident` — fractional identity of best non-self hit (0 if no hit)
2. `best_hit_is_toxic` — binary label of best hit's family (0 if no hit)
3. `top5_frac_toxic` — fraction of top-5 hits that are toxic
4. `neg_log_evalue` — -log10(best hit e-value), capped at 200, normalized to [0,1]

**Handcrafted features**: Atchley factor statistics (mean + std of 5 physicochemical factors per amino acid = 10-dim) + cysteine pattern features (count, fraction, potential disulfide bonds, spacing CV, venom framework indicator = 5-dim). Tested as auxiliary features but found **redundant** with ProtT5 embeddings (see Section 6).

## 4. Training Strategies

| Strategy | Architecture | Input Dim | Classes | Loss | Binary Eval |
|----------|-------------|-----------|---------|------|-------------|
| standard | ModularMLP | 1024 | 38 families | CE/Focal | p_toxic = 1 - p(nontox) |
| binary | ModularMLP | 1024-1030 | 2 (toxic/nontox) | CE/Focal | direct |
| combined | MultiInputMLP | 1024+56 | 38 | CE | p_toxic = 1 - p(nontox) |
| hierarchical | HierarchicalMLP | 1024 | 2 (Stage 2) | CE | direct |
| multitask | MultiTaskMLP | 1024 | 38+2 | weighted CE | binary head |

## 5. Current Results

### Overall Binary Metrics (Test Set, threshold=0.5)

| Method | ROC-AUC | PR-AUC | F1 | MCC | Accuracy | Notes |
|--------|---------|--------|-----|-----|----------|-------|
| Length baseline (73aa) | 0.908 | 0.336 | 0.504 | 0.478 | 92.4% | Sigmoid centered at 73aa |
| ToxinPred2 (AAC-RF) | 0.970 | 0.652 | 0.464 | 0.500 | 86.6% | 20 AA composition + ONNX RF |
| ToxinPred3 (AAC+DPC ET) | 0.916 | 0.533 | 0.617 | 0.604 | 94.1% | AAC + dipeptide + Extra Trees |
| TOXIFY (reimpl.) | 0.959 | 0.610 | 0.552 | 0.561 | 91.3% | Atchley factors + GRU(270), retrained |
| HBI best-hit transfer | 0.860 | 0.615 | 0.773 | 0.760 | 97.4% | MMseqs2 search, label transfer |
| NN binary (ProtT5 only) | 0.986 | 0.838 | 0.769 | 0.758 | 96.9% | Vanilla binary MLP, 1024-dim |
| NN binary+CPP | 0.986 | 0.839 | 0.755 | 0.746 | — | ProtT5 + 100-dim CPP features |
| Confidence routing (NN+HBI) | 0.926 | 0.774 | 0.778 | 0.764 | 97.2% | HBI when confident, else vanilla NN |
| **NN augmented+CP** | **0.990** | **0.892** | **0.780** | **0.774** | **96.9%** | ProtT5+HBI+length+venom, with counterparts |
| Standard (38-class) | 0.980 | 0.709 | — | 0.615 | — | Family classification |

### Per-Family False Negatives (Missed Toxic Sequences, Test Set)

| Family | n_test | NN binary FN | NN aug+CP FN | HBI FN | Conf. routing FN |
|--------|--------|-------------|-------------|--------|-----------------|
| Conotoxin | 169 | 5 | 6 | 17 | 5 |
| Neurotoxin | 66 | 1 | **0** | 2 | 1 |
| Phospholipase | 64 | 0 | **0** | 47 | 42 |
| Snaclec | 46 | 25 | **0** | 6 | 6 |
| other | 45 | 23 | 18 | 26 | 19 |
| Venom Kunitz-type | 38 | 1 | 1 | 12 | 1 |
| Scoloptoxin | 18 | 4 | 4 | 5 | 4 |
| M12B metalloproteinase | 13 | 0 | 0 | 0 | 0 |
| Short scorpion toxin | 13 | 1 | **0** | 5 | 1 |
| Natriuretic/BPP | 13 | 1 | **0** | 4 | 1 |
| **Total FN** | **577** | **73** | **41** | **146** | **91** |

### Nontoxic False Positives (Test Set)

| Method | FP (out of 9,010 nontox) | FP Rate |
|--------|--------------------------|---------|
| HBI best-hit | 92 | 1.0% |
| Confidence routing | 173 | 1.9% |
| NN binary | 215 | 2.4% |
| NN augmented+CP | 240 | 2.7% |
| Length baseline | 517 | 5.7% |

### Key Findings

1. **NN augmented+CP** achieves the best overall metrics: PR-AUC 0.892, ROC-AUC 0.990, MCC 0.774
2. **Counterpart training data** improved augmented model: expanded from 14 to 24 family groups (752 total counterparts, 287 new to training)
3. **Dramatic per-family improvements** with augmented model: Snaclec FN 25->0, Neurotoxin 1->0, Short scorpion 1->0
4. **HBI has lowest FP** (92) but highest FN (146) — conservative predictions with strong specificity
5. **NN augmented reduces total FN** from 73 (vanilla NN) to 41 — 44% fewer missed toxins
6. **ToxinPred2 and ToxinPred3** substantially underperform all our methods on our test set
7. **Length alone** gives ROC-AUC 0.908, confirming a strong length confound (toxic median 60aa vs nontoxic 372aa)

## 6. External Tool Comparison

### Benchmarked on Our Test Set

| Method | ROC-AUC | PR-AUC | MCC | Features | Year |
|--------|---------|--------|-----|----------|------|
| ToxinPred2 (Model 1) | 0.970 | 0.652 | 0.500 | AAC (20-dim) + Random Forest | 2022 |
| ToxinPred3 (Model 1) | 0.916 | 0.533 | 0.604 | AAC + DPC (420-dim) + Extra Trees | 2024 |
| TOXIFY (reimplemented) | 0.959 | 0.610 | 0.561 | Atchley factors (5/AA) + GRU(270) | 2019 |
| **ToxFam augmented+CP** | **0.990** | **0.892** | **0.774** | ProtT5 + HBI + length + venom | 2026 |

**ToxinPred2**: Evaluated via direct ONNX inference with its AAC-RF model (Model 1), bypassing its broken CLI. High ROC-AUC (0.970) but very low PR-AUC (0.652) and MCC (0.500) due to poor calibration.

**ToxinPred3**: Evaluated via isolated Python 3.10 environment with sklearn 1.0.2 (model was pickled with sklearn 1.2.2, incompatible with modern sklearn). Surprisingly performed worse than ToxinPred2 on PR-AUC despite using richer features (DPC + AAC).

**TOXIFY**: Original cannot run on macOS ARM64 (TF 1.8 requires AVX instructions). Reimplemented in PyTorch from scratch — single GRU(270) on Atchley factor encodings (5-dim/AA). Trained on original data from the TOXIFY repo (6,133 venom + 50,000 non-venom). MCC 0.561 — better than ToxinPred2 but still far below our ProtT5-based methods.

### Not Benchmarked

| Tool | Reason |
|------|--------|
| ToxDL 2.0 | Requires AlphaFold2 structures + ESM-2 per-residue embeddings — documented for future work |

### Feature Redundancy Experiment

Tested whether handcrafted features from the literature complement ProtT5 embeddings:

| Feature Set | ROC-AUC | PR-AUC | MCC |
|------------|---------|--------|-----|
| ProtT5+HBI+length+venom (1030-dim) | 0.990 | 0.892 | 0.774 |
| +Atchley factors+cysteine patterns (1045-dim) | 0.989 | 0.878 | 0.776 |
| ProtT5+CPP (1124-dim) | 0.986 | 0.839 | 0.746 |

**Conclusion**: Both handcrafted features (Atchley factors, cysteine patterns) and CPP physicochemical profiles (100-dim across 586 scales) are **redundant** with ProtT5 embeddings — no meaningful improvement. This comprehensively confirms that all sequence-derived physicochemical features (AAC, DPC, Atchley factors, cysteine patterns, CPP) are redundant with ProtT5. The approaches that help are **HBI features** (explicit homology search results) and **non-toxic counterparts** (training data augmentation), not additional sequence-derived features.

## 7. Data Quality Improvements

### Nontox Contamination Removal
- Identified 375 entries in `nontox.tsv` with "venom"/"toxin" in UniProt family names (e.g., Venom Kunitz-type from venomous snakes)
- Added `_remove_nontox_contamination()` to preprocessing pipeline
- These were actual venom proteins mislabeled as non-toxic

### Non-Toxic Counterparts (752 sequences, 287 new to training)
Targeted structural homologs fetched from Swiss-Prot for 24 toxin family groups:

| Counterpart Group | Count | Target Toxic Family |
|-------------------|-------|---------------------|
| defensins | 100 | Scorpion toxins (CSalphabeta fold) |
| mammalian_insulin | 65 | Insulin family |
| Ly6_uPAR | 51 | Three-finger toxin |
| PDGF_VEGF | 37 | PDGF/VEGF growth factor |
| C_type_lectins | 35 | Snaclec |
| serine_proteases | 33 | Peptidase S1 |
| mammalian_PLA2 | 30 | Phospholipase |
| ADAM_metalloproteinases | 25 | Venom metalloproteinase M12B |
| neuropeptides | 25 | NDBP superfamily |
| bradykinin_kininogens | 23 | Bradykinin-related peptide |
| natriuretic_peptides | 24 | Natriuretic/BPP |
| lipocalins | 23 | Calycin superfamily |
| integrins_fibrinogen | 22 | Disintegrin |
| cathelicidins | 20 | Cationic peptide |
| K_channel_subunits | 20 | Sea anemone K+ channel toxin |
| arthropod_defensins | 19 | Short scorpion toxin |
| vasopressin_oxytocin | 17 | Vasopressin/oxytocin |
| MAO | 16 | Flavin monoamine oxidase |
| mast_cell_peptides | 16 | MCD |
| mammalian_CRISP | 14 | CRISP |
| hymenoptera_amps | 14 | Formicidae venom |
| long3cc_defensins | 13 | Long (3 C-C) scorpion toxin |
| perforins | 8 | Actinoporin |
| mammalian_Kunitz | 7 | Venom Kunitz-type |

All counterparts: KW-0800 (Toxin) excluded, signal peptides trimmed using UniProt annotations, ProtT5 embeddings computed via biocentral API. Of 752 fetched, 465 overlapped with existing nontox data; 287 new sequences added to training set.

**Not covered**: Conotoxin (753 test) and Neurotoxin (399 test) lack clear mammalian homologs — conotoxins are unique disulfide-rich miniproteins specific to *Conus*, and neurotoxin overlaps with 3FTx (already covered by Ly6/uPAR counterparts).

### "Other" Bucket Analysis
- 358 entries from 104 families with <10 members each
- Largest has 9 members; none easily rescuable without lowering threshold
- 97.3% misclassification rate in the "other" class

### Normalization Analysis
- Phospholipase: all 327 toxic entries are genuine venom PLA2 from KW-0800; 213 nontox entries are mammalian PLA2 correctly labeled nontox at preprocessing line 76 before normalization applies
- Kunitz: all 148 nontox entries with "venom" in family name were already removed by contamination filter
- Conclusion: normalization refinement not needed — existing pipeline handles contested families correctly

## 8. Publication Figures

All figures generated in `model/model_output/comparison/figures/`:

| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | `fig1_overall_metrics.png` | Grouped bar chart: ROC-AUC, PR-AUC, F1, MCC per method |
| Fig 2 | `fig2_per_family_mcc.png` | Horizontal grouped bars: MCC per family per method |
| Fig 3 | `fig3_dataset_composition.png` | Stacked bar: train/val/test split per family |
| Fig 4 | `fig4_confusion_matrices.png` | Grid of binary confusion matrices per method |
| Fig 5 | `fig5_per_family_confusion.png` | Mini confusion matrices for top problematic families |
| Fig 6 | `fig6_length_distribution.png` | Histogram + box plot: toxic vs nontoxic lengths |
| Fig 7 | `fig7_roc_curves.png` | Overlay ROC curves for all methods |
| Fig 8 | `fig8_pr_curves.png` | Overlay Precision-Recall curves |
| Fig 9 | `fig9_error_venn.png` | Error Venn: Length baseline vs NN binary |
| Fig 9b | `fig9b_error_venn_aug_vs_hbi.png` | Error Venn: NN augmented+CP vs HBI best-hit |

## 9. Modules

| Module | Purpose |
|--------|---------|
| `src/toxfam/data/hbi_features.py` | Leave-one-out HBI feature computation via MMseqs2 |
| `src/toxfam/data/counterpart_acquisition.py` | UniProt counterpart fetching + biocentral embedding |
| `src/toxfam/data/handcrafted_features.py` | Atchley factor + cysteine pattern features (15-dim) |
| `src/toxfam/evaluation/confidence_routing.py` | HBI-first, NN-fallback prediction routing |
| `src/toxfam/evaluation/per_family_eval.py` | Per-family metrics computation |
| `src/toxfam/evaluation/toxify_benchmark.py` | TOXIFY PyTorch reimplementation + benchmark |
| `src/toxfam/evaluation/external_benchmarks.py` | ToxinPred2 + ToxinPred3 + TOXIFY benchmarks |
| `src/toxfam/evaluation/comparison.py` | Full method comparison pipeline + figure orchestration |
| `src/toxfam/visualization/publication.py` | 10 publication-quality figure generators |
| `scripts/toxinpred3_isolated.py` | Standalone ToxinPred3 wrapper for sklearn-compat env |

## 10. CLI Commands

```bash
# Data preparation
uv run toxfam preprocess               # Full preprocessing pipeline
uv run toxfam embed -i <fasta> -o <h5> # Generate ProtT5 embeddings
uv run toxfam taxonomy                 # Generate binary taxonomy vectors
uv run toxfam cpp                      # Generate CPP features
uv run toxfam fetch-counterparts       # Fetch UniProt counterparts + embeddings
uv run toxfam compute-hbi             # Pre-compute HBI features (leave-one-out)
uv run toxfam handcrafted-features    # Compute Atchley + cysteine features

# Training
uv run toxfam train configs/binary_augmented_counterparts.yaml  # Best model

# Evaluation
uv run toxfam eval-test               # Test set evaluation (HBI vs NN)
uv run toxfam eval-comparison         # Full method comparison + figures
uv run toxfam benchmark-external      # Run ToxinPred2/3 benchmarks
uv run toxfam eval-binary <model_dir> # Re-compute binary metrics
uv run toxfam eval-ensemble <dirs>    # Ensemble evaluation
uv run toxfam hbi-baseline            # HBI binary baselines
uv run toxfam profile-data            # Data quality profiling
```

## 11. Key Data Files

| File | Description |
|------|-------------|
| `data/processed/training_data.csv` | Base training split (65,179 sequences) |
| `data/processed/training_data_with_counterparts.csv` | Expanded training split (65,466 sequences) |
| `data/processed/embeddings.h5` | ProtT5 embeddings for base data |
| `data/processed/counterpart_embeddings.h5` | ProtT5 embeddings for 752 counterparts |
| `data/intermediate/hbi/hbi_features.h5` | HBI features for base data |
| `data/intermediate/hbi/hbi_features_with_counterparts.h5` | HBI features for expanded data |
| `data/intermediate/handcrafted/handcrafted_features.h5` | Atchley + cysteine features (65,466 seqs) |
| `data/intermediate/cpp/cpp_features.h5` | CPP physicochemical features (100-dim, 65,179 seqs) |
| `model/model_output/binary_augmented_counterparts_run/` | Best model outputs |
| `model/model_output/comparison/` | All comparison results, metrics, figures |
| `.toxinpred3_env/` | Isolated Python 3.10 + sklearn 1.0.2 for ToxinPred3 |

## 12. Setup for External Benchmarks

### ToxinPred3 Isolated Environment
ToxinPred3's model was pickled with sklearn 1.2.2 and is incompatible with modern sklearn (1.7+). To run it:

```bash
uv venv .toxinpred3_env --python 3.10
uv pip install --python .toxinpred3_env/bin/python scikit-learn==1.0.2 joblib "numpy<2" pandas
```

The `benchmark-external` command automatically detects and uses this environment.

### TOXIFY
Reimplemented in PyTorch (`src/toxfam/evaluation/toxify_benchmark.py`). Training data is auto-downloaded from the TOXIFY GitHub repo. Model is cached after first training (~7 epochs on MPS, ~20 min).

### ToxDL 2.0
Not benchmarked. Requires AlphaFold PDB structures + ESM-2 per-residue embeddings + domain embeddings. See `docs/external_tools_exploration.md` for detailed feasibility analysis and future setup instructions.
