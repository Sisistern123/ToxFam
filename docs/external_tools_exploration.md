# External Toxin Prediction Tools — Exploration

This document covers our investigation into running and learning from external toxin prediction tools. We evaluated TOXIFY, ToxDL 2.0, ToxinPred2, and ToxinPred3 for benchmarking against ToxFam.

## 1. ToxinPred2 (Benchmarked)

**Status: Successfully benchmarked on our test set.**

| Detail | Value |
|--------|-------|
| Paper | Sharma et al. (2022), Briefings in Bioinformatics |
| Repo | [raghavagps/toxinpred2](https://github.com/raghavagps/toxinpred2) |
| Install | `pip install toxinpred2` |
| Model 1 | AAC (20-dim amino acid composition) + Random Forest (ONNX) |
| Model 2 | AAC + MERCI motifs (requires Perl) — skipped |

**How we run it:** Direct ONNX inference bypassing the buggy CLI. We compute AAC features ourselves and feed them to the ONNX Random Forest model.

**Our test set results:** ROC-AUC=0.970, PR-AUC=0.652, MCC=0.500

**Implementation:** `src/toxfam/evaluation/external_benchmarks.py` — `run_toxinpred2_benchmark()`

## 2. ToxinPred3 (Benchmarked)

**Status: Successfully benchmarked via isolated Python 3.10 environment.**

| Detail | Value |
|--------|-------|
| Paper | Sharma et al. (2024), Briefings in Bioinformatics |
| Repo | [raghavagps/toxinpred3](https://github.com/raghavagps/toxinpred3) |
| Install | `pip install toxinpred3` |
| Model 1 | AAC (20-dim) + DPC (400-dim dipeptide composition) + Extra Trees |
| Model 2 | Hybrid with MERCI motifs — skipped |
| Reported MCC | 0.81 (on their test set) |

**Challenge:** The model was pickled with scikit-learn 1.2.2 but our environment uses sklearn 1.7.0. The internal tree node dtype changed between versions, causing `ValueError: node array from the pickle has an incompatible dtype`.

**Solution:** Created an isolated Python 3.10 venv with sklearn 1.0.2:
```bash
uv venv .toxinpred3_env --python 3.10
uv pip install --python .toxinpred3_env/bin/python scikit-learn==1.0.2 joblib "numpy<2" pandas
```

A standalone inference wrapper (`scripts/toxinpred3_isolated.py`) runs in this environment via subprocess. The `benchmark-external` command auto-detects the isolated env.

**Our test set results:** ROC-AUC=0.916, PR-AUC=0.533, MCC=0.604

**Feature analysis:** DPC (dipeptide composition) features are likely redundant with ProtT5 embeddings for concatenation-based fusion. We confirmed this experimentally — adding handcrafted physicochemical features (Atchley factors + cysteine patterns) to ProtT5 did not improve performance (see Section 7).

## 3. TOXIFY (Reimplemented and Benchmarked)

**Status: Reimplemented in PyTorch and benchmarked on our test set.**

| Detail | Value |
|--------|-------|
| Paper | Cole & Bhatt (2019), PeerJ 7:e7200 |
| Repo | [tijeco/toxify](https://github.com/tijeco/toxify) |
| Requirements | Python 3.6.3, TensorFlow 1.8.0 (conda only) |
| Architecture | Atchley factors (5-dim/AA) -> GRU(270) -> Dense(2) |
| Training data | 4,808 venom + 32,391 non-venom proteins (in repo) |
| Reported | AUC 0.96 on their test set |

### Why it cannot run

| Approach | Result |
|----------|--------|
| Native install (Python 3.6 + TF 1.8) | No ARM64 binaries exist for either |
| Conda x86_64 via Rosetta 2 | TF 1.8 uses AVX instructions; Rosetta 2 does NOT support AVX — crashes with "Illegal hardware instruction" |
| Docker `--platform linux/amd64` | QEMU emulation has the same AVX limitation — TF binary crashes |
| Docker on remote x86_64 VM | Works, but requires x86 hardware |
| TF2 `tf.saved_model.load` | Untested; TF 1.8 SavedModel with `tf.nn.dynamic_rnn` may have compatibility issues in TF2 |

The fundamental blocker is that TensorFlow 1.x binaries are compiled with AVX instructions, and Apple Silicon's Rosetta 2 translation layer does not emulate AVX/AVX2/AVX512.

### Reimplementation path (recommended)

TOXIFY's architecture is trivially simple — a single-layer GRU on Atchley factor encodings:

```python
# PyTorch reimplementation (~50 lines)
class ToxifyGRU(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=270, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        return self.fc(h_n.squeeze(0))
```

Input encoding: each amino acid is mapped to 5 Atchley factors (polarity, secondary structure propensity, molecular size, codon diversity, electrostatic charge). Sequences are zero-padded to max 500 residues.

Training data is included in the repo (`sequence_data/training_data/pre.venom.fasta` and `pre.NOT.venom.fasta`). Retraining from scratch on CPU would take minutes.

### Reimplementation and benchmark results

We reimplemented TOXIFY in PyTorch (`src/toxfam/evaluation/toxify_benchmark.py`), trained on the original data from the TOXIFY repo, and benchmarked on our test set:

**Training data** (from TOXIFY repo, larger than reported in paper):
- 6,133 venom proteins
- 50,000 non-venom proteins

**Training**: Adam lr=0.01, batch_size=256, early stopping patience=5. Converged in 7 epochs on Apple MPS (~20 min).

**Our test set results:** ROC-AUC=0.959, PR-AUC=0.610, MCC=0.561

This confirms the architecture is competent at the toxin prediction task but substantially underperforms modern protein language model approaches (ToxFam augmented: MCC=0.774).

**Implementation:** `src/toxfam/evaluation/toxify_benchmark.py` — `run_toxify_benchmark()`

### What we learned from TOXIFY

- Atchley factor encoding is a compact, biologically meaningful representation (5 factors per amino acid)
- GRU captures sequential dependencies that simple composition features (AAC, DPC) miss
- However, modern protein language models (ProtT5, ESM-2) subsume this information

## 4. ToxDL 2.0

**Status: Feasible but heavy. Requires AlphaFold structures + ESM-2 + domain embeddings.**

| Detail | Value |
|--------|-------|
| Paper | Jiang et al. (2025), CSBJ 27:1538-1549 |
| Repo | [shzhulin/ToxDL2](https://github.com/shzhulin/ToxDL2) |
| Install | Manual clone only (no pip/setup.py) |
| Architecture | ESM-2 per-residue embeddings + GCN on contact maps + domain Skip-gram |
| Reported | F1=0.878, MCC=0.869, ROC-AUC=0.992, PR-AUC=0.891 |

### Architecture details

ToxDL 2.0 combines three signal sources:

1. **ESM-2 per-residue embeddings** (1280-dim per residue) — used as GCN node features
2. **3D structural topology** — AlphaFold PDB structures parsed for C-alpha contacts (<8 angstrom), forming graph edges
3. **Domain co-occurrence embeddings** — 256-dim vectors from a Skip-gram Word2Vec model trained on InterPro domain co-occurrence patterns from 200M UniProt proteins

**GCN architecture:** 4 GCN layers (1280->512->512->512->256) with global mean pooling.

**Fusion:** Concatenation of 256-dim GCN output + 256-dim domain embedding = 512-dim -> MLP (512->256->64->1) with sigmoid output.

### Requirements for running on our data

| Requirement | Source | Effort |
|-------------|--------|--------|
| AlphaFold PDB structures | `https://alphafold.ebi.ac.uk/files/AF-{ID}-F1-model_v4.pdb` | Batch download ~15K files |
| ESM-2 per-residue embeddings | esm2_t33_650M_UR50D (~2.5GB model) | GPU hours for 65K sequences |
| InterPro domain annotations | InterPro REST API or UniProt data | API calls or parse existing data |
| PyTorch Geometric | Manual install with CUDA version coupling | Complex dependency management |
| Code adaptation | Hardcoded `cuda:0`, relative imports, no package structure | Moderate refactoring |

### Getting AlphaFold structures

AlphaFold structures are available for most UniProt proteins via REST API:

```
https://alphafold.ebi.ac.uk/api/prediction/{UniProtID}
```

Returns JSON with PDB/mmCIF/bcif URLs. Direct PDB download:
```
https://alphafold.ebi.ac.uk/files/AF-{UniProtID}-F1-model_v4.pdb
```

Our ProtSpace project (`/Users/jcoludar/CascadeProjects/SpeciesEmbedding/tools/protspace`) already implements AlphaFold structure fetching via this API (TypeScript, in `protspace/packages/utils/src/structure/structure-service.ts`). The same URL pattern can be used in Python:

```python
import requests

def download_alphafold_pdb(uniprot_id: str, output_dir: Path) -> Path | None:
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    resp = requests.get(url)
    if resp.status_code == 200:
        out = output_dir / f"AF-{uniprot_id}-F1-model_v4.pdb"
        out.write_bytes(resp.content)
        return out
    return None  # No structure available
```

Coverage: AlphaFold DB covers ~214 million UniProt proteins, so most of our sequences should be available.

### Practical considerations

- **Storage:** Full per-residue ESM-2 embeddings are much larger than ProtT5 per-protein means (L x 1280 vs 1 x 1024 per protein). For our 65K sequences, this could require 50-100GB.
- **Compute:** ESM-2 inference is slower than ProtT5 due to per-residue output (no mean pooling). GPU recommended.
- **Dependencies:** PyTorch Geometric installation is notoriously complex and tightly coupled to CUDA versions.
- **Overlap:** Their training set also uses UniProt KW-0800 toxins with 40% identity reduction, similar to our setup. Test set overlap could bias comparisons.

### What we can learn from ToxDL 2.0

- **Structural topology matters:** The contact map-based GCN captures spatial relationships that sequence-only models miss. This is particularly relevant for disulfide-rich toxins where 3D structure defines function.
- **Domain embeddings:** InterPro domain co-occurrence patterns provide orthogonal information to sequence embeddings. This is conceptually similar to our HBI features (homology-based).
- **Multi-modal fusion:** Simple concatenation of different feature types with an MLP works well — consistent with our own findings.

### Simpler alternative: Structural features without full ToxDL 2.0

Instead of running ToxDL 2.0's full pipeline, we could extract lightweight structural features:
1. Download AlphaFold PDBs for our proteins
2. Extract: contact density, radius of gyration, secondary structure content, pLDDT scores
3. Add as auxiliary features to our existing ProtT5+HBI model

This would capture some structural signal without the complexity of ESM-2 + GCN + domain embeddings.

## 5. Comparison Summary

| Tool | Benchmarked? | Our Test MCC | Features | Feasibility |
|------|-------------|-------------|----------|-------------|
| ToxinPred2 | Yes | 0.500 | AAC (20-dim) + RF | Direct ONNX inference |
| ToxinPred3 | Yes | 0.604 | AAC+DPC (420-dim) + ET | Isolated sklearn 1.0.2 env |
| TOXIFY | Yes (reimpl.) | 0.561 | Atchley (5/AA) + GRU(270) | PyTorch reimplementation |
| ToxDL 2.0 | No | — | ESM-2 + GCN + domains | Feasible but heavy (structures + ESM-2) |
| **ToxFam** | — | **0.774** | ProtT5 + HBI + len + venom | Best on our test set |

## 6. Feature Landscape

| Feature Type | Used By | Info Captured | Redundant with ProtT5? |
|-------------|---------|---------------|----------------------|
| AAC (20-dim) | ToxinPred2, ToxinPred3 | Global AA composition | Yes — ProtT5 encodes this implicitly |
| DPC (400-dim) | ToxinPred3 | Dipeptide frequencies | Yes — attention already captures local context |
| Atchley factors (5/AA) | TOXIFY | Physicochemical properties per residue | Yes — tested, no improvement |
| Cysteine patterns (5-dim) | ToxFam (tested) | Disulfide framework indicators | Yes — tested, no improvement |
| HBI features (4-dim) | ToxFam | Explicit homology search results | **No — complementary** |
| Venom indicator (1-dim) | ToxFam | Organism taxonomy | **No — complementary** |
| ESM-2 per-residue (1280/AA) | ToxDL 2.0 | Deep per-residue context | Partially — different PLM, per-residue vs mean |
| Contact map (graph) | ToxDL 2.0 | 3D spatial relationships | **No — structural info not in sequences** |
| Domain embeddings (256-dim) | ToxDL 2.0 | InterPro domain co-occurrence | **No — functional annotation level** |

### Key insight

Features that are **redundant** with ProtT5: anything derived from sequence composition or physicochemical properties (AAC, DPC, Atchley factors, cysteine patterns).

Features that are **complementary**: explicit database lookups (HBI, domain annotations), organism metadata (venom indicator), and 3D structural topology (contact maps). These capture information from different sources that no sequence model can learn purely from primary structure.

## 7. Handcrafted Feature Experiment

We tested whether features from the literature complement ProtT5 embeddings:

**Added features:**
- Atchley factor statistics: mean + std of 5 factors across the sequence = 10-dim
- Cysteine patterns: count, fraction, potential disulfide bonds, spacing CV, framework indicator = 5-dim

**Results:**

| Model | Input Dim | ROC-AUC | PR-AUC | MCC |
|-------|-----------|---------|--------|-----|
| ProtT5+HBI+length+venom | 1030 | 0.990 | 0.892 | 0.774 |
| +Atchley+cysteine | 1045 | 0.989 | 0.878 | 0.776 |

**Conclusion:** Handcrafted features are redundant with ProtT5. The protein language model already captures physicochemical and compositional information. The features that actually help are those providing **external knowledge**: HBI (explicit homology search results from a database), counterpart training data (curated non-toxic structural homologs), and organism metadata.

## 8. Recommendations for Future Work

### Done: TOXIFY reimplemented (MCC 0.561)
Reimplemented and benchmarked. See Section 3. Confirms ProtT5-based approach is substantially superior to Atchley factor GRU.

### Priority 1: Download AlphaFold structures for lightweight structural features
- Batch download PDBs for our ~3,400 toxic proteins
- Extract: pLDDT scores, contact density, radius of gyration
- Test as auxiliary features alongside ProtT5+HBI

### Priority 2 (optional): Full ToxDL 2.0 benchmark
- Heavy but feasible: AlphaFold PDBs + ESM-2 + domains
- Would provide the most complete comparison
- Risk: training set overlap could bias results
