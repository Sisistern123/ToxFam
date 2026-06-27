# ToxDL 2.0 — Feasibility assessment

**Verdict: GO** (with caveats; full coverage runtime is the only real cost concern, not a blocker).

Date: 2026-06-25. Target: score `benchmark/test_set/_shared/test.fasta` (10,407 UniProt proteins) for toxicity.

## 1. Code location
- Paper: Zhu, Fang, Liu, Shen, De Neve, Pan. "ToxDL 2.0: Protein toxicity prediction using a pretrained
  language model and graph neural networks." *Comput Struct Biotechnol J* 27 (2025) 1538–1549.
  DOI 10.1016/j.csbj.2025.04.002. PMC: PMC12018212.
- Code: **https://github.com/shzhulin/ToxDL2** (cloned at `tools/ToxDL2`, commit
  `a26547515e8cd27095ceb861f7346e49985b0d9d`, 2024-12-23).
- Web server: http://www.csbio.sjtu.edu.cn/bioinf/ToxDL2/

## 2. What inference requires per protein
The model (`src/model.py` `ToxDL_GCN_Network`) is a 2-branch network:
- **Graph branch (GCN):** a residue graph where node features = ESM-2 per-residue embeddings (1280-dim)
  and edges connect residues whose Cα atoms are within 8 Å. Both the node features AND the amino-acid
  sequence are read **from the PDB structure** (`pdb_to_graph` in `src/dataset.py` parses CA ATOM records;
  the sequence is reconstructed from the residue 3-letter codes, not from the FASTA).
- **Domain branch:** mean of skip-gram (Word2Vec) embeddings of the protein's InterPro domains (256-dim).
- The two 256-dim vectors are concatenated (512) → dense head → sigmoid → **P(toxic)** in [0,1].

Concrete requirements:
- **(a) ESM-2 weights:** `esm2_t33_650M_UR50D` via `fair-esm`. Auto-downloaded from the ESM model hub on
  first use. The checkpoint is ~2.5 GB (650M params; `esm2_t33_650M_UR50D.pt` ≈ 2.5 GB plus a small
  contact-regression file). Network is available → not a blocker.
- **(b) AlphaFold2 structures:** **mandatory** — the model is structure-based and there is no
  sequence-only mode. Structures can be fetched from the AlphaFold DB by UniProt accession at
  `https://alphafold.ebi.ac.uk/files/AF-<ACC>-F1-model_v4.pdb`. Our test headers are UniProt accessions,
  so this works for any accession present in AFDB. Accessions absent from AFDB (or obsolete) → recorded
  as NA, not silently dropped. All test sequences are ≤ 2238 aa (max 2238, median 350), so the single
  `-F1-` fragment of AFDB v4 (covers up to 2700 aa) fully covers every protein that exists in AFDB.
- **(c) Domain (InterPro) features:** the InterPro→vector skip-gram model **is committed in the repo**
  (`checkpoints/protein_domain_embeddings.model` + `.npy` sidecars, 45,151 domains × 256-dim). What is NOT
  committed is the per-protein InterPro domain list. Two facts make this a non-blocker:
  1. `get_domain_vector` (`src/utils.py`) **gracefully returns a 256-dim zero vector** when a protein has
     no domains / no domains in the skip-gram vocabulary — i.e. the model runs fine with an empty domain
     list (degraded but valid).
  2. Per-accession InterPro domains can be fetched **online in bulk** from the UniProt REST API
     (`fields=accession,xref_interpro`), avoiding a local InterProScan run. So we can supply the *real*
     domains rather than zeros. Offline this would be a partial blocker (zeros only); online it is fine.

## 3. Pretrained inference weights — AVAILABLE
The trained ToxDL 2.0 model is **committed in the repo**: `checkpoints/ToxDL2_model.pth` (5.6 MB, the full
pickled `torch.nn.Module`), plus the domain skip-gram model. No separate download / no retraining needed.
This is the key gate and it passes. (Loading notes: the pickle was saved on CUDA, so it needs
`map_location` + `weights_only=False`, and the hard-coded `device=cuda:0` in `parameters/test_000.py`
must be repointed to cpu/mps — minor patches in our clone, which we are allowed to edit.)

## 4. Sequence-only / structure-optional mode
**No.** Structure is mandatory (it provides both the graph and the residue features). The paper's ablation
removes the GCN but still consumes structure. Proteins lacking an AFDB structure cannot be scored and are
reported as NA.

## Blockers / risks (none fatal)
- **Runtime, not feasibility:** ESM-2 650M forward + an O(L²) Python Cα-distance edge build for 10,407
  proteins on Apple-Silicon CPU/MPS (no CUDA). Estimated low-single-digit hours for full coverage. The
  run is checkpointed/resumable; a fixed-seed documented subsample is the fallback if full coverage proves
  impractical.
- **Device:** CUDA unavailable; ESM runs on MPS, the small GCN on CPU (avoids MPS scatter edge-cases).
- **AFDB coverage:** some accessions (obsolete / not modelled) will 404 → NA.
- **Contamination caveat (not a feasibility issue but important for interpretation):** ToxDL 2.0's positive
  training set is built from UniProt animal-toxin (ToxProt) annotations — the same provenance as our
  UniProt Toxin-keyword test positives. The repo's bundled `data/domain_data/test.domain` already contains
  entries like `P01546`/`P56409`, confirming UniProt toxin accessions are in its data. Expect substantial
  train/test overlap; treat resulting scores as an optimistic upper bound for the toxin class.

## Decision
GO. Inference weights + domain skip-gram are shipped in the repo; ESM-2 is downloadable; AFDB structures
are fetchable per accession; InterPro domains are fetchable in bulk from UniProt with a graceful zero
fallback. Proceeding to set up the env, fetch structures + domains, and run checkpointed inference,
preferring full coverage and logging counts honestly.
