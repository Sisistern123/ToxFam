# ToxDL 2.0 — run notes

> **UPDATE (9,779 redo):** the headline benchmark now runs on the canonical 9,779
> test set — see `results/RESULTS_9779.md`. This file documents the original 10,407
> run + method; the method is unchanged. Two reproduction notes for the redo:
> (1) set `return_contacts=False` in ToxDL2's `dataset.py` (~8× faster, identical
> outputs, validated to 4e-7); (2) fetch AlphaFold structures with ≤8 concurrent
> workers (AFDB throttles higher concurrency). On 9,779: coverage 94.1% (578 lack a
> structure); **337/515 test toxins (65.4%) are in ToxDL2's training set** —
> contamination is now quantified and a clean-subset comparison is reported.

Status: **GO, completed full coverage.** Date: 2026-06-25 (Apple Silicon, MPS; no CUDA).

## Code / weights
- Repo: https://github.com/shzhulin/ToxDL2 (cloned at `tools/ToxDL2`).
- Commit: `a26547515e8cd27095ceb861f7346e49985b0d9d` (2024-12-23).
- Pretrained inference weights are **committed in the repo** (no download / no retraining):
  - `checkpoints/ToxDL2_model.pth` — full pickled `ToxDL_GCN_Network` (GCN + dense head).
  - `checkpoints/protein_domain_embeddings.model` (+ `.npy` sidecars) — InterPro→256-d skip-gram
    (45,151 domains, vector_size 256).
- ESM-2 backbone: `esm2_t33_650M_UR50D` (fair-esm), auto-downloaded to
  `~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt` (2.4 GB) + contact-regression file.

## Environment
- New venv: `tools/toxdl2_env` (Python 3.11.14, created with uv).
- Key deps: torch 2.12.1, torch-geometric 2.8.0, fair-esm 2.0.0, gensim 4.x, numpy 1.26.4,
  scikit-learn 1.9.0, biopython, requests. (graphein/biotite/logomaker from the repo's training
  requirements are NOT needed for inference and were not installed.)
- Device split: ESM-2 650M forward on **MPS**; the small GCN + dense head on **CPU**
  (`PYTORCH_ENABLE_MPS_FALLBACK=1`). Validated that this CPU/MPS path reproduces the authors'
  own saved CUDA prediction for `P79703` to abs diff 1.6e-6 (`src/validate_one.py`).

## Patches to the cloned repo (under tools/, allowed)
- `parameters/test_000.py`: hard-coded `device = cuda:0` → auto cuda/mps/cpu.
- New scripts in `tools/ToxDL2/src/`: `fetch_domains.py`, `fetch_structures.py`,
  `run_inference.py` (+ `smoke_test.py`, `validate_one.py`, `bench.py`, `finalize.py`).
- `run_inference.py` is a self-contained batched runner. It is numerically identical to the repo's
  `predict_ToxDL2.py`/`dataset.py` path except for two performance-only changes that do not affect
  outputs: (1) the O(L²) Python Cα-distance edge loop was replaced with a vectorized `torch.cdist`
  build — proven to yield the **identical** undirected 8 Å edge set; (2) `return_contacts=False` in
  the ESM call (the contact map is computed but never used by the model; representations[33] are
  unchanged). These cut per-protein time from the original path's many seconds down to ~0.3 s.

## Inputs obtained
- **Structures:** AlphaFold DB, one model per accession. The task's `…-model_v4.pdb` URL is stale —
  AFDB is now at **v6** — so the downloader tries `v6 → v5 → v4 → API pdbUrl`. All test seqs are
  ≤ 2238 aa so the single `-F1-` fragment fully covers every modelled protein.
  - has_structure = **10,157 / 10,407** (97.6%); **250** accessions absent from AFDB (recorded NA).
  - Saved to `benchmark/test_set/toxdl2/structures/` (2.7 GB); manifest in `structure_manifest.tsv`.
- **InterPro domains:** fetched in bulk from the UniProt REST API
  (`fields=accession,xref_interpro`, batches of 100). 9,857 / 10,407 proteins have ≥1 InterPro
  domain; the rest get the model's built-in 256-d zero domain vector (graceful fallback in
  `get_domain_vector`). Saved to `domains.tsv`.

## Run
- Full test set, no subsampling. Runtime ≈ **50 min** for the 10,307 inference proteins
  (ESM on MPS, ~2–3.4 prot/s; faster on the short-toxin regions, slower on long proteins).
  Resumable runner (skips identifiers already in the CSV).
- Counts: **scored = 10,157**, **NA/no-structure = 250**, **errors = 0**, total = 10,407.

## Output
- `test_scores.csv` — columns `identifier,score,native_pred,has_structure` for all 10,407
  accessions (FASTA order). `score` = ToxDL 2.0 sigmoid **P(toxic) ∈ [0,1]**, higher = more toxic.
  `native_pred` = 1 if score ≥ 0.5 else 0 (the authors' native 0.5 threshold, from
  `utils.calc_metrics_for_test`). For the 250 NA proteins `score`/`native_pred` are empty and
  `has_structure = 0`.

## Diagnostics (vs the held-out reference labels; NOT fed to the tool — for interpretation only)
On the 10,157 scored proteins (496 toxic / 9,661 non-toxic):
- ROC-AUC = **0.990**, PR-AUC = **0.770**.
- @0.5: F1 = 0.778, MCC = 0.781, TPR = 0.970, FPR = 0.027.
- Score means: toxic 0.940 (median 0.997) vs non-toxic 0.027 (median 0.000).
- The 250 NA (no-structure) proteins were 45 toxic / 205 non-toxic.

## Contamination caveat (important)
ToxDL 2.0's positive training set is built from UniProt animal-toxin (ToxProt) annotations — the
**same provenance** as our UniProt Toxin-keyword test positives. The repo's bundled
`data/domain_data/test.domain` already contains UniProt toxin accessions (e.g. `P01546`, `P56409`),
confirming such entries are in its data pipeline. The very high AUC and near-saturated toxin scores
(median 0.997) are therefore almost certainly inflated by train/test overlap and should be read as
an **optimistic upper bound** for the toxin class, not an out-of-distribution generalization estimate.
A like-for-like comparison against the other ToxFam benchmark methods should keep this in mind
(ToxDL 2.0 has likely memorized many of these specific toxins).

## Reproduce
```
source tools/toxdl2_env/bin/activate
cd tools/ToxDL2/src
PYTHONPATH=$(cd ../ && pwd) python fetch_domains.py       # -> domains.tsv
PYTHONPATH=$(cd ../ && pwd) python fetch_structures.py    # -> structures/ + manifest
PYTHONWARNINGS=ignore PYTHONPATH=$(cd ../ && pwd) PYTORCH_ENABLE_MPS_FALLBACK=1 python -u run_inference.py
PYTHONPATH=$(cd ../ && pwd) python finalize.py            # verify + reorder
```
