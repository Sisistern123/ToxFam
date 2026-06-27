# ToxinPred 3.0 — run provenance

> Authored from the run configuration (the original agent run was interrupted
> before it wrote its own notes; the scores themselves are complete and final).

## Tool
- **ToxinPred 3.0** (Raghava lab, IIIT-Delhi). PyPI package `toxinpred3` **v1.4**.
- Installed into a dedicated venv `.toxinpred3_env` (Python 3.10). The package
  downloads its model bundle on first use.

## Method / command
- **ML model** (`-m 1`), native decision threshold **0.38** (`-t 0.38`, `-d 2`).
- Upstream scoring is pure-Python amino-acid composition (AAC) + dipeptide
  composition (DPC) → Extra Trees `predict_proba`. Single-threaded, ~0.33 s/seq.
- Because every sequence is scored independently, we parallelised with
  `scripts/external_tools/run_toxinpred3.py`, which splits the FASTA into
  contiguous, length-balanced chunks and runs the **unmodified** upstream CLI on
  each chunk in an isolated working dir, then merges. Scores are identical to a
  whole-file run (no re-implementation of scoring).
- Command (per split):
  ```
  .toxinpred3_env/bin/python scripts/external_tools/run_toxinpred3.py \
    --fasta benchmark/test_set/_shared/<split>.fasta \
    --out   benchmark/test_set/toxinpred3/<split>_scores.csv \
    --workers 8 --model 1 --threshold 0.38 --raw-dir <scratch>
  ```

## Output
- `score` = upstream **"ML Score"** = P(toxic) ∈ [0,1], higher = more toxic.
- `native_pred` = the tool's Toxin / Non-Toxin call at t=0.38; `threshold_used` = 0.38.
- **Coverage: test 10,407/10,407, val 9,495/9,495 scored; 0 failures.**

## Inputs / long sequences
- Inputs are full-length proteins from `_shared/{test,val}.fasta`. ToxinPred 3.0 is
  peptide-oriented but its features (AAC/DPC) are composition-based and
  length-normalised, so it accepts full-length proteins without truncation. This
  domain shift (peptide-trained, protein-applied) is part of why it over-calls
  here (low precision).

## Contamination note
- ToxinPred 3.0 is trained on toxic vs non-toxic **peptide/protein** datasets
  curated from public sources (general toxins/AMPs), **not** the metazoan ToxProt
  (UniProt Toxin keyword KW-0800) family set that ToxFam and ToxDL 2.0 draw from.
  It is therefore a comparatively **clean** comparator — train/test overlap with
  our KW-0800 test positives is far less likely than for ToxDL 2.0. Some overlap
  with SwissProt toxins cannot be fully excluded.

## Resource note
- 14 workers saturate a 14-core machine (one single-threaded process per core).
  Cap `--workers` (we used 8) and/or `renice` to keep the machine responsive.
