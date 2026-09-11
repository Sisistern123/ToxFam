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

**Corrected 2026-09-10. The previous version of this note was wrong** and said
ToxinPred 3.0 was "a comparatively **clean** comparator" whose overlap with our
KW-0800 test positives was "far less likely than for ToxDL 2.0". It is not clean,
and the overlap is measurable.

ToxinPred 3.0's positives were curated from ConoServer, DRAMP, CAMPR3, dbAMP 2.0,
YADAMP, DBAASP-v3 **and UniProt release 2021_03**, and the authors state that they
"also searched for toxic proteins/peptides in SwissProt using specific criteria
such as the keywords 'toxin' or 'toxic' and limiting the search to reviewed
entries" (bioRxiv 10.1101/2023.08.11.552911; restated in the authors' Zenodo
deposit 10.5281/zenodo.19877839 for the published version). UniProt keyword
KW-0800 *is named* "Toxin", so that is our query with extra noise. Redundancy
reduction was **removal of exact duplicates only** — no CD-HIT, no identity
clustering. ConoServer, CAMPR3 and dbAMP are themselves partly UniProt-derived,
so the SwissProt route is not the only one.

Measured against our split by `scripts/external_tools/toxinpred3_overlap.py`
(re-run it rather than trusting these numbers):

| criterion | test toxins | non-toxin background |
|---|---|---|
| byte-identical to a ToxinPred 3.0 positive | **79 / 515 (15.3%)** | 27 / 9,264 (0.3%) |
| shares an exact ≥15-aa segment with one | 188 / 515 (36.5%) | 57 / 9,264 (0.6%) |

All 79 byte-identical toxins are ≤35 aa, i.e. inside its training window; 103 of
our 515 test toxins are that short. Exposure is concentrated by family: 91 of 113
conotoxins, against 0 of 36 three-finger toxins, 0 of 18 scoloptoxins, 0 of 15
snaclecs.

It shows: sensitivity at t=0.5 is **0.937 on the 79 exposed toxins against 0.718
on the other 436**. Excluding the byte-identical ones takes PR-AUC 0.595 → 0.528;
excluding the ≥15-aa set takes it to 0.448.

**The correction favours ToxFam**, so there is no incentive to soften it. Note
that the manuscript's "contamination-excluded subset" removes ToxDL 2.0's overlap
only — ToxinPred 3.0's numbers on that subset are still an upper bound.

Prior art for handling this: ToxiPep (PMC12171765) removed ToxinPred 3.0's
overlapping test entries before comparing; HyPepTox-Fuse (PMC12446778) excluded
the tool entirely for that reason; ToxTeller (PMC11270677) found 91 of 200
independent-test peptides already in ToxinPred's training set.

## Resource note
- 14 workers saturate a 14-core machine (one single-threaded process per core).
  Cap `--workers` (we used 8) and/or `renice` to keep the machine responsive.
