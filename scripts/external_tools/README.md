# External-tool comparison benchmark

Benchmarks **ToxFam** against existing computational tools that mine the same
niche (toxin prediction), on ToxFam's own held-out test set, scored with ToxFam's
own metric code so every method is directly comparable.

## TL;DR

On the binary toxic/non-toxic task (the only axis where a fair external comparison
exists — see *Scope*), evaluated on the common subset of **10,157** test proteins
all three methods scored (496 toxins, 4.88% positive prior):

| Method | ROC-AUC | **PR-AUC** | MCC @ t=0.5 |
|---|---|---|---|
| **ToxFam (emb+tax)** | **0.993** | **0.934** | **0.834** |
| ToxDL 2.0 (2025) | 0.990 | 0.770 | 0.781 |
| ToxinPred 3.0 (2024) | 0.933 | 0.566 | 0.637 |

Paired-bootstrap difference vs ToxFam (2000 resamples; ✓ = 95% CI excludes 0):
- vs **ToxinPred 3.0**: ΔROC **+0.061** ✓, ΔPR **+0.367** ✓
- vs **ToxDL 2.0**: ΔROC +0.003 (tied), ΔPR **+0.164** ✓

**ToxFam wins PR-AUC against both, significantly** — and PR-AUC is the metric that
matters under a ~5% prior (ROC-AUC is near-ceiling for everyone). The result is
strengthened by a contamination asymmetry (below): ToxinPred 3.0 is a *clean*
comparator that ToxFam beats decisively, while ToxDL 2.0 has a train/test-overlap
advantage and ToxFam still beats it on PR-AUC.

## Scope: why only the binary axis

ToxFam emits two things — a 38-class family label and a derived binary toxicity
score — so there are two possible comparison axes. We benchmark only the binary one
externally, by design:

- **Family axis → no fair external comparator exists.** ToxFam's class space is 38
  labels with *all* conotoxins collapsed into a single `Conotoxin family` bucket
  (no conopeptide-superfamily resolution). The candidate family classifiers
  (ConoDictor 2.0, hmmcompete) resolve *below* that granularity (≈19 conopeptide
  superfamilies; spider type/family/group), so there is no shared label space and
  no fair shared metric. This mismatch is itself the manuscript's novelty claim
  ("no method is simultaneously family-level and metazoan-wide"). The honest
  family baseline therefore remains **homology (HBI)**, which is already in the
  paper and uses ToxFam's exact 38 classes. ConoDictor/hmmcompete were considered
  and dropped for this reason.
- **Binary axis → crowded, but most candidates are redundant or contaminated.** We
  selected two representatives: **ToxinPred 3.0** (a clean, non-ToxProt comparator,
  actively maintained) and **ToxDL 2.0** (the 2025 SOTA: ESM-2 + AlphaFold2
  structure + domain features — the strongest "is a frozen pLM enough?" foil). The
  rest (ClanTox, ToxClassifier, TOXIFY, ToxinPred2, ToxIBTL, ATSE, CSM-Toxin,
  Deep-STP, MultiTox, MultiToxPred …) are either redundant with these two,
  peptide/lineage-restricted, mode-of-action (different label space), or
  ToxProt-trained (heavy contamination) — see the manuscript's Supplementary
  Table S1 for the full landscape.

## Methodology

1. **Shared substrate** (`build_harness.py`): from `data/processed/training_data.csv`
   write `test.fasta` / `val.fasta` and a ground-truth table per split
   (`identifier, seq_len, is_toxic, family`). **Binary ground truth**:
   `is_toxic = 0` iff the family label is in `{nontox, nontoxic, nontoxin}`, else 1
   — so the catch-all `other` toxin class counts as toxic, matching ToxFam's
   `toxfam.evaluation.metrics`.
2. **ToxFam score**: `p_toxic = 1 − Σ P(nontox classes)` from the calibrated
   emb+tax model, identical to `toxfam.evaluation.binary.compute_p_toxic`.
3. **External scores**: each tool produces a per-protein toxicity probability
   (`identifier, score`, higher = more toxic). See `results/notes/` for provenance.
4. **Scoring** (`compare.py`): every method goes through the same code —
   ROC-AUC, PR-AUC (average precision), plus MCC/F1/precision/recall at a threshold.
   Threshold policy: Youden-J tuned on the **val** split if val scores exist, else
   the tool's native 0.5. Threshold-free ROC-AUC / PR-AUC are the headline; a
   matched t=0.5 MCC is also reported for fairness (ToxFam's Youden threshold,
   tuned for J not MCC, otherwise understates its MCC).
5. **Fair comparison**: metrics are reported both per-method (own scored subset)
   and on the **common subset** (intersection of all methods' scored proteins), so
   the head-to-head is apples-to-apples. Significance via **paired bootstrap** of
   the metric difference vs ToxFam on the common subset (fixed seed 42).

## Key decision points & caveats

- **Data snapshot: local 10,407, not the paper's 9,779.** The manuscript reports a
  9,779-protein / 515-toxin test set; this checkout's `training_data.csv` defines a
  *different, older* 10,407 / 541 split (the published snapshot post-dates the local
  March/April model runs and was never committed here). A benchmark only needs all
  methods scored on identical proteins with identical ground truth, which this
  gives. The ranking is very unlikely to change on 9,779 (gaps are large, method is
  identical); a 9,779 re-run mainly aligns ToxFam's *absolute* numbers with the
  paper. **To do that:** `toxfam download-data`, retrain the baseline, regenerate
  the split, re-score (ToxinPred is fast; ToxDL 2.0 reuses cached structures),
  re-compare.
- **ToxFam baseline = a fresh `configs/combined.yaml` train** into
  `model/model_output/combined_run`. None of the committed local runs matched the
  published model, and the only existing emb+tax run had a stale `model_config.json`
  plus 1128-d *augmented* inputs (not the plain 1024-d embeddings), so it could not
  be loaded against `embeddings.h5`. Training the committed config is the cleanest,
  fully reproducible baseline (local test: ROC 0.993, PR 0.922 — close to the
  paper's 0.995 / 0.949).
- **Contamination asymmetry (the important one).** ToxDL 2.0 trains on
  ToxProt-provenance positives that overlap our UniProt KW-0800 test positives (its
  bundled data even contains test accessions, e.g. P01546), so its 0.990 / 0.770 is
  an **inflated upper bound**, not OOD generalisation — yet ToxFam still beats it on
  PR-AUC. ToxinPred 3.0 is *not* ToxProt-trained → a cleaner comparator, and ToxFam
  beats it by a wide margin.
- **ToxinPred 3.0 domain shift.** Peptide-oriented tool applied to full-length
  proteins; it over-calls (precision 0.34), which depresses its PR-AUC.
- **250 proteins (45 toxic) had no AlphaFold structure** → unscorable by ToxDL 2.0,
  recorded NA and excluded from the common subset so all three are compared on the
  same 10,157.
- **AlphaFold DB is now v6, not v4** (the v4 URL in older recipes is stale); the
  ToxDL 2.0 downloader tries v6→v5→v4. Flagged for any future structure-based tool.

## Results (detail)

`results/comparison/`:
- `metrics_full.csv` — per method on its own scored subset (with coverage).
- `metrics_common.csv` — all methods on the common 10,157 subset.
- `paired_vs_toxfam.csv` — paired-bootstrap ΔROC / ΔPR vs ToxFam, with 95% CIs.
- `roc_pr.png` — ROC + Precision-Recall overlay on the common subset.
- `summary.txt` — the console report.

`results/scores/<method>/{test,val}_scores.csv` — the per-protein predictions
(UniProt accession + score; **no sequences**), so the table can be reproduced
without re-running any tool.

## Reproduce

Prereqs: `uv sync`; `uv run toxfam download-data` (fetches `training_data.csv`,
embeddings, taxonomy — *not* in git).

**A. Verify the headline table from the committed scores (cheap, no tool installs):**
```bash
# regenerate ground-truth labels only (needs training_data.csv, not the model)
uv run python scripts/external_tools/build_harness.py --shared-only
# score the committed predictions through the shared metric code
uv run python scripts/external_tools/compare.py \
  --scores-base scripts/external_tools/results/scores \
  --labels-dir  benchmark/test_set/_shared \
  --out /tmp/toxfam_extcmp
```

**B. Full reproduction from scratch:**
```bash
# 1. ToxFam (emb+tax) baseline + shared substrate + ToxFam scores
uv run toxfam train configs/combined.yaml          # -> model/model_output/combined_run
uv run python scripts/external_tools/build_harness.py

# 2. ToxinPred 3.0  (PyPI `toxinpred3` v1.4 in a py3.10 venv; ML model, t=0.38)
#    see results/notes/toxinpred3_run_notes.md
<env>/bin/python scripts/external_tools/run_toxinpred3.py \
  --fasta benchmark/test_set/_shared/test.fasta \
  --out   benchmark/test_set/toxinpred3/test_scores.csv \
  --workers 8 --model 1 --threshold 0.38 --raw-dir /tmp/tp3_test
#    (repeat for val.fasta -> val_scores.csv)

# 3. ToxDL 2.0  (github.com/shzhulin/ToxDL2 @ a265475; weights committed in-repo;
#    ESM-2 650M + AlphaFold DB structures + UniProt InterPro domains)
#    see results/notes/toxdl2_run_notes.md  ->  benchmark/test_set/toxdl2/test_scores.csv

# 4. compare
uv run python scripts/external_tools/compare.py
```

## Files

```
scripts/external_tools/
├── README.md              # this file
├── build_harness.py       # shared FASTA + ground truth + ToxFam p_toxic   (--shared-only)
├── run_toxinpred3.py      # parallel driver for the unmodified ToxinPred 3.0 CLI
├── compare.py             # unified metrics + paired bootstrap + ROC/PR figure
└── results/
    ├── comparison/        # metrics_full, metrics_common, paired_vs_toxfam, roc_pr.png, summary.txt
    ├── notes/             # per-tool provenance: toxinpred3, toxdl2 (+ feasibility)
    └── scores/            # per-protein predictions (accession + score; no sequences)
```

Not committed (regenerated / bulk data): FASTAs and label tables (sequences),
ProtT5 embeddings, taxonomy vectors, model weights, AlphaFold structures, tool
checkouts and virtualenvs — all under gitignored `benchmark/`, `data/`,
`model/model_output/`, `tools/`.
