# External-tool comparison benchmark

Benchmarks **ToxFam** against existing computational tools that mine the same
niche (toxin prediction), on ToxFam's own held-out test set, scored with ToxFam's
own metric code so every method is directly comparable.

## TL;DR

On the binary toxic/non-toxic task (the only axis where a fair external comparison
exists — see *Scope*), evaluated on the **9,779-protein test set pinned by
`data/splits/split_manifest.csv`** (sha256 `959e4d5b`), common subset of **9,019**
proteins all methods scored (453 toxins, 5.02% positive prior):

| Method | ROC-AUC | **PR-AUC** | MCC @ t=0.5 |
|---|---|---|---|
| **ToxFam (emb+tax)** | 0.9930 | **0.9586** | 0.8667 |
| EAT (1-NN ProtT5, ours) | **0.9945** | 0.9309 | 0.8703 |
| ToxDL 2.0 (2025) | 0.9909 | 0.7826 | 0.7938 |
| ToxinPred 3.0 (2024) | 0.9253 | 0.5865 | 0.5967 |

Paired-bootstrap difference vs ToxFam (2000 resamples; ✓ = 95% CI excludes 0):
- vs **EAT (1-NN ProtT5)**: ΔROC −0.002 (tied), ΔPR **+0.028** ✓, ΔMCC@0.5 −0.004 (tied)
- vs **ToxinPred 3.0**: ΔROC **+0.068** ✓, ΔPR **+0.371** ✓, ΔMCC@0.5 **+0.270** ✓
- vs **ToxDL 2.0**: ΔROC +0.002 (tied), ΔPR **+0.175** ✓, ΔMCC@0.5 **+0.073** ✓

**EAT** (`toxfam eval eat`, cosine 1-NN over the same ProtT5 embeddings, training
split as a leakage-free reference) is the internal "is the MLP needed for binary
toxicity?" control: it **ties ToxFam on ROC-AUC and at the operating point**
(MCC@0.5) and dominates the external tools — ToxFam's edge over it is on PR-AUC
(+0.028 ✓) and on the family task. **ToxFam wins PR-AUC against the
external tools, significantly** — and PR-AUC is the metric that
matters under a ~5% prior (ROC-AUC is near-ceiling for everyone). The result is
strengthened by a contamination asymmetry (below): ToxinPred 3.0 is a *clean*
comparator that ToxFam beats decisively, while ToxDL 2.0 has a train/test-overlap
advantage (it has seen **61.9% of our test toxins**, 319/515, in training) — and on
the contamination-excluded clean subset ToxFam's **PR-AUC** lead over it widens to
+0.347 ✓, while the two tie on the near-ceiling ROC-AUC (see *Contamination*).

> **Note:** ToxDL 2.0 is the only method here that needs a predicted **3D structure**
> (it embeds an AlphaFold2 model with a GCN). **760 of the 9,779** test proteins (62
> toxic) have no AlphaFold DB entry and are unscorable by it, so it covers
> **9,019/9,779 (92.2%)** and sets the size of the common subset. The other three
> methods are sequence-only and score all 9,779.

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

- **Data snapshot: the split manifest (pinned).** The benchmark runs on the split in
  `data/splits/split_manifest.csv` (sha256 `959e4d5b`; train 45621 / val 9779 /
  **test 9779 / 515 toxins**), which is tracked in git — `build_harness.py` reads it
  through `registry.load_dataset`, never from `training_data.csv`'s own `Split`
  column, so a re-downloaded release CSV cannot move the ground truth. The scores in
  `results/scores/` were produced against **this** split; `numbers_manifest.py`
  refuses to quote them if they cover <90% of it.
- **ToxFam baseline = `configs/combined.yaml`** in
  `model/model_output/combined_run` (shipped as `models-v3`, stamped to manifest
  `959e4d5b`; `build_harness.py` refuses an unstamped or mismatched checkpoint).
  Standalone on the full 9,779 test: **ROC-AUC 0.9932, PR-AUC 0.9614, MCC@0.5
  0.8802**.
- **Contamination of ToxDL 2.0 (the important one).** ToxDL 2.0 trains on
  ToxProt-provenance positives that overlap our UniProt KW-0800 test positives.
  Intersecting its bundled training set (`train.fasta` + `valid.domain`, 15,631
  accessions) with our test set: **319 of 515 test toxins (61.9%) are in ToxDL 2.0's
  training data** (828 / 9,779 proteins overall, 8.5%). So its full-set numbers are
  an **inflated upper bound**, not OOD generalisation — yet ToxFam still beats it on
  PR-AUC. **Contamination-excluded clean subset** (drop the 828 seen proteins;
  8,951 proteins / 196 toxins, common n=8,249): ToxDL 2.0's PR-AUC collapses
  0.783 → 0.551 while ToxFam's barely moves (0.959 → 0.902), widening ToxFam's lead
  to **ΔPR +0.347 ✓**; the two tie on the near-ceiling ROC-AUC (ΔROC −0.003, CI
  includes 0). The PR-AUC gap is the evidence the win is not memorisation.
  ToxinPred 3.0 is *not* ToxProt-trained → a cleaner comparator, beaten by a wide
  margin throughout (clean-subset ΔPR +0.533 ✓).
- **ToxinPred 3.0 domain shift.** Peptide-oriented tool applied to full-length
  proteins; it over-calls (precision 0.34), which depresses its PR-AUC.
- **760 proteins (62 toxic) had no AlphaFold structure** on the 9,779 set, so they
  are unscorable by ToxDL 2.0 (**92.2% coverage**) — it is the only method here that
  needs a predicted 3D structure, since it embeds an AlphaFold2 model with a GCN;
  without one there is no graph and no prediction. Recorded NA (`has_structure=0`)
  and excluded from the common subset so all four compare on the same 9,019.
  (`compare.py` `MIN_COVERAGE` is 0.90 to admit this complete-but-92%-coverage run;
  the gate exists to exclude *incomplete* runs — and it is what caught the stale
  pre-manifest scores at 15.4%.) Those excluded toxins are not a random sample (they
  merely lack an AlphaFold model), so every method's common-subset score shifts
  slightly and equally versus the full set (compare `metrics_full.csv` vs
  `metrics_common.csv`).
- **AlphaFold DB is now v6, not v4** (the v4 URL in older recipes is stale); the
  ToxDL 2.0 downloader tries v6→v5→v4. Flagged for any future structure-based tool.

## Results (detail)

`results/comparison/` (full 9,779) and `results/comparison_clean/` (clean subset):
- `metrics_full.csv` — per method on its own scored subset (with coverage).
- `metrics_common.csv` — all methods on the common subset (full: 9,019; clean:
  8,249). Carries `mcc` (at each method's operating threshold) and `mcc_at_0.5`
  (matched 0.5 threshold; the headline MCC on the full set: ToxFam 0.867 /
  EAT 0.870 / ToxDL2 0.794 / ToxinPred3 0.597). Note: ToxDL 2.0 has no val scores so it uses its
  native 0.5 threshold; the others use Youden@val (threshold-free ROC/PR is the headline).
- `paired_vs_toxfam.csv` — paired-bootstrap ΔROC / ΔPR vs ToxFam, with 95% CIs.
- `roc_pr.png` — ROC + Precision-Recall overlay on the common subset.
- `summary.txt` — the console report.

`results/scores/<method>/{test,val}_scores.csv` — the per-protein predictions
(UniProt accession + score; **no sequences**), so the table can be reproduced
without re-running any tool.

## Reproduce

**A. Verify the headline table from the committed artifacts (cheap, fully
self-contained: only needs `uv sync`; no `download-data`, no tool installs):**
```bash
# full 9,779 comparison
uv run python scripts/external_tools/compare.py \
  --scores-base scripts/external_tools/results/scores \
  --labels-dir  scripts/external_tools/results/ground_truth \
  --out /tmp/toxfam_extcmp
# contamination-excluded clean subset (toxins ToxDL 2.0 never trained on)
uv run python scripts/external_tools/compare.py \
  --scores-base scripts/external_tools/results/scores \
  --labels-dir  scripts/external_tools/results/ground_truth_clean \
  --out /tmp/toxfam_extcmp_clean
```
This reproduces `results/comparison/` and `results/comparison_clean/` exactly from
the committed per-protein scores + committed ground-truth labels (no network, no
model).

**B. Full reproduction from scratch** (needs `uv run toxfam download-data` first, to
fetch `training_data.csv` + embeddings + taxonomy, which are not in git):
```bash
# 1. ToxFam (emb+tax) baseline (retrained on the 9,779 split) + shared substrate + scores
uv run toxfam download-data --force                # canonical 9,779 split training_data.csv
uv run toxfam train configs/combined.yaml          # -> model/model_output/combined_run
uv run python scripts/external_tools/build_harness.py

# 2. ToxinPred 3.0  (py3.10 venv; the v1.4 model pickle is a pre-sklearn-1.3 tree)
#    uv pip install toxinpred3==1.4 scikit-learn==1.2.2 numpy==1.26.4   # newer sklearn -> "node dtype" error
.toxinpred3_env/bin/python scripts/external_tools/run_toxinpred3.py \
  --fasta benchmark/test_set/_shared/test.fasta \
  --out   benchmark/test_set/toxinpred3/test_scores.csv \
  --workers 8 --model 1 --threshold 0.38 --raw-dir /tmp/tp3_test
#    (repeat for val.fasta -> val_scores.csv)

# 3. ToxDL 2.0  (github.com/shzhulin/ToxDL2 @ a265475; weights committed in-repo;
#    ESM-2 650M + AlphaFold DB structures + UniProt InterPro domains).
#    tools/ToxDL2/ and tools/toxdl2_env/ are gitignored (third-party clone + its venv,
#    NOT shipped) — recreate them once with this one-time setup:
git clone https://github.com/shzhulin/ToxDL2 tools/ToxDL2
git -C tools/ToxDL2 checkout a26547515e8cd27095ceb861f7346e49985b0d9d
uv venv --python 3.11 tools/toxdl2_env     # then install torch, torch-geometric, fair-esm,
#    gensim, numpy==1.26.4, scikit-learn, biopython, requests (exact versions in run_notes)
#    REQUIRED patch: set return_contacts=False in tools/ToxDL2/src/dataset.py (~8x faster,
#    identical output). The drivers run IN PLACE from scripts/external_tools/toxdl2/ (no copy);
#    run_inference/validate need ToxDL2's modules on PYTHONPATH — src/ (dataset/model/utils) and
#    the repo root (parameters/). Fetch AF structures with <=8 workers (AFDB throttles higher).
tools/toxdl2_env/bin/python scripts/external_tools/toxdl2/prefetch_structures.py   # pure HTTP, no PYTHONPATH
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=tools/ToxDL2/src:tools/ToxDL2 \
  tools/toxdl2_env/bin/python scripts/external_tools/toxdl2/run_inference_9779.py
#    see results/notes/toxdl2_run_notes.md  ->  benchmark/test_set/toxdl2/test_scores.csv
#
#    NB: re-deriving ONLY the contamination set / clean subset is much lighter — it needs
#    just the clone's two data files (no venv, no inference, no structures):
#      git clone ... tools/ToxDL2 && git -C tools/ToxDL2 checkout a265475
#      uv run python scripts/external_tools/toxdl2/build_clean_subset.py   # prints train=/valid= counts

# 4. compare (full + contamination-excluded clean subset)
uv run python scripts/external_tools/compare.py
uv run python scripts/external_tools/compare.py \
  --labels-dir benchmark/test_set/_shared_clean --out benchmark/test_set/comparison_clean
```

## Files

```
scripts/external_tools/
├── README.md              # this file
├── build_harness.py       # shared FASTA + ground truth + ToxFam p_toxic   (--shared-only)
├── run_toxinpred3.py      # parallel driver for the unmodified ToxinPred 3.0 CLI
├── compare.py             # unified metrics + paired bootstrap + ROC/PR figure
├── toxdl2/                # ToxDL 2.0 drivers (run in place; ToxDL2 modules via PYTHONPATH)
│   ├── prefetch_structures.py   # parallel AlphaFold structure fetch (<=8 workers + backoff)
│   ├── run_inference_9779.py    # resumable ESM(MPS)+GCN(CPU) inference; reuses cached scores
│   ├── validate_9779.py         # one-protein numeric check vs a cached score
│   └── build_clean_subset.py    # contamination set + clean subset (337/515 seen; rebuilds _shared_clean)
└── results/
    ├── RESULTS_9779.md       # full writeup (both tables + contamination)
    ├── comparison/           # full 9,779: metrics_full, metrics_common, paired_vs_toxfam, roc_pr.png, summary.txt
    ├── comparison_clean/     # contamination-excluded clean subset (same files)
    ├── ground_truth/         # test/val_labels.csv (id,seq_len,is_toxic,family; NO seqs) + toxdl2_seen_in_train.txt
    ├── ground_truth_clean/   # clean-subset test_labels (8,929) + full val_labels for thresholding
    ├── notes/                # per-tool provenance: toxinpred3, toxdl2 (+ feasibility)
    └── scores/               # per-protein predictions (accession + score; no sequences)
```

## Snapshot vs live layout

The same artifacts exist in two parallel trees under different names. The committed
`results/` is the **frozen snapshot** the manuscript cites; `benchmark/test_set/` is
the **gitignored working tree** the scripts (re)generate. `compare.py` reads whichever
you point `--scores-base` / `--labels-dir` at (it defaults to the live tree).

| Artifact            | Committed snapshot (in git)     | Live working tree (gitignored, regenerated) |
| ------------------- | ------------------------------- | ------------------------------------------- |
| ground-truth labels | `results/ground_truth/`         | `benchmark/test_set/_shared/`               |
| clean-subset labels | `results/ground_truth_clean/`   | `benchmark/test_set/_shared_clean/`         |
| per-method scores   | `results/scores/<method>/`      | `benchmark/test_set/<method>/`              |
| comparison outputs  | `results/comparison{,_clean}/`  | `benchmark/test_set/comparison{,_clean}/`   |

The `_shared` → `ground_truth` rename is historical: `build_harness.py` writes the
live `_shared/` substrate, and the committed copy was renamed `ground_truth/` for
clarity. The split itself is intentional — never commit the regenerable working tree;
commit a frozen snapshot instead.

Committed (small, reproducible): scripts, the comparison artifacts, per-protein
prediction scores, and ground-truth labels (accession + is_toxic + family + length,
**no sequences**) so path A above is fully self-contained.

Not committed (bulk data): FASTAs (sequences), ProtT5 embeddings, taxonomy vectors,
model weights, AlphaFold structures, tool checkouts and virtualenvs, all under
gitignored `benchmark/`, `data/`, `model/model_output/`, `tools/`.
