# External-tool benchmark on the canonical 9,779 test set

Re-run of the ToxFam vs EAT vs ToxinPred 3.0 vs ToxDL 2.0 binary-toxicity comparison
on the **published `data-v1` test split (9,779 proteins / 515 toxins)** — the set the
manuscript reports — replacing the earlier local 10,407 snapshot. The ToxFam
baseline was **retrained on the matching 9,779 split** (`combined_run`), so there
is no train/test leakage; all methods are scored by the same metric code
(`compare.py`), P(toxic) per protein, identical ground truth.

ToxFam standalone on the full 9,779 test: **ROC-AUC 0.9948, PR-AUC 0.9494,
MCC@0.5 0.8874** — matches the paper's 0.995 / 0.949 (validates the redo).

**Provenance.** Training split = `data/processed/training_data.csv` md5
`944a36346cb9f1f1a438c471f6f73ad2` (canonical `data-v1` release: train 45621 /
val 9779 / test 9779). `combined_run` was retrained on this CSV *after* the swap —
`model/model_output/combined_run/predictions/test_predictions.csv` has 9,779 rows
and its `config.yaml` `input_csv` points at it, so no test id is in its training
set. The contamination set + clean subset are rebuilt by
`scripts/external_tools/toxdl2/build_clean_subset.py`.

## 1. Full 9,779 set — common subset n=9,201 (477 toxic, 5.18% prior)

| Method | ROC-AUC | **PR-AUC** | MCC@0.5 |
|---|---|---|---|
| **ToxFam (emb+tax)** | **0.9949** | **0.9553** | **0.8999** |
| EAT (1-NN ProtT5, ours) | 0.9967 | 0.9491 | 0.8813 |
| ToxDL 2.0 (2025) | 0.9921 | 0.7976 | 0.8078 |
| ToxinPred 3.0 (2024) | 0.9217 | 0.5737 | 0.5970 |

Paired bootstrap vs ToxFam (2000 resamples; ✓ = 95% CI excludes 0):
- vs **EAT (1-NN ProtT5)**: ΔROC −0.002 (tied), ΔPR +0.006 (tied)
- vs **ToxinPred 3.0**: ΔROC **+0.073** ✓, ΔPR **+0.381** ✓
- vs **ToxDL 2.0**: ΔROC +0.003 (tied), ΔPR **+0.155** ✓

Same ranking and story as the old 10,407 snapshot (which had ToxFam PR 0.934,
ToxDL2 0.770, ToxinPred3 0.566). ToxFam wins PR-AUC against the external tools,
significantly; ties ToxDL2 on the near-ceiling ROC-AUC.

**EAT — the embedding nearest-neighbour baseline.** EAT (`toxfam eval eat`) is a
parameter-free k=1 cosine nearest-neighbour over the *same* ProtT5 embeddings
ToxFam uses, with the training split as reference (leakage-free: train is disjoint
from test). It is added here as the "is the MLP even needed for binary toxicity?"
control. Answer: on threshold-free **ranking it ties ToxFam** (ΔROC/ΔPR CIs both
include 0; EAT's ROC-AUC is even marginally higher), and both dominate the external
tools. ToxFam's edge over EAT shows up only at a usable **operating point**
(MCC@0.5 0.900 vs 0.881; better precision) and on the multiclass family task
(EAT family MCC 0.853 vs ToxFam 0.874; see `toxfam eval compare test_set`). Takeaway:
ProtT5 makes toxic/non-toxic almost linearly separable by nearest neighbour — the
learned head buys calibration and family resolution, not raw binary ranking. The
cosine metric was selected on val_set (beat Euclidean on every metric: val ROC-AUC
0.9944 vs 0.9894).

## 2. Contamination of ToxDL 2.0

ToxDL 2.0 is trained on ToxProt-provenance positives. Intersecting its bundled
training set (`train.fasta` + `valid.domain`, 15,631 accessions) with our test set:

- **337 of 515 test toxins (65.4%) are in ToxDL 2.0's training data** — it has
  seen, by accession, two-thirds of the very toxins it is scored on.
- 850 of 9,779 test proteins (8.7%) overall overlap.

So ToxDL 2.0's full-set numbers are an **inflated upper bound**, not OOD
generalization. (ToxinPred 3.0 is not ToxProt-trained → a clean comparator.)

## 3. Contamination-excluded "clean" subset — common subset n=8,392 (164 toxic, 1.95% prior)

Dropping the 850 proteins (incl. 337 toxins) ToxDL 2.0 trained on:

| Method | ROC-AUC | **PR-AUC** | MCC@0.5 |
|---|---|---|---|
| **ToxFam (emb+tax)** | **0.9980** | 0.8891 | 0.8022 |
| EAT (1-NN ProtT5, ours) | 0.9975 | **0.8988** | **0.8133** |
| ToxDL 2.0 | 0.9898 | 0.5567 | 0.6329 |
| ToxinPred 3.0 | 0.8924 | 0.3148 | 0.3948 |

Paired bootstrap vs ToxFam (✓ = 95% CI excludes 0):
- vs **EAT (1-NN ProtT5)**: ΔROC +0.0005 (tied), ΔPR −0.010 (tied)
- vs **ToxinPred 3.0**: ΔROC **+0.104** ✓, ΔPR **+0.570** ✓
- vs **ToxDL 2.0**: ΔROC **+0.008** ✓, ΔPR **+0.330** ✓

(EAT, being leakage-free by construction, is unaffected by the contamination
exclusion — it stays tied with ToxFam on the clean subset too.)

**The key result.** On toxins ToxDL 2.0 never saw in training, its PR-AUC collapses
0.798 → 0.557, and ToxFam now beats it on **both** ROC-AUC (significantly, vs a tie
on the full set) **and** PR-AUC (gap widens +0.155 → +0.330). This shows ToxFam's
win is real, not a snapshot artifact, and that ToxDL 2.0's apparent full-set
strength was largely memorization.

## Artifacts (committed, self-contained — under `scripts/external_tools/results/`)
- Full: `comparison/` (metrics_full, metrics_common, paired_vs_toxfam, roc_pr.png, summary.txt)
- Clean: `comparison_clean/` (same files; labels from `ground_truth_clean/`)
- Per-protein scores: `scores/{toxfam_embtax,eat,toxinpred3,toxdl2}/{test,val}_scores.csv` (no sequences)
- Ground truth: `ground_truth/{test,val}_labels.csv`; clean: `ground_truth_clean/test_labels.csv`
- Contaminated id list (850): `ground_truth/toxdl2_seen_in_train.txt`
- Builders: `../toxdl2/build_clean_subset.py` (contamination + clean subset),
  `../toxdl2/{prefetch_structures,run_inference_9779}.py` (ToxDL 2.0 scoring)

(Regenerable working copies live under the gitignored `benchmark/test_set/`; the
ToxDL 2.0 no-structure NA list is `benchmark/test_set/toxdl2/no_structure.txt`, 578.)

## Reproduce
```bash
# 0. canonical data + retrained ToxFam baseline
uv run toxfam download-data --force          # 9,779 split training_data.csv
uv run toxfam train configs/combined.yaml    # -> model/model_output/combined_run
uv run python scripts/external_tools/build_harness.py   # ToxFam scores + 9,779 substrate

# 1. ToxinPred 3.0 (py3.10 venv: pip install toxinpred3==1.4 scikit-learn==1.2.2 numpy==1.26.4)
.toxinpred3_env/bin/python scripts/external_tools/run_toxinpred3.py \
  --fasta benchmark/test_set/_shared/test.fasta --out benchmark/test_set/toxinpred3/test_scores.csv \
  --workers 8 --model 1 --threshold 0.38 --raw-dir /tmp/tp3_test   # repeat for val

# 2. ToxDL 2.0 (tools/toxdl2_env; clone ToxDL2 @ a265475; dataset.py patched return_contacts=False)
tools/toxdl2_env/bin/python tools/ToxDL2/src/prefetch_structures.py   # parallel AF fetch (8 workers)
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=tools/ToxDL2 \
  tools/toxdl2_env/bin/python tools/ToxDL2/src/run_inference_9779.py  # ESM(MPS)+GCN(CPU)

# 3. compare
uv run python scripts/external_tools/compare.py                                   # full
uv run python scripts/external_tools/compare.py --labels-dir benchmark/test_set/_shared_clean \
  --out benchmark/test_set/comparison_clean                                       # clean subset
```
