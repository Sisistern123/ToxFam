# External-tool benchmark on the pinned 9,779 test set

Re-run of the ToxFam vs EAT vs ToxinPred 3.0 vs ToxDL 2.0 binary-toxicity comparison
on the test split pinned by **`data/splits/split_manifest.csv`** (sha256
`959e4d5b…`; 9,779 proteins / 515 toxins). All methods are scored by the same metric
code (`compare.py`), P(toxic) per protein, identical ground truth.

ToxFam standalone on the full 9,779 test: **ROC-AUC 0.9934, PR-AUC 0.9582,
MCC@0.5 0.8668**.

**Provenance.** The split is read from the git-tracked manifest, never from
`training_data.csv`'s own `Split` column: `build_harness.py` goes through
`registry.load_dataset`, so a re-downloaded release CSV cannot move the ground
truth. The ToxFam baseline is `models-v2`'s `combined_run`, whose
`models/split_provenance.json` is stamped to the same manifest hash —
`build_harness.py` refuses an unstamped or mismatched checkpoint, so no test id can
be in its training set. `numbers_manifest.py` refuses to quote `results/scores/` if
they cover <90% of the current split. The contamination set + clean subset are
rebuilt by `scripts/external_tools/toxdl2/build_clean_subset.py`.

> **Supersedes the pre-manifest run.** The previous scores were produced against a
> split since shown to be contaminated: they covered only 1,507 of this split's
> 9,779 proteins, and 6,805 of *their* "test" proteins are training data under the
> manifest. Do not compare the two sets of numbers.

## 1. Full 9,779 set — common subset n=9,019 (453 toxic, 5.02% prior)

| Method | ROC-AUC | **PR-AUC** | MCC @ t=0.5 |
|---|---|---|---|
| **ToxFam (emb+tax)** | 0.9930 | **0.9586** | 0.8667 |
| EAT (1-NN ProtT5, ours) | **0.9945** | 0.9309 | 0.8703 |
| ToxDL 2.0 (2025) | 0.9909 | 0.7826 | 0.7938 |
| ToxinPred 3.0 (2024) | 0.9253 | 0.5865 | 0.5967 |

Paired bootstrap vs ToxFam (2000 resamples, seed 42; ✓ = 95% CI excludes 0):

| vs | ΔROC-AUC | ΔPR-AUC | ΔMCC@0.5 |
|---|---|---|---|
| EAT (1-NN ProtT5) | −0.002 `[−0.007,+0.004]` | **+0.028** ✓ `[+0.012,+0.046]` | −0.004 `[−0.028,+0.021]` |
| ToxinPred 3.0 | **+0.068** ✓ `[+0.051,+0.085]` | **+0.371** ✓ `[+0.322,+0.419]` | **+0.270** ✓ `[+0.237,+0.304]` |
| ToxDL 2.0 | +0.002 `[−0.005,+0.007]` | **+0.175** ✓ `[+0.132,+0.219]` | **+0.073** ✓ `[+0.050,+0.096]` |

ToxFam wins PR-AUC against every other method, significantly — the metric that
matters at a ~5% prior, where ROC-AUC is near-ceiling (≥0.991) for the three strong
methods. EAT ties ToxFam on ROC-AUC and at the operating point; ToxFam's edge over
it is PR-AUC and the family task (see the capability table in the manuscript).

## 2. Contamination of ToxDL 2.0

ToxDL 2.0 trains on ToxProt-provenance positives that overlap our UniProt KW-0800
test positives. Intersecting its bundled training set (`train.fasta` +
`valid.domain`, 15,631 accessions) with this test split:

- **319 of 515 test toxins (61.9%) are in ToxDL 2.0's training data.**
- 828 of 9,779 test proteins (8.5%) overall overlap.

Its full-set numbers are therefore an **inflated upper bound**, not
out-of-distribution generalisation — and ToxFam still beats it on PR-AUC.
ToxinPred 3.0 is *not* ToxProt-trained and carries no such overlap.

## 3. Contamination-excluded "clean" subset — common subset n=8,249 (168 toxic, 2.04% prior)

Dropping the 828 proteins (incl. 319 toxins) ToxDL 2.0 trained on leaves 8,951
proteins / 196 toxins; 8,249 of those are scored by all four methods.

| Method | ROC-AUC | **PR-AUC** | MCC @ t=0.5 |
|---|---|---|---|
| **ToxFam (emb+tax)** | 0.9853 | **0.9025** | 0.7520 |
| EAT (1-NN ProtT5, ours) | **0.9942** | 0.8588 | **0.7924** |
| ToxDL 2.0 (2025) | 0.9877 | 0.5510 | 0.6260 |
| ToxinPred 3.0 (2024) | 0.9217 | 0.3673 | 0.4552 |

ToxDL 2.0's PR-AUC collapses **0.783 → 0.551** once its own training data is
removed, while ToxFam's barely moves (**0.959 → 0.902**): ToxFam's paired PR-AUC
lead widens to **+0.347 ✓** `[+0.268,+0.424]`. The two are statistically
indistinguishable on ROC-AUC here (−0.003, `[−0.019,+0.010]`), as expected at a
near-ceiling metric. The PR-AUC gap — not ToxDL 2.0's absolute fall, which partly
reflects the lower prior — is the evidence that its apparent strength was largely
memorisation and ToxFam's is not.

## 4. Structural coverage (ToxDL 2.0 only)

ToxDL 2.0 is the **only method here that requires a predicted 3D structure**: it
builds a graph embedding from an AlphaFold2 model (GCN over residue contacts, ESM-2
node features), so a protein with no AlphaFold DB entry yields no graph and cannot
be scored at all.

- **9,019 / 9,779 scored (92.2%)**; **760 proteins (62 toxic) have no AlphaFold
  model** and are recorded `has_structure=0` with an empty `score` — the NA list is
  `benchmark/test_set/toxdl2/no_structure.txt`.
- This coverage sets the size of the common subset: the other three methods are
  sequence-only and score all 9,779.
- `compare.py`'s `MIN_COVERAGE = 0.90` admits this complete-but-92% run; the gate
  exists to exclude *incomplete* runs, and it is what caught the stale pre-manifest
  scores at 15.4%.
- Those 760 are not a random sample (they merely lack a model), so every method's
  common-subset score shifts slightly and equally versus the full set — compare
  `metrics_full.csv` against `metrics_common.csv`.
- AlphaFold DB is now **v6**, not v4 (the v4 URL in older recipes is stale); the
  downloader tries v6→v5→v4→API `pdbUrl`.

## Artifacts (committed, self-contained — under `scripts/external_tools/results/`)
- Full: `comparison/` (metrics_full, metrics_common, paired_vs_toxfam, roc_pr.png, summary.txt)
- Clean: `comparison_clean/` (same files; labels from `ground_truth_clean/`)
- Per-protein scores: `scores/{toxfam_embtax,eat,toxinpred3,toxdl2}/{test,val}_scores.csv` (no sequences)
- Ground truth: `ground_truth/{test,val}_labels.csv`; clean: `ground_truth_clean/test_labels.csv`
- Contaminated id list (828): `ground_truth/toxdl2_seen_in_train.txt`
- Builders: `../toxdl2/build_clean_subset.py` (contamination + clean subset),
  `../toxdl2/{prefetch_structures,run_inference_9779}.py` (ToxDL 2.0 scoring)

(Regenerable working copies live under the gitignored `benchmark/test_set/`; the
ToxDL 2.0 no-structure NA list is `benchmark/test_set/toxdl2/no_structure.txt`, 760.)

## Reproduce
```bash
# 0. data + the stamped ToxFam baseline (models-v2); no retrain needed
uv run toxfam download-data
uv run python scripts/external_tools/build_harness.py   # ToxFam scores + 9,779 substrate

# 1. ToxinPred 3.0 (py3.10 venv: uv pip install toxinpred3==1.4 scikit-learn==1.2.2 numpy==1.26.4)
.toxinpred3_env/bin/python scripts/external_tools/run_toxinpred3.py \
  --fasta benchmark/test_set/_shared/test.fasta --out benchmark/test_set/toxinpred3/test_scores.csv \
  --workers 8 --model 1 --threshold 0.38 --raw-dir /tmp/tp3_test   # repeat for val

# 2. EAT (ours): identifier,p_toxic lifted from the eval output
uv run toxfam eval eat test_set && uv run toxfam eval eat val_set
#    benchmark/{test_set,val_set}/eat/predictions.csv -> [identifier, p_toxic]
#    saved as benchmark/test_set/eat/{test,val}_scores.csv

# 3. ToxDL 2.0 (tools/toxdl2_env; clone ToxDL2 @ a265475; dataset.py patched return_contacts=False)
uv run --with requests python scripts/external_tools/toxdl2/prefetch_structures.py  # parallel AF fetch (8 workers)
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=tools/ToxDL2/src:tools/ToxDL2 \
  tools/toxdl2_env/bin/python scripts/external_tools/toxdl2/run_inference_9779.py  # ESM(MPS)+GCN(CPU)
uv run python scripts/external_tools/toxdl2/build_clean_subset.py

# 4. compare
uv run python scripts/external_tools/compare.py                                   # full
uv run python scripts/external_tools/compare.py --labels-dir benchmark/test_set/_shared_clean \
  --out benchmark/test_set/comparison_clean                                       # clean subset

# 5. refresh the committed snapshot, then the manuscript numbers
#    cp benchmark/test_set/<method>/{test,val}_scores.csv scripts/external_tools/results/scores/<method>/
#    cp benchmark/test_set/_shared/{test,val}_labels.csv  scripts/external_tools/results/ground_truth/
uv run python -m paper.figures.numbers_manifest
```
