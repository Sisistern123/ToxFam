# ToxFam Results — Analysis & Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce every number, statistic, and figure for the three main Results display items + supplementary, per `docs/results_section_plan.md`, with the load-bearing statistics unit-tested against verified values.

**Architecture:** Reusable, unit-tested statistics live in a new package module `src/toxfam/evaluation/manuscript.py` (imported as `from toxfam.evaluation.manuscript import ...`). Manuscript-specific figure rendering lives in thin scripts under `analysis/figures/` that import the tested stats and write PNG+PDF to `analysis/manuscript_figures/`. Stale benchmark CSVs and the missing binary/OOD artifacts are regenerated via the existing `toxfam eval` CLI before any figure runs.

**Tech stack:** Python 3.12, uv, pandas, numpy, scipy.stats (chi2), scikit-learn (already a dep), matplotlib, pytest. Run everything with `uv run`.

**Verified anchor values (use as test targets — confirmed from `benchmark/test_set/*/predictions.csv`):**
- Test n=9779; non-toxin actual = 9264 (0.9473); toxin-only n=515.
- Toxin-only accuracy: NN-combined **0.9243**, HBI **0.8544**.
- All-class accuracy: NN-combined **0.9860**, HBI **0.9815**.
- McNemar discordant: NN-right/HBI-wrong **127**, HBI-right/NN-wrong **83** → χ²≈**8.80**, p≈**0.003**.
- No-hit subset n=**74** (10 toxin, 64 non-toxin); NN accuracy on it **0.9459**; HBI 0 by construction.
- Toxin-only <30 aa: n=62, NN ≈0.903, HBI ≈0.565.
- Macro-F1 (38-class, no-hit lowers recall): HBI ≈0.851, NN-combined ≈0.792; support>5 flips to NN ≈0.882 vs HBI ≈0.846.
- Ivan's adjudication (`analysis/model_test_wrong_conf_annotated.csv`, 63 rows): assessment correct 33 / partial 10 / incorrect 20; verdict tox 46 / nontox 17; 38 nontox-labelled→verdict toxin.

---

## Phase 0 — Branch & scaffolding

### Task 0.1: Create working branch and output dirs

- [ ] **Step 1: Branch from main**

```bash
cd <repo-root>
git checkout -b results-analysis
mkdir -p analysis/figures analysis/manuscript_figures
```

- [ ] **Step 2: Make `analysis/` an importable package and commit scaffolding**

Figure scripts use package-qualified imports (`from analysis.figures._common import ...`) and are run as modules from the repo root (`uv run python -m analysis.figures.<name>`), so both dirs need `__init__.py`.

```bash
touch analysis/__init__.py analysis/figures/__init__.py
git add analysis/__init__.py analysis/figures/__init__.py
git commit -m "chore: scaffold analysis/figures package for manuscript results"
```

---

## Phase 1 — Manuscript statistics module (TDD foundation)

All figures depend on these functions. Build and test them first. File map:
- Create: `src/toxfam/evaluation/manuscript.py`
- Test: `tests/test_manuscript.py`

### Task 1.1: Subset accuracy + paired comparison primitives

- [ ] **Step 1: Write the failing test** — append to `tests/test_manuscript.py`

```python
"""Tests for toxfam.evaluation.manuscript — manuscript statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from toxfam.evaluation.manuscript import (
    mcnemar_test,
    paired_bootstrap_accuracy_diff,
    subset_accuracy,
)


def _toy_preds():
    # 4 samples: a correct on all but #3; b correct on #0 only
    return pd.DataFrame(
        {
            "identifier": ["x0", "x1", "x2", "x3"],
            "actual_label": ["A", "A", "B", "B"],
            "predicted_label": ["A", "A", "B", "A"],  # a: wrong on x3
        }
    )


def test_subset_accuracy_all():
    df = _toy_preds()
    assert subset_accuracy(df) == pytest.approx(0.75)


def test_subset_accuracy_masked():
    df = _toy_preds()
    mask = df["actual_label"] == "A"  # x0,x1 both correct
    assert subset_accuracy(df, mask) == pytest.approx(1.0)


def test_mcnemar_counts_and_significance():
    correct_a = np.array([1, 1, 1, 0, 1])
    correct_b = np.array([1, 0, 0, 0, 1])  # b wrong where a right on idx1,2
    res = mcnemar_test(correct_a, correct_b)
    assert res["b01"] == 2  # a right, b wrong
    assert res["b10"] == 0  # a wrong, b right
    assert res["n_discordant"] == 2
    assert "chi2" in res and "p_value" in res


def test_paired_bootstrap_diff_sign_and_ci():
    rng_correct_a = np.array([1] * 90 + [0] * 10)
    rng_correct_b = np.array([1] * 80 + [0] * 20)
    res = paired_bootstrap_accuracy_diff(rng_correct_a, rng_correct_b, n_boot=2000, seed=42)
    assert res["diff"] == pytest.approx(0.10, abs=1e-9)
    assert res["ci_low"] < res["diff"] < res["ci_high"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_manuscript.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'toxfam.evaluation.manuscript'`

- [ ] **Step 3: Create `src/toxfam/evaluation/manuscript.py` with the primitives**

```python
"""Manuscript-specific evaluation statistics (reusable, unit-tested).

Functions here compute the load-bearing numbers for the Results section:
subset/toxin-only accuracy, paired significance (McNemar + paired bootstrap),
accuracy-vs-length, per-family F1 differences, macro-F1 conventions, binary
calibration/reliability, and the confident-error adjudication summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2

from toxfam.evaluation.metrics import NONTOXIN_LABELS


def correctness(preds: pd.DataFrame) -> np.ndarray:
    """Boolean array: predicted_label == actual_label."""
    return (preds["predicted_label"].values == preds["actual_label"].values)


def subset_accuracy(preds: pd.DataFrame, mask: np.ndarray | pd.Series | None = None) -> float:
    """Accuracy over all rows, or over rows where ``mask`` is True."""
    correct = correctness(preds)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        correct = correct[mask]
    return float(correct.mean()) if len(correct) else float("nan")


def toxin_mask(preds: pd.DataFrame, label_col: str = "actual_label") -> np.ndarray:
    """True where the ground-truth label is a toxin (not a non-toxin class)."""
    return ~preds[label_col].str.lower().isin(NONTOXIN_LABELS).values


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired McNemar test on two boolean correctness vectors (a vs b).

    b01 = a-correct & b-wrong; b10 = a-wrong & b-correct. Uses the
    continuity-corrected chi-square with 1 dof.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    b01 = int(np.sum(a & ~b))
    b10 = int(np.sum(~a & b))
    n = b01 + b10
    chi2 = ((abs(b01 - b10) - 1) ** 2) / n if n > 0 else 0.0
    p = float(_chi2.sf(chi2, df=1)) if n > 0 else 1.0
    return {"b01": b01, "b10": b10, "n_discordant": n, "chi2": float(chi2), "p_value": p}


def paired_bootstrap_accuracy_diff(
    correct_a: np.ndarray, correct_b: np.ndarray, *, n_boot: int = 10000, seed: int = 42
) -> dict:
    """Paired bootstrap of accuracy(a) - accuracy(b) over the same samples.

    Returns the point difference and a 95% percentile CI.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return {
        "diff": float(a.mean() - b.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n": n,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_manuscript.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/toxfam/evaluation/manuscript.py tests/test_manuscript.py
git commit -m "feat(manuscript): subset accuracy + McNemar + paired bootstrap"
```

### Task 1.2: Accuracy-by-length and rolling-accuracy

- [ ] **Step 1: Add failing tests** to `tests/test_manuscript.py`

```python
from toxfam.evaluation.manuscript import accuracy_by_length_bins, rolling_accuracy_vs_length


def test_accuracy_by_length_bins_basic():
    preds = pd.DataFrame(
        {
            "identifier": [f"x{i}" for i in range(6)],
            "actual_label": ["A"] * 6,
            "predicted_label": ["A", "B", "A", "A", "B", "A"],  # wrong at idx1,4
        }
    )
    lengths = pd.Series([10, 20, 40, 60, 80, 200], index=preds["identifier"])
    out = accuracy_by_length_bins(preds, lengths, bins=[0, 30, 100, 1000])
    assert list(out["n"]) == [2, 3, 1]
    assert out.loc[out["bin_label"] == "0-30", "accuracy"].iloc[0] == pytest.approx(0.5)


def test_rolling_accuracy_monotone_length_sorted():
    preds = pd.DataFrame(
        {
            "identifier": [f"x{i}" for i in range(5)],
            "actual_label": ["A"] * 5,
            "predicted_label": ["A"] * 5,
        }
    )
    lengths = pd.Series([5, 4, 3, 2, 1], index=preds["identifier"])
    out = rolling_accuracy_vs_length(preds, lengths, window=2)
    assert (out["length"].values == np.array([1, 2, 3, 4, 5])).all()
    assert (out["accuracy"].values == 1.0).all()
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_manuscript.py -q` → ImportError.

- [ ] **Step 3: Implement** — append to `manuscript.py`

```python
def _lengths_for(preds: pd.DataFrame, lengths: pd.Series) -> np.ndarray:
    return lengths.reindex(preds["identifier"].values).to_numpy(dtype=float)


def accuracy_by_length_bins(
    preds: pd.DataFrame, lengths: pd.Series, *, bins: list[int]
) -> pd.DataFrame:
    """Accuracy within fixed length bins. ``lengths`` indexed by identifier."""
    ln = _lengths_for(preds, lengths)
    correct = correctness(preds).astype(float)
    labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    cat = pd.cut(ln, bins=bins, labels=labels, include_lowest=True, right=False)
    df = pd.DataFrame({"bin_label": cat, "correct": correct})
    g = df.groupby("bin_label", observed=True)["correct"]
    out = g.agg(accuracy="mean", n="size").reset_index()
    out["se"] = np.sqrt(out["accuracy"] * (1 - out["accuracy"]) / out["n"])
    return out


def rolling_accuracy_vs_length(
    preds: pd.DataFrame, lengths: pd.Series, *, window: int = 50
) -> pd.DataFrame:
    """Length-sorted rolling-window mean accuracy (centered)."""
    ln = _lengths_for(preds, lengths)
    correct = correctness(preds).astype(float)
    order = np.argsort(ln, kind="stable")
    s = pd.Series(correct[order])
    roll = s.rolling(window=window, center=True, min_periods=max(1, window // 2)).mean()
    return pd.DataFrame({"length": ln[order], "accuracy": roll.to_numpy()})
```

- [ ] **Step 4: Run, expect PASS** — `uv run pytest tests/test_manuscript.py -q`

- [ ] **Step 5: Commit**

```bash
git add src/toxfam/evaluation/manuscript.py tests/test_manuscript.py
git commit -m "feat(manuscript): accuracy-by-length + rolling accuracy"
```

### Task 1.3: Per-family F1 difference + support-stratified macro-F1 + no-hit conventions

- [ ] **Step 1: Add failing tests**

```python
from toxfam.evaluation.manuscript import (
    macro_f1_by_support,
    macro_f1_conventions,
    per_family_f1_difference,
)


def _two_method_preds():
    actual = ["A"] * 5 + ["B"] * 5 + ["nontox"] * 5
    a_pred = ["A"] * 5 + ["B"] * 4 + ["A"] + ["nontox"] * 5          # NN-like
    b_pred = ["A"] * 4 + ["no hit"] + ["B"] * 5 + ["nontox"] * 5     # HBI-like (one no hit)
    return (
        pd.DataFrame({"identifier": [f"x{i}" for i in range(15)], "actual_label": actual, "predicted_label": a_pred}),
        pd.DataFrame({"identifier": [f"x{i}" for i in range(15)], "actual_label": actual, "predicted_label": b_pred}),
    )


def test_per_family_f1_difference_columns():
    a, b = _two_method_preds()
    out = per_family_f1_difference(a, b, class_list=["A", "B", "nontox"])
    assert set(["family", "f1_a", "f1_b", "diff", "support"]).issubset(out.columns)
    assert (out["diff"] == (out["f1_a"] - out["f1_b"])).all()


def test_macro_f1_by_support_threshold():
    a, b = _two_method_preds()
    out = macro_f1_by_support(a, b, class_list=["A", "B", "nontox"], support_threshold=4)
    assert {"group", "macro_f1_a", "macro_f1_b", "n_families"}.issubset(out.columns)


def test_macro_f1_conventions_nohit_penalised_le_restricted():
    _, b = _two_method_preds()
    conv = macro_f1_conventions(b, class_list=["A", "B", "nontox"])
    assert conv["macro_f1_nohit_wrong"] <= conv["macro_f1_restricted"] + 1e-9
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — append to `manuscript.py`

```python
from toxfam.evaluation.metrics import calculate_metrics
from toxfam.evaluation.hbi import NO_HIT_LABEL


def _per_class_f1(preds: pd.DataFrame, class_list: list[str]) -> dict[str, dict]:
    m = calculate_metrics(preds["actual_label"], preds["predicted_label"], class_list=class_list)
    return m.classification_report


def per_family_f1_difference(
    preds_a: pd.DataFrame, preds_b: pd.DataFrame, *, class_list: list[str]
) -> pd.DataFrame:
    """Per-family F1 for method a minus method b, with true support.

    Non-toxin classes are excluded from the family view.
    """
    rep_a = _per_class_f1(preds_a, class_list)
    rep_b = _per_class_f1(preds_b, class_list)
    rows = []
    for fam in class_list:
        if fam.lower() in NONTOXIN_LABELS:
            continue
        fa = rep_a.get(fam, {})
        fb = rep_b.get(fam, {})
        rows.append(
            {
                "family": fam,
                "f1_a": float(fa.get("f1-score", 0.0)),
                "f1_b": float(fb.get("f1-score", 0.0)),
                "support": int(fa.get("support", 0)),
            }
        )
    out = pd.DataFrame(rows)
    out["diff"] = out["f1_a"] - out["f1_b"]
    return out.sort_values("diff").reset_index(drop=True)


def macro_f1_by_support(
    preds_a: pd.DataFrame, preds_b: pd.DataFrame, *, class_list: list[str], support_threshold: int = 5
) -> pd.DataFrame:
    """Macro-F1 of each method split by family support (> vs <= threshold)."""
    fam = per_family_f1_difference(preds_a, preds_b, class_list=class_list)
    rows = []
    for label, sub in (
        (f"support>{support_threshold}", fam[fam["support"] > support_threshold]),
        (f"support<={support_threshold}", fam[fam["support"] <= support_threshold]),
    ):
        rows.append(
            {
                "group": label,
                "macro_f1_a": float(sub["f1_a"].mean()) if len(sub) else float("nan"),
                "macro_f1_b": float(sub["f1_b"].mean()) if len(sub) else float("nan"),
                "n_families": int(len(sub)),
                "n_sequences": int(sub["support"].sum()),
            }
        )
    return pd.DataFrame(rows)


def macro_f1_conventions(preds: pd.DataFrame, *, class_list: list[str]) -> dict:
    """Macro-F1 under two no-hit conventions for a single method.

    - nohit_wrong: no-hit predictions kept (map to OOV → lower true-class recall).
    - restricted: drop rows whose prediction is 'no hit' before scoring.
    """
    m_all = calculate_metrics(preds["actual_label"], preds["predicted_label"], class_list=class_list)
    keep = preds["predicted_label"] != NO_HIT_LABEL
    sub = preds[keep]
    m_res = calculate_metrics(sub["actual_label"], sub["predicted_label"], class_list=class_list)
    return {
        "macro_f1_nohit_wrong": float(m_all.classification_report["macro avg"]["f1-score"]),
        "macro_f1_restricted": float(m_res.classification_report["macro avg"]["f1-score"]),
        "n_no_hit": int((~keep).sum()),
    }
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add src/toxfam/evaluation/manuscript.py tests/test_manuscript.py
git commit -m "feat(manuscript): per-family F1 diff + support-stratified + no-hit conventions"
```

### Task 1.4: Binary reliability/ECE + adjudication summary

- [ ] **Step 1: Add failing tests**

```python
from toxfam.evaluation.manuscript import adjudication_summary, binary_reliability


def test_binary_reliability_perfect_calibration():
    # scores equal to true probability in two clean bins
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    out = binary_reliability(y, p, n_bins=2)
    assert out["ece"] == pytest.approx(0.0, abs=1e-9)


def test_adjudication_summary_counts(tmp_path):
    csv = tmp_path / "adj.csv"
    csv.write_text(
        "identifier,verdict,actual_label,predicted_label,assessment,assessment_category\n"
        "p1,tox,nontox,Phospholipase family,correct,family_correct\n"
        "p2,nontox,nontox,other,incorrect,false_positive_nonspecific\n"
        "p3,tox,nontox,Venom Kunitz-type family,partial,family_adjacent\n"
    )
    s = adjudication_summary(csv)
    assert s["n"] == 3
    assert s["assessment"]["correct"] == 1
    assert s["assessment"]["incorrect"] == 1
    assert s["n_annotation_gaps"] == 2  # nontox-labelled & verdict tox (p1,p3)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — append to `manuscript.py`

```python
from collections import Counter
from pathlib import Path


def binary_reliability(
    y_true: np.ndarray, p_toxic: np.ndarray, *, n_bins: int = 15
) -> dict:
    """Reliability-diagram data + Expected Calibration Error for the binary head.

    Equal-width confidence bins on max(p, 1-p); accuracy = P(predicted class correct).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_toxic, dtype=float)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs, props = [], [], [], []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        prop = in_bin.mean()
        centers.append((lo + hi) / 2)
        if prop > 0:
            acc_bin = correct[in_bin].mean()
            conf_bin = conf[in_bin].mean()
            ece += abs(conf_bin - acc_bin) * prop
            accs.append(acc_bin); confs.append(conf_bin); props.append(prop)
        else:
            accs.append(np.nan); confs.append(np.nan); props.append(0.0)
    return {"bin_center": centers, "bin_accuracy": accs, "bin_confidence": confs,
            "bin_proportion": props, "ece": float(ece)}


def adjudication_summary(csv_path: str | Path) -> dict:
    """Summarize Ivan's confident-error adjudication CSV for Figure 3 Panel B."""
    df = pd.read_csv(csv_path)
    gaps = df[(df["actual_label"].str.lower().isin(NONTOXIN_LABELS)) & (df["verdict"].str.lower() == "tox")]
    return {
        "n": int(len(df)),
        "assessment": dict(Counter(df["assessment"].str.strip())),
        "assessment_category": dict(Counter(df["assessment_category"].str.strip())),
        "verdict": dict(Counter(df["verdict"].str.strip())),
        "n_annotation_gaps": int(len(gaps)),
        "annotation_gap_ids": gaps["identifier"].tolist(),
    }
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add src/toxfam/evaluation/manuscript.py tests/test_manuscript.py
git commit -m "feat(manuscript): binary reliability/ECE + adjudication summary"
```

---

## Phase 2 — Regenerate fresh evaluation artifacts (operational)

The committed benchmark CSVs predate `confidence_uncalibrated`; binary metrics and OOD benchmarks are absent. Regenerate them with the existing CLI. No code changes.

### Task 2.1: Refresh NN multiclass predictions (adds confidence_uncalibrated)

- [ ] **Step 1: Re-run NN test+val eval for both models**

```bash
uv run toxfam eval model test_set --model-dir model/model_output/combined_run
uv run toxfam eval model test_set --model-dir model/model_output/standard_run
uv run toxfam eval model val_set  --model-dir model/model_output/combined_run
uv run toxfam eval model val_set  --model-dir model/model_output/standard_run
```

- [ ] **Step 2: Verify the refreshed CSVs and that anchors still hold**

```bash
uv run python - <<'PY'
import pandas as pd
from toxfam.evaluation.manuscript import subset_accuracy, toxin_mask
df = pd.read_csv("benchmark/test_set/nn_combined_run/predictions.csv")
assert "confidence_uncalibrated" in df.columns, "uncalibrated col missing"
acc = subset_accuracy(df); tox = subset_accuracy(df, toxin_mask(df))
print(f"all-class acc={acc:.4f}  toxin-only acc={tox:.4f}")
assert abs(acc - 0.9860) < 0.003 and abs(tox - 0.9243) < 0.01
print("OK")
PY
```
Expected: prints `all-class acc=0.986x  toxin-only acc=0.924x` then `OK`.

- [ ] **Step 3: Commit refreshed benchmarks**

```bash
git add benchmark/test_set benchmark/val_set
git commit -m "data: refresh NN benchmark predictions with uncalibrated confidence"
```

### Task 2.2: Compute binary toxic/non-toxic metrics (closes blocker)

- [ ] **Step 1: Run the existing binary pipeline for both models**

```bash
uv run toxfam eval binary model/model_output/combined_run
uv run toxfam eval binary model/model_output/standard_run
```

- [ ] **Step 2: Verify binary_metrics.json now exists with sane values**

```bash
uv run python - <<'PY'
import json
d = json.load(open("model/model_output/combined_run/metrics/binary_metrics.json"))
td = d["test_default"]
print("ROC-AUC", td["roc_auc"], "PR-AUC", td["pr_auc"], "MCC", td["mcc"], "thr*", d["optimized_threshold"])
assert 0.95 <= td["roc_auc"] <= 1.0
PY
```
Expected: prints ROC-AUC/PR-AUC/MCC/threshold; assertion passes.

- [ ] **Step 3: Commit**

```bash
git add model/model_output/combined_run/metrics/binary_metrics.json \
        model/model_output/standard_run/metrics/binary_metrics.json \
        model/model_output/*/plots/binary_roc.png model/model_output/*/plots/binary_pr.png
git commit -m "eval: compute binary toxic/nontoxin metrics + ROC/PR for both models"
```

### Task 2.3: Regenerate non-metazoan OOD benchmark (binary)

- [ ] **Step 1: Run model eval on non_metazoan**

```bash
uv run toxfam eval model non_metazoan --model-dir model/model_output/combined_run
```

- [ ] **Step 2: Verify output exists and is binary**

```bash
uv run python - <<'PY'
import json, pandas as pd
m = json.load(open("benchmark/non_metazoan/nn_combined_run/run_metadata.json"))
assert m["task"] == "binary", m["task"]
p = pd.read_csv("benchmark/non_metazoan/nn_combined_run/predictions.csv")
print("non_metazoan n =", len(p))
PY
```

- [ ] **Step 3: Commit**

```bash
git add benchmark/non_metazoan
git commit -m "eval: regenerate non-metazoan OOD binary benchmark (combined model)"
```

### Task 2.4: Unreviewed inference-only (no labels → confidence distribution)

`unreviewed` has no labels TSV, so `toxfam eval model` cannot score it. Run inference-only.

- [ ] **Step 1: Create `analysis/figures/run_unreviewed_inference.py`**

```python
"""Inference-only on the unreviewed TrEMBL set (no labels available).

Writes analysis/manuscript_figures/unreviewed_predictions.csv with
identifier, predicted_label, confidence, confidence_uncalibrated, p_toxic.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd

from toxfam._paths import evaluation_data_dir, get_project_root
from toxfam.model.inference import run_inference

MODEL_DIR = get_project_root() / "model" / "model_output" / "combined_run"
OUT = get_project_root() / "analysis" / "manuscript_figures" / "unreviewed_predictions.csv"


def main() -> None:
    h5 = evaluation_data_dir() / "unreviewed" / "unreviewed.h5"
    with h5py.File(h5, "r") as f:
        ids = list(f.keys())
    df = pd.DataFrame({"identifier": ids})
    out = run_inference(df, h5, MODEL_DIR)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} unreviewed predictions to {OUT}")
    print(out["predicted_label"].value_counts().head(10))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python -m analysis.figures.run_unreviewed_inference`
Expected: prints row count + top predicted families; CSV written.

- [ ] **Step 3: Commit**

```bash
git add analysis/figures/run_unreviewed_inference.py analysis/manuscript_figures/unreviewed_predictions.csv
git commit -m "eval: unreviewed inference-only predictions (no labels)"
```

---

## Phase 3 — Shared figure helpers

### Task 3.1: `analysis/figures/_common.py` (loaders + style)

- [ ] **Step 1: Create `analysis/figures/_common.py`**

```python
"""Shared loaders and matplotlib style for manuscript figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from toxfam._paths import benchmark_dir, get_project_root, processed_dir

FIG_DIR = get_project_root() / "analysis" / "manuscript_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_preds(dataset: str, method: str) -> pd.DataFrame:
    return pd.read_csv(benchmark_dir() / dataset / method / "predictions.csv")


def test_class_list() -> list[str]:
    """The 38-class label space = sorted unique actual labels on the test set."""
    df = load_preds("test_set", "nn_combined_run")
    return sorted(df["actual_label"].unique().tolist())


def sequence_lengths() -> pd.Series:
    df = pd.read_csv(processed_dir() / "training_data.csv")
    return pd.Series(df["Sequence"].str.len().values, index=df["identifier"].values)


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save both PNG (300 dpi) and PDF (vector) into FIG_DIR."""
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}.png / .pdf")


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120, "savefig.bbox": "tight",
    })


# Consistent method colors/labels across all figures
METHODS = {
    "hbi": ("HBI", "#7f7f7f"),
    "nn_standard_run": ("ToxFam (emb)", "#5b9bd5"),
    "nn_combined_run": ("ToxFam (emb+tax)", "#c0504d"),
}
```

- [ ] **Step 2: Smoke-test the helpers**

Run:
```bash
uv run python -c "from analysis.figures._common import test_class_list, sequence_lengths; print(len(test_class_list()), 'classes;', len(sequence_lengths()), 'lengths')"
```
Expected: `38 classes; 65179 lengths`

- [ ] **Step 3: Commit**

```bash
git add analysis/figures/_common.py
git commit -m "feat(figures): shared loaders + style for manuscript figures"
```

---

## Phase 4 — Figure 1: Capability + validated superiority

File: Create `analysis/figures/figure1_capability.py`. Three panels (A capability map, B headline bars, C macro/weighted P/R/F1).

### Task 4.1: Figure 1 generation script

- [ ] **Step 1: Create `analysis/figures/figure1_capability.py`**

```python
"""Figure 1 — capability across 38 families + validated superiority over HBI."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.figures._common import apply_style, load_preds, save_fig, test_class_list
from toxfam.evaluation.manuscript import (
    correctness, mcnemar_test, paired_bootstrap_accuracy_diff, per_family_f1_difference,
    subset_accuracy, toxin_mask,
)
from toxfam.evaluation.metrics import calculate_metrics


def main() -> None:
    apply_style()
    classes = test_class_list()
    hbi = load_preds("test_set", "hbi")
    nn = load_preds("test_set", "nn_combined_run")
    std = load_preds("test_set", "nn_standard_run")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # --- Panel A: per-family support (log) vs NN F1 (capability across Metazoa) ---
    fam = per_family_f1_difference(nn, hbi, class_list=classes)  # has f1_a (NN), support
    fam = fam.sort_values("support", ascending=False)
    axA = axes[0]
    axA.scatter(fam["support"], fam["f1_a"], s=18, color="#c0504d", alpha=0.8)
    axA.set_xscale("log")
    axA.set_xlabel("Family support (test, log)"); axA.set_ylabel("ToxFam F1")
    axA.set_title(f"A. {len(fam)} toxin families resolved across Metazoa")
    axA.set_ylim(0, 1.02)

    # --- Panel B: toxin-only accuracy headline + all-class reference ---
    axB = axes[1]
    methods = [("HBI", hbi, "#7f7f7f"), ("ToxFam (emb+tax)", nn, "#c0504d")]
    tox = [subset_accuracy(d, toxin_mask(d)) for _, d, _ in methods]
    allc = [subset_accuracy(d) for _, d, _ in methods]
    x = np.arange(len(methods))
    axB.bar(x - 0.18, tox, 0.36, label="toxin-only", color=[c for *_, c in methods])
    axB.bar(x + 0.18, allc, 0.36, label="all-class", color=[c for *_, c in methods], alpha=0.45)
    for xi, (t, a) in enumerate(zip(tox, allc)):
        axB.text(xi - 0.18, t + 0.005, f"{t:.3f}", ha="center", fontsize=8)
        axB.text(xi + 0.18, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
    mc = mcnemar_test(correctness(nn), correctness(hbi))
    bs = paired_bootstrap_accuracy_diff(correctness(nn), correctness(hbi))
    axB.set_xticks(x); axB.set_xticklabels([m for m, *_ in methods])
    axB.set_ylim(0.8, 1.0); axB.set_ylabel("Accuracy"); axB.legend(loc="lower left")
    axB.set_title(f"B. Toxin-only headline (McNemar p={mc['p_value']:.3f};\n"
                  f"Δacc {bs['diff']:+.4f} [{bs['ci_low']:+.4f},{bs['ci_high']:+.4f}])")
    axB.text(0.5, 0.80, "non-toxin prior = 94.73%", transform=axB.transAxes,
             ha="center", va="bottom", fontsize=7, color="gray")

    # --- Panel C: macro & weighted P/R/F1 for the three methods ---
    axC = axes[2]
    rep = {n: calculate_metrics(d["actual_label"], d["predicted_label"], class_list=classes)
           for n, d in [("HBI", hbi), ("emb", std), ("emb+tax", nn)]}
    metrics = [("macro\nprec", "macro avg", "precision"), ("macro\nrecall", "macro avg", "recall"),
               ("macro\nF1", "macro avg", "f1-score"), ("weighted\nF1", "weighted avg", "f1-score")]
    width = 0.25
    for j, (mname, m) in enumerate(rep.items()):
        vals = [m.classification_report[avg][k] for _, avg, k in metrics]
        axC.bar(np.arange(len(metrics)) + (j - 1) * width, vals, width, label=mname)
    axC.set_xticks(np.arange(len(metrics))); axC.set_xticklabels([m for m, *_ in metrics])
    axC.set_ylim(0, 1.05); axC.legend(fontsize=7); axC.set_title("C. Macro / weighted P-R-F1")

    fig.tight_layout()
    save_fig(fig, "figure1_capability")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and eyeball**

Run: `uv run python -m analysis.figures.figure1_capability`
Expected: prints `saved figure1_capability.png / .pdf`; open the PNG and confirm Panel B shows toxin-only ≈0.924 (ToxFam) vs ≈0.854 (HBI) and the McNemar p≈0.003 in the title.

- [ ] **Step 3: Commit**

```bash
git add analysis/figures/figure1_capability.py analysis/manuscript_figures/figure1_capability.*
git commit -m "feat(figures): Figure 1 — capability + validated superiority"
```

---

## Phase 5 — Figure 2: Where homology breaks

File: Create `analysis/figures/figure2_homology.py`. Panels: A toxin-only rolling accuracy vs length (HBI+NN), B no-hit coverage (split toxin/non-toxin), C non-metazoan binary recognition.

### Task 5.1: Figure 2 generation script

- [ ] **Step 1: Create `analysis/figures/figure2_homology.py`**

```python
"""Figure 2 — ToxFam's advantage is concentrated where homology breaks."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from analysis.figures._common import apply_style, load_preds, save_fig, sequence_lengths
from toxfam.evaluation.manuscript import (
    accuracy_by_length_bins, rolling_accuracy_vs_length, subset_accuracy, toxin_mask,
)
from toxfam._paths import benchmark_dir
from toxfam.evaluation.hbi import NO_HIT_LABEL


def main() -> None:
    apply_style()
    hbi = load_preds("test_set", "hbi")
    nn = load_preds("test_set", "nn_combined_run")
    lengths = sequence_lengths()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # --- Panel A: toxin-only rolling accuracy vs length, HBI vs NN ---
    axA = axes[0]
    for d, (name, color) in ((hbi, ("HBI", "#7f7f7f")), (nn, ("ToxFam", "#c0504d"))):
        tox = d[toxin_mask(d)]
        roll = rolling_accuracy_vs_length(tox, lengths, window=50)
        axA.plot(roll["length"], roll["accuracy"], color=color, label=name, lw=1.6)
    axA.axvspan(0, 30, color="orange", alpha=0.08)
    axA.set_xscale("log"); axA.set_xlabel("Sequence length (aa, log)")
    axA.set_ylabel("Toxin-only accuracy"); axA.set_ylim(0.4, 1.02); axA.legend()
    # annotate the <30 aa collapse from fixed bins
    binsA = accuracy_by_length_bins(hbi[toxin_mask(hbi)], lengths, bins=[0, 30, 50, 75, 150, 5000])
    binsN = accuracy_by_length_bins(nn[toxin_mask(nn)], lengths, bins=[0, 30, 50, 75, 150, 5000])
    a30 = binsN.loc[binsN["bin_label"] == "0-30", "accuracy"].iloc[0]
    h30 = binsA.loc[binsA["bin_label"] == "0-30", "accuracy"].iloc[0]
    axA.set_title(f"A. <30 aa: HBI {h30:.3f} vs ToxFam {a30:.3f}")

    # --- Panel B: no-hit coverage, split toxin vs non-toxin ---
    axB = axes[1]
    nohit_ids = hbi.loc[hbi["predicted_label"] == NO_HIT_LABEL, "identifier"]
    nn_nh = nn[nn["identifier"].isin(nohit_ids)]
    tox_m = toxin_mask(nn_nh)
    groups = [("toxin no-hit", nn_nh[tox_m]), ("non-toxin no-hit", nn_nh[~tox_m])]
    labels, nn_acc, hbi_acc, ns = [], [], [], []
    for gname, g in groups:
        labels.append(f"{gname}\n(n={len(g)})"); nn_acc.append(subset_accuracy(g)); hbi_acc.append(0.0); ns.append(len(g))
    x = np.arange(len(groups))
    axB.bar(x - 0.2, hbi_acc, 0.4, label="HBI (no hit)", color="#7f7f7f")
    axB.bar(x + 0.2, nn_acc, 0.4, label="ToxFam", color="#c0504d")
    axB.set_xticks(x); axB.set_xticklabels(labels); axB.set_ylim(0, 1.05); axB.legend()
    axB.set_title(f"B. No-hit coverage (n={len(nn_nh)}: HBI 0% by construction)")

    # --- Panel C: non-metazoan binary recognition ---
    axC = axes[2]
    nm_dir = benchmark_dir() / "non_metazoan" / "nn_combined_run"
    if (nm_dir / "metrics.json").exists():
        nm = json.load(open(nm_dir / "metrics.json"))["numeric_metrics"]
        meta = json.load(open(nm_dir / "run_metadata.json"))
        axC.bar(["Accuracy", "MCC"], [nm["Test_Accuracy"], nm["Test_MCC"]], color="#c0504d")
        axC.set_ylim(0, 1.05)
        axC.set_title(f"C. Non-metazoan binary recognition (n={meta['n_samples']})")
    else:
        axC.text(0.5, 0.5, "non_metazoan benchmark missing\n(run Task 2.3)", ha="center")
        axC.axis("off")

    fig.tight_layout()
    save_fig(fig, "figure2_homology")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify Panel B n=74 split (10 toxin / 64 non-toxin)**

Run: `uv run python -m analysis.figures.figure2_homology`
Expected: `saved figure2_homology.*`; Panel B title shows `n=74`; bars labelled `toxin no-hit (n=10)` and `non-toxin no-hit (n=64)`.

- [ ] **Step 3: Commit**

```bash
git add analysis/figures/figure2_homology.py analysis/manuscript_figures/figure2_homology.*
git commit -m "feat(figures): Figure 2 — where homology breaks"
```

---

## Phase 6 — Figure 3: Per-family resolution + adjudication

File: Create `analysis/figures/figure3_perfamily.py`. Panel A per-family F1 difference + support stratification; Panel B adjudication stacked bar from Ivan's CSV.

### Task 6.1: Figure 3 generation script

- [ ] **Step 1: Create `analysis/figures/figure3_perfamily.py`**

```python
"""Figure 3 — per-family resolution + confident-error adjudication."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.figures._common import apply_style, load_preds, save_fig, test_class_list
from toxfam._paths import get_project_root
from toxfam.evaluation.manuscript import (
    adjudication_summary, macro_f1_by_support, per_family_f1_difference,
)

ADJ_CSV = get_project_root() / "analysis" / "model_test_wrong_conf_annotated.csv"


def main() -> None:
    apply_style()
    classes = test_class_list()
    hbi = load_preds("test_set", "hbi")
    nn = load_preds("test_set", "nn_combined_run")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: per-family F1 difference (NN - HBI), sorted, sized by support ---
    fam = per_family_f1_difference(nn, hbi, class_list=classes)
    axA = axes[0]
    colors = np.where(fam["diff"] >= 0, "#c0504d", "#7f7f7f")
    sizes = 10 + 3 * np.sqrt(fam["support"].clip(lower=1))
    axA.scatter(fam["diff"], np.arange(len(fam)), s=sizes, color=colors)
    axA.axvline(0, color="black", lw=0.6)
    axA.set_yticks(np.arange(len(fam))); axA.set_yticklabels(fam["family"], fontsize=6)
    axA.set_xlabel("F1 difference (ToxFam - HBI)")
    strat = macro_f1_by_support(nn, hbi, class_list=classes, support_threshold=5)
    sup = strat[strat["group"] == "support>5"].iloc[0]
    low = strat[strat["group"] == "support<=5"].iloc[0]
    axA.set_title(
        f"A. Per-family F1 (marker∝support)\n"
        f"support>5: ToxFam {sup['macro_f1_a']:.3f} vs HBI {sup['macro_f1_b']:.3f} | "
        f"support≤5 (n={low['n_sequences']}): {low['macro_f1_a']:.3f} vs {low['macro_f1_b']:.3f}"
    )

    # --- Panel B: confident-error adjudication stacked bar ---
    axB = axes[1]
    s = adjudication_summary(ADJ_CSV)
    order = ["correct", "partial", "incorrect"]
    counts = [s["assessment"].get(k, 0) for k in order]
    colors_b = ["#4caf50", "#ffb300", "#7f7f7f"]
    bottom = 0
    for k, c, col in zip(order, counts, colors_b):
        axB.bar(0, c, bottom=bottom, color=col, label=f"{k} ({c})"); bottom += c
    axB.set_xlim(-1, 1); axB.set_xticks([]); axB.set_ylabel("Confident (≥0.8) errors")
    axB.legend(loc="upper right", fontsize=8)
    axB.set_title(
        f"B. Adjudicated confident errors (n={s['n']})\n"
        f"{s['assessment'].get('correct',0)+s['assessment'].get('partial',0)}/{s['n']} model-vindicated; "
        f"{s['n_annotation_gaps']} candidate ToxProt gaps"
    )
    # worked examples annotation
    axB.text(0.0, -0.12, "e.g. P00601 (PLA2), F8J2F6 (Kunitz) — labelled nontox, absent from ToxProt",
             transform=axB.transAxes, ha="center", fontsize=7, color="gray")

    fig.tight_layout()
    save_fig(fig, "figure3_perfamily")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify the adjudication counts**

Run: `uv run python -m analysis.figures.figure3_perfamily`
Expected: Panel B title shows `n=63`, `43/63 model-vindicated`, `38 candidate ToxProt gaps`; Panel A title shows support>5 ToxFam ≈0.882 vs HBI ≈0.846.

- [ ] **Step 3: Commit**

```bash
git add analysis/figures/figure3_perfamily.py analysis/manuscript_figures/figure3_perfamily.*
git commit -m "feat(figures): Figure 3 — per-family resolution + adjudication"
```

---

## Phase 7 — Supplementary figures + numbers manifest

### Task 7.1: Identity-null + reliability/ECE + confusion-matrix supplementary script

- [ ] **Step 1: Create `analysis/figures/supplementary.py`**

```python
"""Supplementary figures: reliability/ECE + macro-F1 convention table.

(Identity-binned null and confusion matrices reuse existing analysis/plots and
model_output confusion matrices; this script adds the calibration + convention
artifacts that did not previously exist.)
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures._common import apply_style, load_preds, save_fig, test_class_list
from toxfam.evaluation.manuscript import binary_reliability, macro_f1_conventions
from toxfam.evaluation.metrics import to_binary_class


def reliability_panel() -> None:
    """Binary reliability diagram + ECE from non-metazoan/test p_toxic.

    Uses the test-set NN predictions' calibrated 'confidence' as a multiclass
    reliability proxy; for the binary head, recompute p_toxic via eval binary
    (model_output/.../metrics/binary_metrics.json holds AUROC/AUPRC).
    """
    apply_style()
    nn = load_preds("test_set", "nn_combined_run")
    # multiclass reliability from calibrated vs uncalibrated max-confidence
    correct = (nn["predicted_label"] == nn["actual_label"]).astype(float).values
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    for col, name, color in (("confidence", "calibrated", "#c0504d"),
                             ("confidence_uncalibrated", "uncalibrated", "#7f7f7f")):
        if col not in nn.columns:
            continue
        conf = nn[col].values
        edges = np.linspace(0, 1, 16)
        xs, ys, ece = [], [], 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.mean() > 0:
                xs.append(conf[m].mean()); ys.append(correct[m].mean())
                ece += abs(conf[m].mean() - correct[m].mean()) * m.mean()
        ax.plot(xs, ys, "o-", color=color, label=f"{name} (ECE={ece:.3f})", ms=3)
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.legend()
    ax.set_title("Reliability (multiclass top-class)")
    save_fig(fig, "supp_reliability")


def convention_table() -> None:
    """Write the macro-F1 no-hit convention values to a CSV for the manuscript."""
    classes = test_class_list()
    hbi = load_preds("test_set", "hbi")
    conv = macro_f1_conventions(hbi, class_list=classes)
    out = pd.DataFrame([
        {"method": "HBI", **conv},
    ])
    from analysis.figures._common import FIG_DIR
    out.to_csv(FIG_DIR / "supp_macro_f1_conventions.csv", index=False)
    print(out.to_string(index=False))


def main() -> None:
    reliability_panel()
    convention_table()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

Run: `uv run python -m analysis.figures.supplementary`
Expected: writes `supp_reliability.*`; prints HBI macro-F1 conventions (nohit_wrong ≈0.85 vs restricted ≈0.87) and writes `supp_macro_f1_conventions.csv`.

- [ ] **Step 3: Commit**

```bash
git add analysis/figures/supplementary.py analysis/manuscript_figures/supp_*
git commit -m "feat(figures): supplementary reliability/ECE + macro-F1 convention table"
```

### Task 7.2: Numbers manifest for the manuscript text

A single JSON of every number cited in Results, so the LaTeX prose and abstract quote one source of truth and the 63-vs-81 confident-error count is reconciled.

- [ ] **Step 1: Create `analysis/figures/numbers_manifest.py`**

```python
"""Emit analysis/manuscript_figures/results_numbers.json — every cited number."""
from __future__ import annotations

import json

from analysis.figures._common import FIG_DIR, load_preds, test_class_list
from toxfam.evaluation.manuscript import (
    adjudication_summary, correctness, macro_f1_by_support, macro_f1_conventions,
    mcnemar_test, paired_bootstrap_accuracy_diff, subset_accuracy, toxin_mask,
)
from toxfam._paths import get_project_root
from toxfam.evaluation.hbi import NO_HIT_LABEL

ADJ_CSV = get_project_root() / "analysis" / "model_test_wrong_conf_annotated.csv"


def main() -> None:
    classes = test_class_list()
    hbi = load_preds("test_set", "hbi"); nn = load_preds("test_set", "nn_combined_run")
    nohit = hbi["predicted_label"] == NO_HIT_LABEL
    nn_nh = nn[nn["identifier"].isin(hbi.loc[nohit, "identifier"])]
    out = {
        "n_test": int(len(nn)),
        "non_toxin_prior": round(float((nn["actual_label"].str.lower() == "nontox").mean()), 4),
        "toxin_only_n": int(toxin_mask(nn).sum()),
        "toxin_only_acc": {"nn_combined": subset_accuracy(nn, toxin_mask(nn)),
                            "hbi": subset_accuracy(hbi, toxin_mask(hbi))},
        "all_class_acc": {"nn_combined": subset_accuracy(nn), "hbi": subset_accuracy(hbi)},
        "mcnemar": mcnemar_test(correctness(nn), correctness(hbi)),
        "paired_bootstrap": paired_bootstrap_accuracy_diff(correctness(nn), correctness(hbi)),
        "no_hit": {"n": int(nohit.sum()), "n_toxin": int(toxin_mask(nn_nh).sum()),
                   "nn_acc": subset_accuracy(nn_nh)},
        "macro_f1_by_support": macro_f1_by_support(nn, hbi, class_list=classes).to_dict("records"),
        "macro_f1_conventions_hbi": macro_f1_conventions(hbi, class_list=classes),
        "adjudication": adjudication_summary(ADJ_CSV),
    }
    (FIG_DIR / "results_numbers.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and sanity-check against anchors**

Run: `uv run python -m analysis.figures.numbers_manifest`
Expected: `toxin_only_acc.nn_combined ≈ 0.9243`, `hbi ≈ 0.8544`; `mcnemar.b01=127, b10=83, p≈0.003`; `no_hit.n=74, n_toxin=10`; `adjudication.n=63`.

- [ ] **Step 3: Commit**

```bash
git add analysis/figures/numbers_manifest.py analysis/manuscript_figures/results_numbers.json
git commit -m "feat(figures): single-source numbers manifest for Results text"
```

---

## Phase 8 — Final verification

### Task 8.1: Full regeneration + test suite

- [ ] **Step 1: Run the whole test suite**

Run: `uv run pytest -q`
Expected: all tests pass (existing + new `tests/test_manuscript.py`).

- [ ] **Step 2: Regenerate all figures end-to-end**

```bash
uv run python -m analysis.figures.figure1_capability
uv run python -m analysis.figures.figure2_homology
uv run python -m analysis.figures.figure3_perfamily
uv run python -m analysis.figures.supplementary
uv run python -m analysis.figures.numbers_manifest
```
Expected: 5 figure PNG/PDF pairs + results_numbers.json in `analysis/manuscript_figures/`.

- [ ] **Step 3: Update the design doc status + commit**

Mark in `docs/results_section_plan.md` Section 5 that all four analysis tracks are executed (figures in `analysis/manuscript_figures/`), then:

```bash
git add docs/results_section_plan.md
git commit -m "docs: mark Results analysis tracks executed; figures generated"
```

- [ ] **Step 4: Open a PR (optional)**

```bash
git push -u origin results-analysis
gh pr create --title "Results analyses + manuscript figures" --body "Implements docs/results_analysis_plan.md: manuscript stats module (tested), refreshed benchmarks, binary/ECE, OOD, and Figures 1-3 + supplementary."
```

---

## Open items the author still owns (not automatable)
- **Abstract macro-F1 value:** decide whether to quote the no-hit-as-wrong (~0.85) or restricted (~0.87) HBI macro-F1; the manifest reports both.
- **63-vs-81 reconciliation:** the manifest fixes the adjudicated denominator at 63 (Ivan's CSV); confirm whether the remaining confident errors were intentionally excluded.
- **Taxonomy-fusion contribution:** if it is to be headlined as architectural, add a per-taxon/family breakdown of where taxonomy helps (PLA2 / Peptidase S1 / Insulin / CRISP) — a follow-up task, not in this plan.
- **Figure aesthetics:** these scripts produce correct, publication-grade-but-plain figures; final typography/paneling for the journal is a manual polish pass.
