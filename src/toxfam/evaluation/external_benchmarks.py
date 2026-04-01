"""External tool benchmarks: run ToxinPred2 and ToxinPred3 on our test set.

Compares external predictions against our binary ground truth using the same
metrics as our internal models.

ToxinPred2's CLI is buggy with modern pandas, so we call its core functions
(AAC computation + ONNX inference) directly for Model 1 (AAC-RF).

ToxinPred3 uses AAC + DPC (dipeptide) features with Extra Trees. We call
the joblib model directly, bypassing the CLI (same file I/O issues).
Model 2 (Hybrid with MERCI motifs) requires perl and is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from toxfam._paths import get_project_root
from toxfam.evaluation.metrics import (
    calculate_binary_metrics_with_scores,
    to_binary_class,
)


def _compute_aac(sequences: list[str]) -> np.ndarray:
    """Compute amino acid composition (20-dim) for each sequence.

    Replicates ToxinPred2's aac_comp function without file I/O.
    """
    std = list("ACDEFGHIKLMNPQRSTVWY")
    n = len(sequences)
    aac = np.zeros((n, 20), dtype=np.float32)

    for i, seq in enumerate(sequences):
        seq_upper = seq.upper()
        seq_len = len(seq_upper)
        if seq_len == 0:
            continue
        for j, aa in enumerate(std):
            count = seq_upper.count(aa)
            aac[i, j] = (count / seq_len) * 100

    return aac


def run_toxinpred2_benchmark(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    output_dir: Path,
) -> dict:
    """Run ToxinPred2 Model 1 (AAC-RF) on test sequences via direct ONNX inference.

    Parameters
    ----------
    test_df : DataFrame with identifier and Sequence columns.
    y_true : binary ground truth (1=toxic, 0=nontoxic).
    output_dir : where to save results.

    Returns dict with binary metrics, or empty dict on failure.
    """
    try:
        import onnxruntime
    except ImportError:
        print("onnxruntime not installed. Install with: uv add onnxruntime")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find ToxinPred2's ONNX model
    try:
        import toxinpred2.python_scripts.toxinpred2 as tp2_mod
        tp2_script = Path(tp2_mod.__file__)
        model_path = tp2_script.parent.parent / "model" / "RF_model.onnx"
        if not model_path.exists():
            print(f"ToxinPred2 ONNX model not found at {model_path}")
            return {}
    except (ImportError, TypeError, AttributeError):
        print("ToxinPred2 not installed. Install with: uv add toxinpred2")
        return {}

    sequences = test_df["Sequence"].tolist()
    print(f"Running ToxinPred2 (AAC-RF) on {len(sequences)} sequences...")

    # 1. Compute amino acid composition
    aac = _compute_aac(sequences)
    print(f"  AAC features computed: shape {aac.shape}")

    # 2. ONNX inference
    sess = onnxruntime.InferenceSession(str(model_path))
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[1].name  # probability output

    scores_raw = sess.run([label_name], {input_name: aac})[0]
    # Output is a list of dicts: [{0: p_nontox, 1: p_toxic}, ...]
    if isinstance(scores_raw, list) and isinstance(scores_raw[0], dict):
        p_toxic = np.array([d[1] for d in scores_raw], dtype=np.float64)
    else:
        # Fallback: numpy array shape (N, 2)
        scores_arr = np.array(scores_raw)
        p_toxic = scores_arr[:, -1].astype(np.float64)

    print(f"  Predictions: {(p_toxic >= 0.5).sum()} toxic, {(p_toxic < 0.5).sum()} non-toxic")

    metrics = calculate_binary_metrics_with_scores(y_true, p_toxic)

    # Save
    serializable = {
        k: v for k, v in metrics.items()
        if k not in ("fpr", "tpr", "precision_curve", "recall_curve",
                      "roc_thresholds", "pr_thresholds")
    }
    serializable["model"] = "AAC-RF (Model 1)"
    serializable["n_predictions"] = len(sequences)

    (output_dir / "toxinpred2_model1_metrics.json").write_text(
        json.dumps(serializable, indent=4)
    )

    print(
        f"  ToxinPred2 AAC-RF: "
        f"ROC-AUC={metrics['roc_auc']:.4f}, "
        f"PR-AUC={metrics['pr_auc']:.4f}, "
        f"MCC={metrics['mcc']:.4f}"
    )

    return metrics


def _compute_dpc(sequences: list[str]) -> np.ndarray:
    """Compute dipeptide composition (400-dim) for each sequence.

    Replicates ToxinPred3's dpc_comp function without file I/O.
    """
    std = list("ACDEFGHIKLMNPQRSTVWY")
    n = len(sequences)
    dpc = np.zeros((n, 400), dtype=np.float32)

    for i, seq in enumerate(sequences):
        seq_upper = seq.upper()
        seq_len = len(seq_upper)
        if seq_len < 2:
            continue
        idx = 0
        for j_aa in std:
            for k_aa in std:
                dipep = j_aa + k_aa
                count = 0
                for m in range(seq_len - 1):
                    if seq_upper[m:m + 2] == dipep:
                        count += 1
                dpc[i, idx] = (count / (seq_len - 1)) * 100
                idx += 1

    return dpc


def _find_toxinpred3_model() -> Path | None:
    """Find ToxinPred3's model pkl file."""
    try:
        import toxinpred3.python_scripts.toxinpred3 as tp3_mod
        tp3_script = Path(tp3_mod.__file__)
        model_path = tp3_script.parent.parent / "model" / "toxinpred3.0_model.pkl"
        if model_path.exists():
            return model_path
    except (ImportError, TypeError, AttributeError):
        pass
    return None


def _run_toxinpred3_isolated(
    test_df: pd.DataFrame,
    model_path: Path,
    output_dir: Path,
) -> np.ndarray | None:
    """Run ToxinPred3 via isolated venv with sklearn 1.0.2.

    Creates a temporary venv, installs sklearn 1.0.2, and runs the model
    via subprocess to avoid version conflicts with the main environment.
    """
    import shutil
    import subprocess
    import tempfile

    root = get_project_root()
    wrapper_script = root / "scripts" / "toxinpred3_isolated.py"
    if not wrapper_script.exists():
        print("  Wrapper script not found at scripts/toxinpred3_isolated.py")
        return None

    # Check for isolated venv
    isolated_venv = root / ".toxinpred3_env"
    isolated_python = isolated_venv / "bin" / "python"

    if not isolated_python.exists():
        print("  Creating isolated ToxinPred3 environment (one-time setup)...")

        # Check if uv is available for venv creation
        uv_path = shutil.which("uv")
        if uv_path is None:
            print("  uv not available for creating isolated venv. Skipping.")
            return None

        try:
            subprocess.run(
                [uv_path, "venv", str(isolated_venv), "--python", "3.10"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [uv_path, "pip", "install",
                 "--python", str(isolated_python),
                 "scikit-learn==1.0.2", "joblib", "numpy<2", "pandas"],
                check=True,
                capture_output=True,
            )
            print("  Isolated environment created.")
        except subprocess.CalledProcessError as e:
            print(f"  Failed to create isolated environment: {e}")
            # Clean up partial venv
            if isolated_venv.exists():
                shutil.rmtree(isolated_venv)
            return None

    # Write sequences to temp CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "input.csv"
        output_csv = Path(tmpdir) / "output.csv"

        test_df[["identifier", "Sequence"]].to_csv(input_csv, index=False)

        try:
            result = subprocess.run(
                [
                    str(isolated_python),
                    str(wrapper_script),
                    "--input", str(input_csv),
                    "--output", str(output_csv),
                    "--model-path", str(model_path),
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                print(f"  Isolated inference failed: {result.stderr}")
                return None

            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    print(f"  {line}")

            pred_df = pd.read_csv(output_csv)
            return pred_df["p_toxic"].to_numpy()

        except subprocess.TimeoutExpired:
            print("  Isolated inference timed out (10min limit)")
            return None
        except Exception as e:
            print(f"  Isolated inference error: {e}")
            return None


def run_toxinpred3_benchmark(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    output_dir: Path,
) -> dict:
    """Run ToxinPred3 Model 1 (AAC+DPC Extra Trees).

    Tries direct joblib loading first. If sklearn version is incompatible,
    falls back to running inference in an isolated venv with sklearn 1.0.2.

    Returns dict with binary metrics, or empty dict on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = _find_toxinpred3_model()
    if model_path is None:
        print("ToxinPred3 not installed. Install with: uv add toxinpred3")
        return {}

    sequences = test_df["Sequence"].tolist()
    print(f"Running ToxinPred3 (AAC+DPC ET) on {len(sequences)} sequences...")

    # Try 1: Direct joblib loading
    p_toxic = None
    try:
        import joblib
        clf = joblib.load(str(model_path))

        aac = _compute_aac(sequences)
        dpc = _compute_dpc(sequences)
        features = np.concatenate([aac, dpc], axis=1)
        p_scores = clf.predict_proba(features)
        p_toxic = p_scores[:, -1].astype(np.float64)
        print("  Loaded model directly (sklearn compatible)")

    except (ValueError, ModuleNotFoundError, ImportError):
        print("  Direct loading failed (sklearn version mismatch)")
        print("  Trying isolated environment...")
        p_toxic = _run_toxinpred3_isolated(test_df, model_path, output_dir)

    if p_toxic is None:
        print("  ToxinPred3 benchmark could not be run.")
        print("  To enable: create isolated env with sklearn 1.0.2:")
        print("    uv venv .toxinpred3_env --python 3.10")
        print("    .toxinpred3_env/bin/pip install scikit-learn==1.0.2 joblib numpy<2 pandas")
        return {}

    print(f"  Predictions: {(p_toxic >= 0.38).sum()} toxic, {(p_toxic < 0.38).sum()} non-toxic (threshold=0.38)")

    metrics = calculate_binary_metrics_with_scores(y_true, p_toxic)

    # Save
    serializable = {
        k: v for k, v in metrics.items()
        if k not in ("fpr", "tpr", "precision_curve", "recall_curve",
                      "roc_thresholds", "pr_thresholds")
    }
    serializable["model"] = "AAC+DPC Extra Trees (Model 1)"
    serializable["n_predictions"] = len(sequences)

    (output_dir / "toxinpred3_model1_metrics.json").write_text(
        json.dumps(serializable, indent=4)
    )

    print(
        f"  ToxinPred3 AAC+DPC ET: "
        f"ROC-AUC={metrics['roc_auc']:.4f}, "
        f"PR-AUC={metrics['pr_auc']:.4f}, "
        f"MCC={metrics['mcc']:.4f}"
    )

    return metrics


def run_all_external_benchmarks(
    input_csv: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run all available external benchmarks on the test set."""
    root = get_project_root()

    if input_csv is None:
        input_csv = root / "data" / "processed" / "training_data.csv"
    if output_dir is None:
        output_dir = root / "model" / "model_output" / "external_benchmarks"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    df = pd.read_csv(input_csv)
    test_df = df[df["Split"] == "test"].copy()

    test_df["is_toxic"] = test_df["Protein families"].apply(
        lambda x: 0 if to_binary_class(x) == "nontoxin" else 1
    )
    y_true = test_df["is_toxic"].to_numpy()

    print(f"Test set: {len(test_df)} sequences ({y_true.sum()} toxic, {len(y_true) - y_true.sum()} nontox)")

    all_results = {}

    # ToxinPred2 Model 1 (AAC-RF) — direct ONNX inference
    print("\n--- ToxinPred2 Model 1 (AAC-RF) ---")
    m1 = run_toxinpred2_benchmark(test_df, y_true, output_dir)
    if m1:
        all_results["toxinpred2_model1"] = m1

    # ToxinPred3 Model 1 (AAC+DPC Extra Trees) — direct joblib inference
    print("\n--- ToxinPred3 Model 1 (AAC+DPC ET) ---")
    m3 = run_toxinpred3_benchmark(test_df, y_true, output_dir)
    if m3:
        all_results["toxinpred3_model1"] = m3

    # TOXIFY (reimplemented) — Atchley GRU trained from scratch
    print("\n--- TOXIFY (reimplemented, Atchley GRU) ---")
    from toxfam.evaluation.toxify_benchmark import run_toxify_benchmark

    m_tox = run_toxify_benchmark(test_df, y_true, output_dir)
    if m_tox:
        all_results["toxify"] = m_tox

    # Summary
    print("\n" + "=" * 60)
    print("EXTERNAL BENCHMARK SUMMARY")
    print("=" * 60)
    for name, m in all_results.items():
        roc = m.get("roc_auc", "—")
        pr = m.get("pr_auc", "—")
        mcc = m.get("mcc", "—")

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        print(f"  {name:<25} ROC-AUC={fmt(roc)}  PR-AUC={fmt(pr)}  MCC={fmt(mcc)}")

    return all_results
