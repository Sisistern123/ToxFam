"""Tests for the evaluation runner's compare_methods schema robustness.

The score-based external-tool benchmark and the package eval methods both write
into benchmark/{dataset}/<method>/, but with different metrics.json schemas.
compare_methods must skip foreign metrics.json instead of crashing.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from toxfam.evaluation import runner


def _write_method(
    method_dir: Path, *, numeric: bool, ids: list[str] | None = None
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    if numeric:  # a toxfam eval method
        metrics = {
            "numeric_metrics": {
                "Test_Accuracy": 0.9,
                "Test_MCC": 0.8,
                "Test_Micro_MCC": 0.9,
                "Test_Std_Error": 0.01,
            },
            "classification_report": {},
        }
    else:  # foreign (external-tool) score-based schema
        metrics = {"method": "ToxFam (emb+tax)", "test_default": {}, "test_optimized": {}}
    (method_dir / "metrics.json").write_text(json.dumps(metrics))
    pd.DataFrame({"identifier": ids if ids is not None else ["a", "b"]}).to_csv(
        method_dir / "predictions.csv", index=False
    )


def test_compare_methods_skips_foreign_metrics(tmp_path, monkeypatch):
    bench = tmp_path / "benchmark"
    dataset = "test_set"
    _write_method(bench / dataset / "eat", numeric=True)
    _write_method(bench / dataset / "toxfam_embtax", numeric=False)  # foreign schema
    monkeypatch.setattr(runner, "benchmark_dir", lambda: bench)

    summary = runner.compare_methods(dataset)  # must not raise KeyError

    methods = set(summary["Method"])
    assert "eat" in methods
    assert "toxfam_embtax" not in methods

    full = json.loads((bench / dataset / "comparison" / "full_report.json").read_text())
    assert "eat" in full
    assert "toxfam_embtax" not in full


def test_compare_methods_rejects_different_protein_sets(tmp_path, monkeypatch):
    """Two methods run against two different versions of "the test set" report the
    same sample count while sharing few proteins. Row counts cannot see it; only
    identifier-set equality can. This is how an April HBI run was once tabulated
    against a June model run sharing 21% of its proteins.
    """
    bench = tmp_path / "benchmark"
    dataset = "test_set"
    _write_method(bench / dataset / "hbi", numeric=True, ids=["a", "b", "c"])
    _write_method(bench / dataset / "nn_combined", numeric=True, ids=["a", "x", "y"])
    monkeypatch.setattr(runner, "benchmark_dir", lambda: bench)

    with pytest.raises(ValueError, match="different protein sets"):
        runner.compare_methods(dataset)


def test_compare_methods_accepts_identical_protein_sets(tmp_path, monkeypatch):
    bench = tmp_path / "benchmark"
    dataset = "test_set"
    # Same proteins, different row order — order must not matter.
    _write_method(bench / dataset / "hbi", numeric=True, ids=["a", "b", "c"])
    _write_method(bench / dataset / "nn_combined", numeric=True, ids=["c", "a", "b"])
    monkeypatch.setattr(runner, "benchmark_dir", lambda: bench)

    summary = runner.compare_methods(dataset)
    assert set(summary["Method"]) == {"hbi", "nn_combined"}
