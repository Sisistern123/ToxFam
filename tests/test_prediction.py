"""Tests for the pure helpers behind `toxfam predict`.

`prediction.py` is the flagship user-facing path but its GPU-free seams —
identifier normalization, organism-mask routing (which model predicts each
protein), and the binary-threshold fallback (the toxic/non-toxic call) — were
untested. A regression in any of these silently produces wrong predictions, so
these cheap in-memory tests guard the correctness gates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from toxfam.prediction import (
    _organism_mask,
    _read_input,
    _read_optimized_threshold,
    _suffixed,
)

# --------------------------------------------------------------------------- #
# _read_input                                                                  #
# --------------------------------------------------------------------------- #


def _write_tsv(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, sep="\t", index=False)
    return path


def test_read_input_keeps_identifier(tmp_path):
    tsv = _write_tsv(
        tmp_path / "in.tsv",
        pd.DataFrame({"identifier": ["P1", "P2"], "Sequence": ["MK", "AC"]}),
    )
    df = _read_input(tsv)
    assert list(df["identifier"]) == ["P1", "P2"]


def test_read_input_renames_entry_to_identifier(tmp_path):
    tsv = _write_tsv(
        tmp_path / "in.tsv",
        pd.DataFrame({"Entry": ["P1", "P2"], "Sequence": ["MK", "AC"]}),
    )
    df = _read_input(tsv)
    assert "identifier" in df.columns
    assert "Entry" not in df.columns
    assert list(df["identifier"]) == ["P1", "P2"]


def test_read_input_prefers_existing_identifier_over_entry(tmp_path):
    """If both columns exist, identifier is kept and Entry left untouched."""
    tsv = _write_tsv(
        tmp_path / "in.tsv",
        pd.DataFrame({"Entry": ["X1"], "identifier": ["P1"]}),
    )
    df = _read_input(tsv)
    assert list(df["identifier"]) == ["P1"]
    assert "Entry" in df.columns  # not renamed away


def test_read_input_missing_identifier_raises(tmp_path):
    tsv = _write_tsv(tmp_path / "in.tsv", pd.DataFrame({"Sequence": ["MK"]}))
    with pytest.raises(ValueError, match="identifier"):
        _read_input(tsv)


# --------------------------------------------------------------------------- #
# _organism_mask                                                               #
# --------------------------------------------------------------------------- #


def test_organism_mask_missing_column_is_all_false():
    df = pd.DataFrame({"identifier": ["P1", "P2"]})
    mask = _organism_mask(df)
    assert mask.tolist() == [False, False]
    assert list(mask.index) == list(df.index)


def test_organism_mask_numeric_vs_nonnumeric():
    df = pd.DataFrame(
        {
            "identifier": ["P1", "P2", "P3", "P4"],
            "Organism (ID)": [9606, "0", "not-a-taxid", None],
        }
    )
    mask = _organism_mask(df)
    # 9606 and "0" coerce to numbers -> usable; the string and None do not.
    assert mask.tolist() == [True, True, False, False]


def test_organism_mask_preserves_index_after_filtering():
    """The mask must align with a non-default index so df[mask] routes correctly."""
    df = pd.DataFrame(
        {"identifier": ["P1", "P2"], "Organism (ID)": [9606, None]},
        index=[10, 20],
    )
    mask = _organism_mask(df)
    assert list(mask.index) == [10, 20]
    assert list(df[mask]["identifier"]) == ["P1"]


# --------------------------------------------------------------------------- #
# _read_optimized_threshold                                                    #
# --------------------------------------------------------------------------- #


def _write_binary_metrics(model_dir: Path, payload) -> None:
    metrics_dir = model_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "binary_metrics.json").write_text(payload)


def test_threshold_reads_optimized_value(tmp_path):
    _write_binary_metrics(tmp_path, json.dumps({"optimized_threshold": 0.73}))
    assert _read_optimized_threshold(tmp_path) == pytest.approx(0.73)


def test_threshold_missing_file_defaults_to_half(tmp_path):
    assert _read_optimized_threshold(tmp_path) == 0.5


def test_threshold_missing_key_defaults_to_half(tmp_path):
    _write_binary_metrics(tmp_path, json.dumps({"something_else": 1.0}))
    assert _read_optimized_threshold(tmp_path) == 0.5


def test_threshold_malformed_json_defaults_to_half(tmp_path):
    _write_binary_metrics(tmp_path, "{not valid json")
    assert _read_optimized_threshold(tmp_path) == 0.5


def _write_binary_calibrator(model_dir: Path, threshold: float) -> None:
    models = model_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "binary_calibrator.json").write_text(
        json.dumps(
            {
                "a": 0.7,
                "b": -2.3,
                "eps": 1e-6,
                "threshold": threshold,
                "threshold_space": "platt",
            }
        )
    )


def test_threshold_prefers_deployed_calibrator(tmp_path):
    """When the deployed calibrator is present, its (calibrated-space) threshold wins."""
    _write_binary_metrics(tmp_path, json.dumps({"optimized_threshold": 0.73}))
    _write_binary_calibrator(tmp_path, threshold=0.31)
    assert _read_optimized_threshold(tmp_path) == pytest.approx(0.31)


def test_threshold_falls_back_to_binary_metrics_without_calibrator(tmp_path):
    _write_binary_metrics(tmp_path, json.dumps({"optimized_threshold": 0.73}))
    assert _read_optimized_threshold(tmp_path) == pytest.approx(0.73)


def test_threshold_malformed_calibrator_falls_back(tmp_path):
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "binary_calibrator.json").write_text("{bad json")
    _write_binary_metrics(tmp_path, json.dumps({"optimized_threshold": 0.73}))
    assert _read_optimized_threshold(tmp_path) == pytest.approx(0.73)


def test_threshold_calibrated_metrics_without_calibrator_is_half(tmp_path):
    """A calibrated-space threshold with no calibrator would be applied to the raw
    P(toxic) — score-space mismatch; degrade to 0.5 rather than reuse it."""
    _write_binary_metrics(
        tmp_path,
        json.dumps({"optimized_threshold": 0.03, "score_space": "platt_calibrated"}),
    )
    assert _read_optimized_threshold(tmp_path) == 0.5


# --------------------------------------------------------------------------- #
# _suffixed                                                                    #
# --------------------------------------------------------------------------- #


def test_suffixed_inserts_tag_before_suffix():
    assert _suffixed(Path("out/preds.tsv"), "combined") == Path(
        "out/preds_combined.tsv"
    )


def test_suffixed_handles_no_suffix():
    assert _suffixed(Path("preds"), "standard") == Path("preds_standard")
