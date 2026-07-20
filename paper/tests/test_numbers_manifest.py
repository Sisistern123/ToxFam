"""The manuscript numbers pipeline must refuse a pre-deploy (raw) binary head."""

from __future__ import annotations

import pytest

from paper.figures.numbers_manifest import _require_deployed_binary_head


def test_deployed_binary_head_passes():
    # score_space=="platt_calibrated" is the deployed head — allowed through.
    _require_deployed_binary_head(
        {"score_space": "platt_calibrated"}, "binary_metrics.json", "combined_run"
    )


def test_raw_binary_head_is_refused():
    with pytest.raises(SystemExit, match="platt_calibrated"):
        _require_deployed_binary_head(
            {"score_space": "raw"}, "binary_metrics.json", "combined_run"
        )


def test_missing_score_space_is_refused():
    with pytest.raises(SystemExit, match="platt_calibrated"):
        _require_deployed_binary_head({}, "binary_metrics.json", "combined_run")
