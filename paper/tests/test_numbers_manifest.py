"""The manuscript numbers pipeline must refuse a pre-deploy (raw) binary head."""

from __future__ import annotations

import pytest

from paper.figures.numbers_manifest import (
    _require_deployed_binary_head,
    _require_matching_toxin_cohort,
)


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


def test_matching_toxin_cohort_passes():
    _require_matching_toxin_cohort(
        {"hbi_nontoxin_best_hit": {"n_toxins": 515}, "toxin_only_n": 515}
    )


def test_mismatched_toxin_cohort_is_refused():
    r"""A divergent cohort must abort, not emit a fraction with the wrong denominator.

    \HbiNontoxBestHit is counted on the HBI frame and \NumTox on the combined model's;
    the Discussion prints them as one fraction. If the NN path ever drops rows lacking
    an embedding, "68 of the 514" would read as a real number and be wrong.
    """
    with pytest.raises(SystemExit, match="toxin cohort mismatch"):
        _require_matching_toxin_cohort(
            {"hbi_nontoxin_best_hit": {"n_toxins": 515}, "toxin_only_n": 514}
        )
