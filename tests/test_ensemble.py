"""Tests for ensemble evaluation module."""


def test_ensemble_module_imports():
    from toxfam.evaluation.ensemble import evaluate_ensemble

    assert callable(evaluate_ensemble)
