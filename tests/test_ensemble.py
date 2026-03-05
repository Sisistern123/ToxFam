"""Tests for ensemble evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestEnsembleImports:
    def test_module_imports(self):
        from toxfam.evaluation.ensemble import evaluate_ensemble

        assert callable(evaluate_ensemble)

    def test_load_calibrated_model_import(self):
        from toxfam.evaluation.ensemble import _load_calibrated_model

        assert callable(_load_calibrated_model)

    def test_get_model_probs_import(self):
        from toxfam.evaluation.ensemble import _get_model_probs

        assert callable(_get_model_probs)
