"""Tests for toxfam._paths."""

from __future__ import annotations

from toxfam._paths import (
    data_dir,
    get_project_root,
    intermediate_dir,
    processed_dir,
    raw_dir,
)


def test_project_root_exists():
    root = get_project_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists()


def test_data_dirs_are_under_root():
    root = get_project_root()
    assert data_dir() == root / "data"
    assert raw_dir() == root / "data" / "raw"
    assert intermediate_dir() == root / "data" / "intermediate"
    assert processed_dir() == root / "data" / "processed"
