"""Project root detection and standard directory helpers.

The root is the repo checkout (found by walking up to ``pyproject.toml``); all
data/model/benchmark paths hang off it. Set ``TOXFAM_ROOT`` to override when
running from outside a checkout. Cached once per process (``lru_cache``), so the
env var is read a single time.
"""

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    override = os.environ.get("TOXFAM_ROOT")
    if override:
        root = Path(override).resolve()
        if not root.exists():
            raise RuntimeError(
                f"TOXFAM_ROOT is set to '{override}', but that path does not exist."
            )
        return root
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(
        "Could not locate the ToxFam project root (no pyproject.toml found above "
        f"{Path(__file__).resolve()}). Run from a cloned checkout, or set "
        "TOXFAM_ROOT to the repo path."
    )


def data_dir() -> Path:
    return get_project_root() / "data"


def raw_dir() -> Path:
    return data_dir() / "raw"


def intermediate_dir() -> Path:
    return data_dir() / "intermediate"


def processed_dir() -> Path:
    return data_dir() / "processed"


def evaluation_data_dir() -> Path:
    return data_dir() / "evaluation"


def benchmark_dir() -> Path:
    return get_project_root() / "benchmark"
