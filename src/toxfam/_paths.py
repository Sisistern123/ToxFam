from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate project root")


def data_dir() -> Path:
    return get_project_root() / "data"


def configs_dir() -> Path:
    return get_project_root() / "configs"
