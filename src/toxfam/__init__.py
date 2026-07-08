"""ToxFam — animal toxin protein family classification on ProtT5 embeddings."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("toxfam")
except PackageNotFoundError:  # running from a bare source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
