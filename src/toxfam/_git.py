"""Git provenance helpers.

A leaf module (no toxfam imports), so the light ``data`` layer and the
``evaluation`` layer can both stamp run artifacts without depending on one
another.
"""

import subprocess


def git_commit_short() -> str:
    """Short HEAD hash, or ``"unknown"`` outside a checkout / without git."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """True if the working tree has uncommitted changes.

    A bare short SHA silently hides that results were produced from a modified
    tree; this flags it so provenance is not misleading.
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except Exception:
        return False
