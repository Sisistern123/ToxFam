"""Shared `gh release` plumbing for the developer upload scripts.

`upload_data.py` and `package_models.py` publish different payloads to different
tags, but the publishing *rules* are identical: never silently overwrite a tag,
because every checkout pinned to it downloads different bytes afterwards and
stops being reproducible. Keeping the rule in one place means a change to it
(the `--replace` guard, `--prerelease`, `--target`) lands once rather than
twice-and-drifting.

Developer-only: not part of the installable `toxfam` wheel.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = "Sisistern123/ToxFam"


def add_release_args(parser: argparse.ArgumentParser, *, notes_help: str) -> None:
    """Add the release-publishing flags shared by both scripts."""
    parser.add_argument("--notes-file", type=Path, default=None, help=notes_help)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="destroy an existing release AND its tag before recreating it "
        "(breaks reproducibility for anything pinned to that tag)",
    )
    parser.add_argument(
        "--prerelease", action="store_true", help="mark the release as a pre-release"
    )
    parser.add_argument(
        "--target",
        default=None,
        help="commit-ish the tag should point at (default: the repo's default branch)",
    )


def release_exists(tag: str) -> bool:
    return (
        subprocess.run(
            ["gh", "release", "view", tag, "--repo", REPO], capture_output=True
        ).returncode
        == 0
    )


def guard_existing_tag(tag: str, *, replace: bool, remediation: str) -> bool:
    """Exit unless publishing to *tag* is safe. Returns whether the release exists.

    *remediation* names the caller-specific way to publish under a new tag.
    """
    exists = release_exists(tag)
    if exists and not replace:
        print(
            f"ERROR: release '{tag}' already exists. {remediation} rather than "
            f"overwriting it, or pass --replace if you really mean to destroy the "
            f"existing tag and its assets.",
            file=sys.stderr,
        )
        sys.exit(1)
    return exists


def create_release(
    tag: str,
    assets: list[Path],
    *,
    title: str,
    notes: str,
    exists: bool,
    replace: bool,
    prerelease: bool = False,
    target: str | None = None,
) -> None:
    """Create the release at *tag*, replacing it first when asked to.

    *exists* is the value returned by :func:`guard_existing_tag`, which must have
    already approved the publish.
    """
    try:
        if exists and replace:
            print(f"  --replace given: deleting existing release '{tag}' ...")
            subprocess.run(
                ["gh", "release", "delete", tag, "--yes", "--cleanup-tag"],
                capture_output=True,
            )

        print(f"  creating release '{tag}' ...")
        subprocess.run(
            [
                "gh",
                "release",
                "create",
                tag,
                *(str(a) for a in assets),
                "--repo",
                REPO,
                "--title",
                title,
                "--notes",
                notes,
                *(["--prerelease"] if prerelease else []),
                *(["--target", target] if target else []),
            ],
            check=True,
        )
        print("Done.")
    except FileNotFoundError:
        print(
            "ERROR: `gh` CLI not found. Install from https://cli.github.com",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
