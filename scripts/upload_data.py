#!/usr/bin/env python3
"""Upload raw and processed data to a GitHub Release.

Developer-only script — re-creates the data-v1 release with:
  - data/raw/0800.tsv             (frozen raw toxin data)
  - data/raw/nontox.tsv           (frozen raw non-toxin data)
  - data/processed/training_data.csv
  - data/processed/embeddings.h5
  - data/processed/hbi_train_all.csv
  - data/processed/hbi_train_all.fasta
  - data/intermediate/sp6/        (zipped as sp6_cache.zip)
  - data/evaluation/              (zipped as evaluation_data.zip)

Usage:
    uv run scripts/upload_data.py [--tag data-v1]

Requires the `gh` CLI (https://cli.github.com) to be installed and authenticated.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SP6_DIR = ROOT / "data" / "intermediate" / "sp6"
EVAL_DIR = ROOT / "data" / "evaluation"

REPO = "Sisistern123/ToxFam"
DEFAULT_TAG = "data-v1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tag", default=DEFAULT_TAG, help="GitHub release tag (default: %(default)s)"
    )
    parser.add_argument(
        "--notes-file",
        type=Path,
        default=None,
        help="markdown file with the release notes (split manifest hash, provenance)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="destroy an existing release AND its tag before recreating it "
        "(breaks reproducibility for every checkout pinned to that tag)",
    )
    parser.add_argument(
        "--prerelease", action="store_true", help="mark the release as a pre-release"
    )
    parser.add_argument(
        "--target",
        default=None,
        help="commit-ish the tag should point at (default: the repo's default branch)",
    )
    args = parser.parse_args()
    tag: str = args.tag

    # Overwriting a published tag makes every commit that pins it unreproducible:
    # the assets it downloads are silently replaced by different data. New artifacts
    # belong on a new tag.
    exists = (
        subprocess.run(
            ["gh", "release", "view", tag, "--repo", REPO], capture_output=True
        ).returncode
        == 0
    )
    if exists and not args.replace:
        print(
            f"ERROR: release '{tag}' already exists. Publish new data under a new tag "
            f"(and bump RELEASE_TAG in src/toxfam/cli.py) rather than overwriting it, "
            f"or pass --replace if you really mean to destroy the existing tag.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify source files
    sources = {
        "0800.tsv": RAW / "0800.tsv",
        "nontox.tsv": RAW / "nontox.tsv",
        "training_data.csv": PROCESSED / "training_data.csv",
        "embeddings.h5": PROCESSED / "embeddings.h5",
        "hbi_train_all.csv": PROCESSED / "hbi_train_all.csv",
        "hbi_train_all.fasta": PROCESSED / "hbi_train_all.fasta",
        "sp6 cache": SP6_DIR / "sp6_cache.json",
        "evaluation data": EVAL_DIR / "non_metazoan" / "non_metazoan.tsv",
    }
    for label, path in sources.items():
        if not path.exists():
            print(f"ERROR: {label} not found at {path}", file=sys.stderr)
            sys.exit(1)

    # Build zips
    tmp_dir = Path(tempfile.mkdtemp())

    sp6_zip = tmp_dir / "sp6_cache.zip"
    print("  zipping sp6 cache ...")
    with zipfile.ZipFile(sp6_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(SP6_DIR.rglob("*")):
            if file.is_file() and "_batch" not in file.parts:
                zf.write(file, file.relative_to(SP6_DIR))

    eval_zip = tmp_dir / "evaluation_data.zip"
    print("  zipping evaluation data ...")
    with zipfile.ZipFile(eval_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(EVAL_DIR.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(EVAL_DIR))

    try:
        if exists and args.replace:
            print(f"  --replace given: deleting existing release '{tag}' ...")
            subprocess.run(
                ["gh", "release", "delete", tag, "--yes", "--cleanup-tag"],
                capture_output=True,
            )

        # Create new release
        print(f"  creating release '{tag}' ...")
        subprocess.run(
            [
                "gh",
                "release",
                "create",
                tag,
                str(RAW / "0800.tsv"),
                str(RAW / "nontox.tsv"),
                str(PROCESSED / "training_data.csv"),
                str(PROCESSED / "embeddings.h5"),
                str(PROCESSED / "hbi_train_all.csv"),
                str(PROCESSED / "hbi_train_all.fasta"),
                str(sp6_zip),
                str(eval_zip),
                "--title",
                f"Data {tag.removeprefix('data-')}",
                "--notes",
                args.notes_file.read_text()
                if args.notes_file
                else "Download with `uv run toxfam download-data`.",
                *(["--prerelease"] if args.prerelease else []),
                *(["--target", args.target] if args.target else []),
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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
