#!/usr/bin/env python3
"""Upload raw and processed data to a GitHub Release.

Developer-only script — re-creates the current data release with:
  - data/raw/0800.tsv             (frozen raw toxin data)
  - data/raw/nontox.tsv           (frozen raw non-toxin data)
  - data/processed/training_data.csv
  - data/processed/embeddings.h5
  - data/processed/hbi_train_all.csv
  - data/processed/hbi_train_all.fasta
  - data/intermediate/sp6/        (zipped as sp6_cache.zip)
  - data/evaluation/              (zipped as evaluation_data.zip)

Usage:
    uv run scripts/upload_data.py [--tag data-v2]

Requires the `gh` CLI (https://cli.github.com) to be installed and authenticated.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

# Sibling import: works both when run as `uv run scripts/upload_data.py` and when
# the file is loaded by path (importlib), which does not put scripts/ on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gh_release import (  # noqa: E402
    add_release_args,
    create_release,
    guard_existing_tag,
)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SP6_DIR = ROOT / "data" / "intermediate" / "sp6"
EVAL_DIR = ROOT / "data" / "evaluation"

# Must track RELEASE_TAG in src/toxfam/cli.py — the tag `toxfam download-data`
# actually pulls from. Pinned by test_upload_default_tag_matches_download_tag.
DEFAULT_TAG = "data-v2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tag", default=DEFAULT_TAG, help="GitHub release tag (default: %(default)s)"
    )
    add_release_args(
        parser,
        notes_help="markdown file with the release notes "
        "(split manifest hash, provenance)",
    )
    args = parser.parse_args()
    tag: str = args.tag

    exists = guard_existing_tag(
        tag,
        replace=args.replace,
        remediation="Publish new data under a new tag (and bump RELEASE_TAG in "
        "src/toxfam/cli.py)",
    )

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
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

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

        create_release(
            tag,
            [
                RAW / "0800.tsv",
                RAW / "nontox.tsv",
                PROCESSED / "training_data.csv",
                PROCESSED / "embeddings.h5",
                PROCESSED / "hbi_train_all.csv",
                PROCESSED / "hbi_train_all.fasta",
                sp6_zip,
                eval_zip,
            ],
            title=f"Data {tag.removeprefix('data-')}",
            notes=args.notes_file.read_text()
            if args.notes_file
            else "Download with `uv run toxfam download-data`.",
            exists=exists,
            replace=args.replace,
            prerelease=args.prerelease,
            target=args.target,
        )


if __name__ == "__main__":
    main()
