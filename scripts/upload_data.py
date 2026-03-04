#!/usr/bin/env python3
"""Upload processed data to a GitHub Release.

Developer-only script — re-creates the data-v1 release with:
  - data/processed/training_data.csv
  - data/processed/embeddings.h5  (uploaded as training_data.h5)
  - data/intermediate/sp6/        (zipped as sp6_cache.zip)

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
PROCESSED = ROOT / "data" / "processed"
SP6_DIR = ROOT / "data" / "intermediate" / "sp6"

REPO = "Sisistern123/ToxFam"
DEFAULT_TAG = "data-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default=DEFAULT_TAG, help="GitHub release tag (default: %(default)s)")
    args = parser.parse_args()
    tag: str = args.tag

    # Verify source files
    sources = {
        "training_data.csv": PROCESSED / "training_data.csv",
        "embeddings.h5": PROCESSED / "embeddings.h5",
        "sp6 cache": SP6_DIR / "sp6_cache.json",
    }
    for label, path in sources.items():
        if not path.exists():
            print(f"ERROR: {label} not found at {path}", file=sys.stderr)
            sys.exit(1)

    # Build SP6 zip
    tmp_dir = Path(tempfile.mkdtemp())
    sp6_zip = tmp_dir / "sp6_cache.zip"
    print("  zipping sp6 cache ...")
    with zipfile.ZipFile(sp6_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(SP6_DIR.rglob("*")):
            if file.is_file() and "_batch" not in file.parts:
                zf.write(file, file.relative_to(SP6_DIR))

    try:
        # Delete old release (ignore errors if it doesn't exist)
        print(f"  deleting old release '{tag}' ...")
        subprocess.run(
            ["gh", "release", "delete", tag, "--yes", "--cleanup-tag"],
            capture_output=True,
        )

        # Create new release
        print(f"  creating release '{tag}' ...")
        subprocess.run(
            [
                "gh", "release", "create", tag,
                str(PROCESSED / "training_data.csv"),
                str(PROCESSED / "embeddings.h5"),
                str(sp6_zip),
                "--title", "Processed Data v1",
                "--notes", "Download with `uv run toxfam download-data`.",
            ],
            check=True,
        )
        print("Done.")
    except FileNotFoundError:
        print("ERROR: `gh` CLI not found. Install from https://cli.github.com", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
