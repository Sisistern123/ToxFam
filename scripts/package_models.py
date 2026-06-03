#!/usr/bin/env python3
"""Package trained models into a slimmed models.zip and publish a GitHub Release.

Developer-only script. Builds ``models.zip`` from the trained runs in
``model/model_output/``, keeping only the files needed for inference:

  standard_run/  &  combined_run/
    model_config.json
    class_indices.json
    config.yaml
    models/best_model_calibrated.pt

Plots, predictions, metrics, and the uncalibrated ``best_model.pt`` are
excluded so the asset stays small (~2.5 MB). The Colab prediction notebook
downloads and unzips this asset.

Usage:
    uv run scripts/package_models.py                  # build + upload to models-v1
    uv run scripts/package_models.py --no-upload      # only build models.zip locally
    uv run scripts/package_models.py --tag models-v1

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
MODEL_OUTPUT = ROOT / "model" / "model_output"

REPO = "Sisistern123/ToxFam"
DEFAULT_TAG = "models-v1"

# Runs to bundle and the per-run files inference needs.
RUNS = ("standard_run", "combined_run")
KEEP_FILES = (
    "model_config.json",
    "class_indices.json",
    "config.yaml",
    "models/best_model_calibrated.pt",
)


def build_zip(dest: Path) -> None:
    """Write a slimmed models.zip containing only inference-required files."""
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for run in RUNS:
            run_dir = MODEL_OUTPUT / run
            for rel in KEEP_FILES:
                src = run_dir / rel
                if not src.exists():
                    print(f"ERROR: required file not found: {src}", file=sys.stderr)
                    sys.exit(1)
                zf.write(src, f"{run}/{rel}")
    size_mb = dest.stat().st_size / 1e6
    print(f"  built {dest.name} ({size_mb:.2f} MB)")


def upload(tag: str, zip_path: Path) -> None:
    """Recreate the release at *tag* with *zip_path* as its only asset."""
    try:
        print(f"  deleting old release '{tag}' (if any) ...")
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
                str(zip_path),
                "--repo",
                REPO,
                "--title",
                "Models v1",
                "--notes",
                "Trained ToxFam models (standard + combined) for the Colab "
                "prediction notebook. Slimmed to inference-required files.",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tag", default=DEFAULT_TAG, help="GitHub release tag (default: %(default)s)"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="only build models.zip locally, do not touch the GitHub release",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="where to write models.zip when --no-upload is set "
        "(default: a temp dir, or ./models.zip with --no-upload)",
    )
    args = parser.parse_args()

    if args.no_upload:
        dest = args.output or (ROOT / "models.zip")
        build_zip(dest)
        print(f"models.zip written to {dest}")
        return

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        zip_path = tmp_dir / "models.zip"
        build_zip(zip_path)
        upload(args.tag, zip_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
