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
import sys
import tempfile
import zipfile
from pathlib import Path

# Sibling import: works both when run as `uv run scripts/package_models.py` and when
# the file is loaded by path (importlib), which does not put scripts/ on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gh_release import (  # noqa: E402
    add_release_args,
    create_release,
    guard_existing_tag,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_OUTPUT = ROOT / "model" / "model_output"

DEFAULT_TAG = "models-v1"

# Runs to bundle and the per-run files inference needs.
RUNS = ("standard_run", "combined_run")
KEEP_FILES = (
    "model_config.json",
    "class_indices.json",
    "config.yaml",
    "models/best_model_calibrated.pt",
    # Records the split manifest the checkpoint was calibrated against. `eval` and
    # `predict test_set/val_set` refuse a checkpoint without it.
    "models/split_provenance.json",
)


def build_zip(dest: Path, runs: tuple[str, ...] = RUNS) -> None:
    """Write a slimmed models.zip containing only inference-required files."""
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for run in runs:
            run_dir = MODEL_OUTPUT / run
            for rel in KEEP_FILES:
                src = run_dir / rel
                if not src.exists():
                    print(f"ERROR: required file not found: {src}", file=sys.stderr)
                    sys.exit(1)
                zf.write(src, f"{run}/{rel}")
    size_mb = dest.stat().st_size / 1e6
    print(f"  built {dest.name} ({size_mb:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tag", default=DEFAULT_TAG, help="GitHub release tag (default: %(default)s)"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(RUNS),
        help="which trained runs to bundle (default: %(default)s)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="only build models.zip locally, do not touch the GitHub release",
    )
    add_release_args(
        parser,
        notes_help="markdown file with the release notes "
        "(metrics, split manifest hash)",
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

    runs = tuple(args.runs)

    if args.no_upload:
        dest = args.output or (ROOT / "models.zip")
        build_zip(dest, runs)
        print(f"models.zip written to {dest}")
        return

    notes = (
        args.notes_file.read_text()
        if args.notes_file
        else (
            "Trained ToxFam models (standard + combined) for the Colab "
            "prediction notebook. Slimmed to inference-required files."
        )
    )

    tag: str = args.tag
    next_tag = (
        f"{tag[:-1]}{int(tag[-1]) + 1}" if tag[-1:].isdigit() else "a new version"
    )
    exists = guard_existing_tag(
        tag,
        replace=args.replace,
        remediation=f"Publish new artifacts under a new tag (e.g. {next_tag})",
    )

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "models.zip"
        build_zip(zip_path, runs)
        create_release(
            tag,
            [zip_path],
            title=f"Models {tag.removeprefix('models-')}",
            notes=notes,
            exists=exists,
            replace=args.replace,
            prerelease=args.prerelease,
            target=args.target,
        )


if __name__ == "__main__":
    main()
