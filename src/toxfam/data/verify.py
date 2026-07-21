"""Whole-pipeline consistency check: `toxfam verify`.

Aggregates the two guard layers into one green/red report:

1. Provenance stamps — does each split-derived artifact's
   ``<artifact>.provenance.json`` (or a benchmark run's ``run_metadata.json``)
   record the manifest hash on disk?
2. Content invariants — do the artifacts actually satisfy the split (no HBI
   reference leakage, embeddings cover the manifest, taxonomy matches)?

``run_checks`` returns structured rows so callers (the CLI table, and the figure
pipeline's refuse-on-red gate) share one source of truth. ``verify_or_raise`` is
the programmatic gate used by ``numbers_manifest``/figures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from toxfam._paths import benchmark_dir, processed_dir
from toxfam.data.invariants import InvariantResult, all_invariants
from toxfam.data.split_manifest import (
    SplitManifestError,
    manifest_exists,
    manifest_sha256,
    read_provenance,
)

# File artifacts whose *value* depends on the split and so must carry a manifest
# stamp. Embeddings and taxonomy vectors are split-independent (a protein's vector
# is the same in any split); their guard is a coverage invariant, not a stamp, so
# a benign manifest change does not falsely fail them.
STAMPED_ARTIFACTS = ("hbi_train_all.csv",)


@dataclass
class CheckRow:
    """One line of the verify report."""

    name: str
    status: str  # "ok" | "fail" | "skip"
    detail: str


class PipelineNotVerified(SplitManifestError):
    """`toxfam verify` found a red check; refuse to proceed."""


def _stamp_row(name: str, path: Path, current: str) -> CheckRow:
    """Provenance-sidecar check for one file artifact."""
    if not path.exists():
        return CheckRow(name, "skip", f"{path.name} absent")
    prov = read_provenance(path)
    if prov is None:
        return CheckRow(name, "fail", f"{path.name} has no provenance stamp")
    stamped = prov.get("split_manifest_sha256", "")
    if stamped != current:
        return CheckRow(
            name, "fail", f"stamped {stamped[:8] or '??'} ≠ manifest {current[:8]}"
        )
    return CheckRow(name, "ok", f"pinned to {current[:8]}")


def _benchmark_rows(current: str, dataset: str | None) -> list[CheckRow]:
    """Check each benchmark run's recorded split against the manifest."""
    root = benchmark_dir()
    if not root.exists():
        return []
    datasets = (
        [root / dataset] if dataset else sorted(p for p in root.iterdir() if p.is_dir())
    )
    rows: list[CheckRow] = []
    for ds_dir in datasets:
        if not ds_dir.is_dir():
            continue
        for meta_path in sorted(ds_dir.glob("*/run_metadata.json")):
            name = f"benchmark/{ds_dir.name}/{meta_path.parent.name}"
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                rows.append(CheckRow(name, "fail", f"unreadable: {e}"))
                continue
            stamped = meta.get("split_manifest_sha256")
            if stamped is None:
                rows.append(CheckRow(name, "fail", "no split stamp (pre-guard run)"))
            elif stamped != current:
                rows.append(
                    CheckRow(name, "fail", f"stale: {stamped[:8]} ≠ {current[:8]}")
                )
            else:
                rows.append(CheckRow(name, "ok", f"pinned to {current[:8]}"))
    return rows


def _invariant_row(r: InvariantResult) -> CheckRow:
    return CheckRow(
        r.name, "skip" if r.skipped else ("ok" if r.ok else "fail"), r.detail
    )


def run_checks(dataset: str | None = None) -> list[CheckRow]:
    """Run every provenance + invariant check. Returns report rows.

    ``dataset`` limits the benchmark scan to one dataset (e.g. ``test_set``);
    None scans all.
    """
    if not manifest_exists():
        return [CheckRow("split_manifest", "fail", "no split_manifest.csv found")]

    current = manifest_sha256()
    rows: list[CheckRow] = [CheckRow("split_manifest", "ok", f"sha {current[:8]}")]

    proc = processed_dir()
    for fname in STAMPED_ARTIFACTS:
        rows.append(_stamp_row(f"stamp:{fname}", proc / fname, current))

    rows.extend(_invariant_row(r) for r in all_invariants())
    rows.extend(_benchmark_rows(current, dataset))
    return rows


def has_failures(rows: list[CheckRow]) -> bool:
    return any(r.status == "fail" for r in rows)


def verify_or_raise(dataset: str | None = None) -> None:
    """Raise ``PipelineNotVerified`` if any check is red. Used by the figure gate."""
    rows = run_checks(dataset)
    failed = [r for r in rows if r.status == "fail"]
    if failed:
        lines = "\n".join(f"  ✗ {r.name}: {r.detail}" for r in failed)
        raise PipelineNotVerified(
            "Pipeline is not consistent with the split manifest:\n"
            + lines
            + "\n\nRun 'toxfam verify' for the full report, regenerate the flagged "
            "artifacts, or pass --force to override (not recommended)."
        )
