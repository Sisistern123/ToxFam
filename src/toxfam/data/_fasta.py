"""Minimal FASTA I/O — replaces Bio.SeqIO for our simple use case."""

from __future__ import annotations

import hashlib
import os
from collections import namedtuple
from pathlib import Path
from typing import Iterator

import pandas as pd

FastaRecord = namedtuple("FastaRecord", ["id", "seq"])


def parse_fasta(path: str | os.PathLike) -> Iterator[FastaRecord]:
    """Parse a FASTA file, yielding records with `id` and `seq` attributes."""
    with open(path) as f:
        header = None
        chunks: list[str] = []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield FastaRecord(header, "".join(chunks))
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield FastaRecord(header, "".join(chunks))


def read_fasta_as_dict(path: str | os.PathLike) -> dict[str, str]:
    """Read a FASTA file into a dict mapping identifier -> sequence.

    Identifiers are cleaned: '/' and '.' replaced with '_'.
    Non-standard residues (U, Z, O) are *not* replaced here — call sites
    should handle that if needed.
    """
    sequences: dict[str, str] = {}
    current_id = None
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                current_id = current_id.replace("/", "_").replace(".", "_")
                sequences[current_id] = ""
            elif current_id is not None:
                sequences[current_id] += "".join(line.split()).upper().replace("-", "")
    return sequences


def write_fasta(
    df: pd.DataFrame,
    filename: str | os.PathLike,
    *,
    id_col: str = "identifier",
    seq_col: str = "Sequence",
) -> None:
    """Write a FASTA file from a DataFrame. Skips writing if content is unchanged (MD5)."""
    new_content = "".join(
        f">{row[id_col]}\n{row[seq_col]}\n" for _, row in df.iterrows()
    )
    path = Path(filename)
    if path.exists():
        new_hash = hashlib.md5(new_content.encode()).hexdigest()
        old_hash = hashlib.md5(path.read_bytes()).hexdigest()
        if new_hash == old_hash:
            return
    with open(path, "w") as f:
        f.write(new_content)
