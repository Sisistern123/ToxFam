"""Minimal FASTA parser — replaces Bio.SeqIO for our simple use case."""

from __future__ import annotations

import os
from collections import namedtuple
from typing import Iterator

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
