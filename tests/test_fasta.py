"""Tests for toxfam.data._fasta."""

from __future__ import annotations

import pandas as pd

from toxfam.data._fasta import parse_fasta, read_fasta_as_dict, write_fasta


def test_parse_fasta_yields_records(sample_fasta):
    records = list(parse_fasta(sample_fasta))
    assert len(records) == 3
    assert records[0].id == "P001"
    assert records[0].seq == "MKTAYIAKQR"
    assert records[2].id == "P003"


def test_read_fasta_as_dict(sample_fasta):
    d = read_fasta_as_dict(sample_fasta)
    assert isinstance(d, dict)
    assert len(d) == 3
    assert d["P001"] == "MKTAYIAKQR"


def test_read_fasta_as_dict_cleans_identifiers(tmp_path):
    fasta = tmp_path / "special.fasta"
    fasta.write_text(">sp|Q12345|PROT_HUMAN\nACDE\n")
    d = read_fasta_as_dict(fasta)
    # '/' and '.' in id get replaced with '_'
    key = list(d.keys())[0]
    assert "/" not in key
    assert "." not in key


def test_write_fasta_creates_file(tmp_path):
    df = pd.DataFrame({"identifier": ["X1", "X2"], "Sequence": ["ACDE", "FGHI"]})
    out = tmp_path / "out.fasta"
    write_fasta(df, out)
    assert out.exists()
    text = out.read_text()
    assert ">X1\nACDE\n" in text
    assert ">X2\nFGHI\n" in text


def test_write_fasta_custom_columns(tmp_path):
    df = pd.DataFrame({"Entry": ["E1"], "Seq": ["MNOP"]})
    out = tmp_path / "custom.fasta"
    write_fasta(df, out, id_col="Entry", seq_col="Seq")
    text = out.read_text()
    assert ">E1\nMNOP\n" in text


def test_write_fasta_md5_skip(tmp_path):
    df = pd.DataFrame({"identifier": ["A"], "Sequence": ["XYZ"]})
    out = tmp_path / "skip.fasta"
    write_fasta(df, out)
    mtime1 = out.stat().st_mtime_ns

    # Write again — content unchanged, file should NOT be rewritten
    write_fasta(df, out)
    mtime2 = out.stat().st_mtime_ns
    assert mtime1 == mtime2


def test_write_fasta_rewrites_on_change(tmp_path):
    out = tmp_path / "change.fasta"
    df1 = pd.DataFrame({"identifier": ["A"], "Sequence": ["XYZ"]})
    write_fasta(df1, out)
    old_text = out.read_text()

    df2 = pd.DataFrame({"identifier": ["A"], "Sequence": ["ABC"]})
    write_fasta(df2, out)
    new_text = out.read_text()
    assert old_text != new_text
    assert "ABC" in new_text
