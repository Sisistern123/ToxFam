"""Tests for toxfam.data.xml_parser."""

from __future__ import annotations

import tempfile
from pathlib import Path

from toxfam.data.xml_parser import _extract_family, parse_uniprot_xml

SAMPLE_XML = """\
<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<uniprot xmlns="http://uniprot.org/uniprot">
<entry dataset="Swiss-Prot" created="2020-01-01" modified="2023-01-01" version="1" xmlns="http://uniprot.org/uniprot">
  <accession>P12345</accession>
  <name>TOX1_SNAKE</name>
  <protein>
    <recommendedName>
      <fullName>Alpha-neurotoxin 1</fullName>
    </recommendedName>
  </protein>
  <organism>
    <name type="scientific">Naja naja</name>
    <dbReference type="NCBI Taxonomy" id="8637"/>
  </organism>
  <comment type="function">
    <text>Binds to nicotinic acetylcholine receptors.</text>
  </comment>
  <comment type="similarity">
    <text>Belongs to the three-finger toxin family.</text>
  </comment>
  <sequence length="71" mass="7800" checksum="ABC123" modified="2020-01-01" version="1">
MKTLLLTLVVVTIVCLDLGYTLTCYNGETNCY
KQWSDHRGTIIERGCGCPTVKPGIKLSCCED
SSFCNK
  </sequence>
</entry>
<entry dataset="Swiss-Prot" created="2020-01-01" modified="2023-01-01" version="1" xmlns="http://uniprot.org/uniprot">
  <accession>Q99999</accession>
  <name>CONO_SNAIL</name>
  <protein>
    <recommendedName>
      <fullName>Conotoxin alpha-1</fullName>
    </recommendedName>
  </protein>
  <organism>
    <name type="scientific">Conus geographus</name>
    <dbReference type="NCBI Taxonomy" id="6491"/>
  </organism>
  <sequence length="20" mass="2200" checksum="DEF456" modified="2020-01-01" version="1">
ECCNPACGRHYSCGK
  </sequence>
</entry>
</uniprot>
"""


def test_extract_family():
    assert _extract_family("Belongs to the three-finger toxin family.") == "three-finger toxin family"
    assert _extract_family("Belongs to the conotoxin A superfamily.") == "conotoxin A superfamily"


def test_parse_uniprot_xml():
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(SAMPLE_XML)
        tmp_path = Path(f.name)

    df = parse_uniprot_xml(tmp_path)
    tmp_path.unlink()

    assert len(df) == 2

    row1 = df[df["identifier"] == "P12345"].iloc[0]
    assert row1["Protein families"] == "three-finger toxin family"
    assert row1["Organism (ID)"] == "8637"
    assert row1["protein_name"] == "Alpha-neurotoxin 1"
    assert "Binds to nicotinic" in row1["function"]
    assert len(row1["Sequence"]) > 0

    row2 = df[df["identifier"] == "Q99999"].iloc[0]
    assert row2["Protein families"] == ""  # no similarity comment
    assert row2["Organism (ID)"] == "6491"


def test_parse_xml_columns():
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(SAMPLE_XML)
        tmp_path = Path(f.name)

    df = parse_uniprot_xml(tmp_path)
    tmp_path.unlink()

    expected_cols = {"identifier", "Sequence", "Protein families", "Organism (ID)", "function", "protein_name"}
    assert set(df.columns) == expected_cols
