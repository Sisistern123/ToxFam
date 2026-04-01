"""Parse UniProt XML exports into DataFrames matching ToxFam conventions."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

_NS = "{http://uniprot.org/uniprot}"

# Regex to extract the family name from "Belongs to the ... family." etc.
_BELONGS_RE = re.compile(r"Belongs to the (.+?)\.?$")


def _extract_family(similarity_text: str) -> str:
    """Extract the raw family string from the similarity comment.

    Example input:  "Belongs to the conotoxin A superfamily."
    Example output: "conotoxin A superfamily"
    """
    m = _BELONGS_RE.match(similarity_text.strip().rstrip("."))
    if m:
        return m.group(1).strip().rstrip(".")
    return similarity_text.strip()


def parse_uniprot_xml(xml_path: str | Path) -> pd.DataFrame:
    """Parse a UniProt XML file into a DataFrame.

    Returns a DataFrame with columns matching ToxFam conventions:
      - identifier: UniProt accession (e.g. "A0A068B6Q6")
      - Sequence: amino acid sequence
      - Protein families: extracted from <comment type="similarity">
      - Organism (ID): NCBI taxonomy ID
      - function: from <comment type="function">
      - protein_name: from <protein><recommendedName><fullName>
    """
    records: list[dict] = []

    for _event, elem in ET.iterparse(str(xml_path), events=("end",)):
        if elem.tag != f"{_NS}entry":
            continue

        acc_elem = elem.find(f"{_NS}accession")
        accession = acc_elem.text if acc_elem is not None else ""

        # Sequence
        seq_elem = elem.find(f"{_NS}sequence")
        sequence = ""
        if seq_elem is not None and seq_elem.text:
            sequence = seq_elem.text.replace("\n", "").replace(" ", "").strip()

        # Protein name
        protein_name = ""
        prot_elem = elem.find(f"{_NS}protein")
        if prot_elem is not None:
            rec_name = prot_elem.find(f"{_NS}recommendedName")
            if rec_name is not None:
                fn = rec_name.find(f"{_NS}fullName")
                if fn is not None and fn.text:
                    protein_name = fn.text

        # Organism taxonomy ID
        tax_id = ""
        org = elem.find(f"{_NS}organism")
        if org is not None:
            for ref in org.findall(f"{_NS}dbReference"):
                if ref.get("type") == "NCBI Taxonomy":
                    tax_id = ref.get("id", "")
                    break

        # Family from similarity comment
        family = ""
        function_text = ""
        for comment in elem.findall(f"{_NS}comment"):
            ctype = comment.get("type", "")
            text_elem = comment.find(f"{_NS}text")
            if text_elem is None or not text_elem.text:
                continue
            if ctype == "similarity":
                family = _extract_family(text_elem.text)
            elif ctype == "function":
                function_text = text_elem.text

        records.append(
            {
                "identifier": accession,
                "Sequence": sequence,
                "Protein families": family,
                "Organism (ID)": tax_id,
                "function": function_text,
                "protein_name": protein_name,
            }
        )

        elem.clear()

    df = pd.DataFrame(records)
    return df
