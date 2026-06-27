"""Smoke test: reproduce one cached ToxDL2 score with the rebuilt runner."""
import os, csv
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import torch

SRC = Path(__file__).resolve().parent
REPO = SRC.parent
TOXFAM_ROOT = REPO.parents[1]
CACHE_OLD = TOXFAM_ROOT / "benchmark/_score_cache/toxdl2_old_test.csv"
TEST_FASTA = TOXFAM_ROOT / "benchmark/test_set/_shared/test.fasta"

import parameters.test_000 as P
P.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
import dataset, utils
import model as M
M.device = torch.device("cpu")
gcn = torch.load(str(REPO / "checkpoints/ToxDL2_model.pth"), map_location="cpu", weights_only=False).to("cpu").eval()
print("ESM", dataset.device, "GCN", M.device)

# pick a cached accession that is ALSO in our 9,779 test set, has_structure=1
test_ids = {l[1:].split()[0].strip() for l in open(TEST_FASTA) if l.startswith(">")}
cache = {}
with open(CACHE_OLD) as f:
    for row in csv.DictReader(f):
        if row.get("has_structure") == "1" and row.get("score") and row["identifier"] in test_ids:
            cache[row["identifier"]] = float(row["score"])
if not cache:
    raise SystemExit("No cached accession is also in the 9,779 test set with "
                     "has_structure=1 and a numeric score — nothing to validate against.")
acc = next(iter(cache))
ref = cache[acc]
print(f"validating {acc}  cached_score={ref}")

import requests
dest = TOXFAM_ROOT / "benchmark/test_set/toxdl2/structures" / f"{acc}.pdb"
dest.parent.mkdir(parents=True, exist_ok=True)
for v in ("v6", "v5", "v4"):
    r = requests.get(f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_{v}.pdb", timeout=60)
    if r.status_code == 200 and r.text.startswith(("HEADER", "ATOM", "MODEL", "REMARK", "TITLE", "CRYST")):
        dest.write_text(r.text); print("structure", v); break

url = ("https://rest.uniprot.org/uniprotkb/accessions"
       f"?accessions={acc}&fields=accession,xref_interpro&format=tsv")
r = requests.get(url, timeout=60)
rows = list(csv.reader(r.text.splitlines(), delimiter="\t"))
iprs = ([x for x in rows[1][1].replace(" ", "").split(";") if x.startswith("IPR")]
        if len(rows) > 1 and len(rows[1]) > 1 else [])
print("domains", iprs)

feat = utils.obtain_protein_feature(str(dest), iprs, str(REPO / "checkpoints/protein_domain_embeddings.model"))
feat.batch = torch.zeros(feat.x.shape[0], dtype=torch.long)
with torch.no_grad():
    score = float(gcn(feat).squeeze().item())
print(f"RECOMPUTED {acc}: {score:.6f}  | cached {ref:.6f}  | absdiff {abs(score-ref):.2e}")
