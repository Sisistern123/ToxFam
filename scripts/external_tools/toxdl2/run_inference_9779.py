"""Resumable ToxDL 2.0 inference for ToxFam's 9,779 test set.

Recreates the (deleted) validated runner. Numerically follows the repo's own
predict path (`utils.obtain_protein_feature` -> `model.forward`); the only changes
are performance/portability:
  * ESM-2 650M forward on MPS (via dataset.device), node features returned on CPU.
  * GCN + dense head on CPU (model.device patched to cpu) for numeric safety.
  * AlphaFold DB structures fetched v6->v5->v4 (v4 URL in the repo recipe is stale).
  * InterPro domains fetched in batches from the UniProt REST API.

Run from tools/ToxDL2/src with PYTHONPATH=<repo root>:
  PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=$(cd .. && pwd) \
    tools/toxdl2_env/bin/python run_inference_9779.py
"""
import os, sys, time, csv, io
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import requests
import torch

# ---- paths (absolute, relative to the ToxFam project root) ----
SRC = Path(__file__).resolve().parent                 # tools/ToxDL2/src
REPO = SRC.parent                                      # tools/ToxDL2
TOXFAM_ROOT = REPO.parents[1]                          # students/ToxFam
TEST_FASTA = TOXFAM_ROOT / "benchmark/test_set/_shared/test.fasta"
OUT_CSV = TOXFAM_ROOT / "benchmark/test_set/toxdl2/test_scores.csv"
CACHE_OLD = TOXFAM_ROOT / "benchmark/_score_cache/toxdl2_old_test.csv"
NO_STRUCT = TOXFAM_ROOT / "benchmark/test_set/toxdl2/no_structure.txt"
STRUCT_DIR = TOXFAM_ROOT / "benchmark/test_set/toxdl2/structures"
DOMAIN_MODEL = REPO / "checkpoints/protein_domain_embeddings.model"
GCN_PATH = REPO / "checkpoints/ToxDL2_model.pth"
STRUCT_DIR.mkdir(parents=True, exist_ok=True)

# ---- device wiring: ESM on MPS, GCN on CPU ----
import parameters.test_000 as P
P.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
import dataset            # binds dataset.device = P.device, loads ESM-2 onto it
import utils              # obtain_protein_feature, get_domain_vector
import model as M
M.device = torch.device("cpu")   # GCN forward sends prot_domain here
print(f"ESM device={dataset.device}  GCN device={M.device}", flush=True)

gcn = torch.load(str(GCN_PATH), map_location="cpu", weights_only=False)
gcn = gcn.to("cpu").eval()


def read_fasta_ids(path):
    ids = []
    for line in open(path):
        if line.startswith(">"):
            ids.append(line[1:].split()[0].strip())
    return ids


def load_cache(path):
    """accession -> (score, native_pred, has_structure) from a prior run."""
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["identifier"]] = (row.get("score", ""), row.get("native_pred", ""),
                                      row.get("has_structure", ""))
    return out


def fetch_structure(acc):
    """Download AF structure to STRUCT_DIR/{acc}.pdb (v6->v5->v4). Return path or None."""
    dest = STRUCT_DIR / f"{acc}.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    for v in ("v6", "v5", "v4"):
        url = f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_{v}.pdb"
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and r.text.startswith(("HEADER", "ATOM", "MODEL", "REMARK", "TITLE", "CRYST")):
                dest.write_text(r.text)
                return dest
        except requests.RequestException:
            continue
    return None


def fetch_domains_batch(accs):
    """UniProt REST: accession -> [IPR...]. Batches of 100."""
    dom = {a: [] for a in accs}
    for i in range(0, len(accs), 100):
        batch = accs[i:i + 100]
        url = ("https://rest.uniprot.org/uniprotkb/accessions"
               f"?accessions={','.join(batch)}&fields=accession,xref_interpro&format=tsv")
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=120)
                if r.status_code == 200:
                    rdr = csv.reader(io.StringIO(r.text), delimiter="\t")
                    rows = list(rdr)
                    if rows:
                        header = rows[0]
                        ipr_col = next((j for j, c in enumerate(header) if "InterPro" in c or "interpro" in c.lower()), 1)
                        acc_col = 0
                        for row in rows[1:]:
                            if len(row) > ipr_col:
                                a = row[acc_col]
                                iprs = [x for x in row[ipr_col].replace(" ", "").split(";") if x.startswith("IPR")]
                                if a in dom:
                                    dom[a] = iprs
                    break
            except requests.RequestException:
                time.sleep(2)
        print(f"  domains {min(i+100,len(accs))}/{len(accs)}", flush=True)
    return dom


def main():
    all_ids = read_fasta_ids(TEST_FASTA)
    cache = load_cache(CACHE_OLD)
    print(f"test ids={len(all_ids)}  reusable cached={sum(1 for a in all_ids if a in cache)}", flush=True)

    # resume: skip ids already truly scored, and ids with no AlphaFold structure
    done = load_cache(OUT_CSV)
    done_scored = {a for a, rec in done.items() if rec[0] != ""}
    no_struct = {l.strip() for l in open(NO_STRUCT)} if NO_STRUCT.exists() else set()

    todo = [a for a in all_ids if a not in cache and a not in done_scored and a not in no_struct]
    print(f"resume: cached={sum(1 for a in all_ids if a in cache)} "
          f"already_scored={len(done_scored)} no_structure={len(no_struct)} "
          f"-> to compute {len(todo)}", flush=True)

    domains = fetch_domains_batch(todo) if todo else {}

    results = {}  # acc -> (score, native_pred, has_structure)
    t0 = time.time()
    for n, acc in enumerate(todo, 1):
        pdb = fetch_structure(acc)
        if pdb is None:
            results[acc] = ("", "", "0")
        else:
            try:
                feat = utils.obtain_protein_feature(str(pdb), domains.get(acc, []), str(DOMAIN_MODEL))
                feat.batch = torch.zeros(feat.x.shape[0], dtype=torch.long)
                feat.x = feat.x.to("cpu"); feat.edge_index = feat.edge_index.to("cpu")
                with torch.no_grad():
                    score = float(gcn(feat).squeeze().item())
                results[acc] = (f"{score:.6f}", "1" if score >= 0.5 else "0", "1")
            except Exception as e:
                print(f"  [err] {acc}: {e}", flush=True)
                results[acc] = ("", "", "0")
        if n % 50 == 0 or n == len(todo):
            rate = n / (time.time() - t0)
            print(f"  infer {n}/{len(todo)}  {rate:.2f}/s  eta {(len(todo)-n)/max(rate,1e-9)/60:.1f}m", flush=True)
            _write(all_ids, cache, done, results)
    _write(all_ids, cache, done, results)
    scored = sum(1 for a in all_ids if (a in cache or a in done or a in results) and
                 (cache.get(a, done.get(a, results.get(a, ("",))))[0] != ""))
    print(f"DONE_TOXDL2  total={len(all_ids)} scored={scored} -> {OUT_CSV}", flush=True)


def _write(all_ids, cache, done, results):
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identifier", "score", "native_pred", "has_structure"])
        for a in all_ids:
            rec = results.get(a) or done.get(a) or cache.get(a) or ("", "", "0")
            w.writerow([a, rec[0], rec[1], rec[2]])


if __name__ == "__main__":
    main()
