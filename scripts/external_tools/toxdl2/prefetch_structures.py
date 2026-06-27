"""Parallel AlphaFold structure prefetch for the ToxDL2 9,779 run.

Downloads AF structures concurrently (atomic temp+rename) so the inference loop
finds them on disk and never blocks on HTTP. Only fetches accessions we still
need (9,779 test minus cached-reuse minus already-scored).
"""
import os, csv, time, random, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# ToxFam root = nearest ancestor with pyproject.toml (runs in place; no copy).
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
TEST_FASTA = ROOT / "benchmark/test_set/_shared/test.fasta"
CACHE_OLD = ROOT / "benchmark/_score_cache/toxdl2_old_test.csv"
OUT_CSV = ROOT / "benchmark/test_set/toxdl2/test_scores.csv"
STRUCT_DIR = ROOT / "benchmark/test_set/toxdl2/structures"
STRUCT_DIR.mkdir(parents=True, exist_ok=True)
MISSING_LOG = STRUCT_DIR.parent / "no_structure.txt"

def ids_from_fasta(p):
    return [l[1:].split()[0].strip() for l in open(p) if l.startswith(">")]

def cached_ids(p):
    if not p.exists(): return set()
    return {r["identifier"] for r in csv.DictReader(open(p)) if r.get("score")}

test = ids_from_fasta(TEST_FASTA)
reuse = {r["identifier"] for r in csv.DictReader(open(CACHE_OLD))} if CACHE_OLD.exists() else set()
done = cached_ids(OUT_CSV)
todo = [a for a in test if a not in reuse and a not in done]
print(f"test={len(test)} reuse={len(reuse)} done_scored={len(done)} -> prefetch {len(todo)}", flush=True)

def fetch(acc):
    dest = STRUCT_DIR / f"{acc}.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        return acc, "have"
    saw_throttle = False
    for attempt in range(4):
        for v in ("v6", "v5", "v4"):
            try:
                r = requests.get(f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_{v}.pdb", timeout=30)
                if r.status_code == 200 and r.text.startswith(("HEADER","ATOM","MODEL","REMARK","TITLE","CRYST")):
                    # pid alone is NOT unique across ThreadPoolExecutor threads;
                    # add the thread id so concurrent fetches never share a tmp path.
                    tmp = dest.with_suffix(f".pdb.tmp{os.getpid()}.{threading.get_ident()}")
                    tmp.write_text(r.text); os.replace(tmp, dest)
                    return acc, "ok"
                if r.status_code in (429, 500, 502, 503, 504):
                    saw_throttle = True
            except requests.RequestException:
                saw_throttle = True
        if saw_throttle and attempt < 3:
            time.sleep((2 ** attempt) + random.random())  # backoff before next attempt
            saw_throttle = False
        else:
            break
    return acc, "missing"

ok = have = missing = 0
miss_ids = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(fetch, a): a for a in todo}
    for i, f in enumerate(as_completed(futs), 1):
        acc, st = f.result()
        if st == "ok": ok += 1
        elif st == "have": have += 1
        else: missing += 1; miss_ids.append(acc)
        if i % 500 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} ok={ok} have={have} missing={missing}", flush=True)
MISSING_LOG.write_text("\n".join(sorted(miss_ids)))
print(f"DONE_PREFETCH ok={ok} have={have} missing={missing} (missing list -> {MISSING_LOG})", flush=True)
