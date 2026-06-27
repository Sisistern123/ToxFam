#!/usr/bin/env python
"""
Parallel driver for ToxinPred 3.0 (PyPI package `toxinpred3` v1.4).

Why: the upstream CLI computes AAC + DPC features in pure Python (nested loops),
~0.33 s/seq, single-threaded. Each sequence is scored fully independently
(per-row AAC/DPC -> ExtraTrees predict_proba -> per-row class call), so splitting
the FASTA into contiguous chunks and running the *unmodified* upstream CLI on each
chunk in its own working directory yields scores identical to a whole-file run,
while using all cores.

This script does NOT reimplement any scoring. It only:
  1. splits the input FASTA into N contiguous chunks,
  2. runs `python -m toxinpred3.python_scripts.toxinpred3 -i chunk -o out -m <model>`
     for each chunk (each in an isolated cwd so the hardcoded temp files
     seq.aac/seq.dpc/Sequence_1/... do not collide),
  3. merges chunk outputs and emits identifier,score[,native_pred,threshold_used]
     for EVERY input identifier in original order (missing -> empty score).

Usage:
  run_toxinpred3.py --fasta <in.fasta> --out <scores.csv> --workers 14 \
      --model 1 --threshold 0.38 [--raw-dir DIR]
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

PYEXE = sys.executable
MODULE = "toxinpred3.python_scripts.toxinpred3"


def parse_fasta(fp):
    ids, seqs, name, buf = [], [], None, []
    with open(fp) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    seqs.append("".join(buf))
                parts = line[1:].split()
                name = parts[0] if parts else f"_unnamed_{len(ids)}"
                ids.append(name)
                buf = []
            else:
                buf.append(line.strip())
    if name is not None:
        seqs.append("".join(buf))
    return ids, seqs


def run_chunk(args):
    idx, ids, seqs, workdir, model, threshold = args
    os.makedirs(workdir, exist_ok=True)
    chunk_fa = os.path.join(workdir, "chunk.fasta")
    chunk_out = os.path.join(workdir, "chunk_out.csv")
    with open(chunk_fa, "w") as fh:
        for i, s in zip(ids, seqs):
            fh.write(f">{i}\n{s}\n")
    cmd = [PYEXE, "-m", MODULE, "-i", chunk_fa, "-o", chunk_out,
           "-m", str(model), "-t", str(threshold), "-d", "2"]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    ok = proc.returncode == 0 and os.path.exists(chunk_out)
    return idx, chunk_out if ok else None, proc.returncode, proc.stderr[-2000:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--model", type=int, default=1, choices=[1, 2])
    ap.add_argument("--threshold", type=float, default=0.38)
    ap.add_argument("--raw-dir", required=True,
                    help="scratch dir for per-chunk working dirs")
    args = ap.parse_args()

    t0 = time.time()
    ids, seqs = parse_fasta(args.fasta)
    n = len(ids)
    print(f"[{args.fasta}] parsed {n} sequences", flush=True)

    nchunks = min(args.workers, n)
    # Greedy length-balanced partition: per-seq cost ~ proportional to length
    # (DPC scans the sequence 400x). Assign each seq (longest first) to the
    # currently-lightest bin so all workers finish around the same time.
    # Order is irrelevant downstream since results are rebuilt by ID.
    order = sorted(range(n), key=lambda i: len(seqs[i]), reverse=True)
    bins = [[] for _ in range(nchunks)]
    load = [0] * nchunks
    for i in order:
        b = min(range(nchunks), key=lambda k: load[k])
        bins[b].append(i)
        load[b] += len(seqs[i]) + 1
    tasks = []
    for c in range(nchunks):
        wd = os.path.join(args.raw_dir, f"chunk_{c:03d}")
        cids = [ids[i] for i in bins[c]]
        cseqs = [seqs[i] for i in bins[c]]
        tasks.append((c, cids, cseqs, wd, args.model, args.threshold))

    score_col = "Hybrid Score" if args.model == 2 else "ML Score"
    results = {}  # id -> (score, native_pred)
    failed_chunks = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_chunk, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            idx, out_path, rc, err = fut.result()
            if out_path is None:
                failed_chunks.append((idx, rc, err))
                print(f"  chunk {idx} FAILED rc={rc}\n{err}", flush=True)
                continue
            with open(out_path) as fh:
                r = csv.DictReader(fh)
                cols = r.fieldnames
                if not cols:  # empty/header-less output: don't crash, skip the chunk
                    print(f"  chunk {idx} produced empty output; treating as failed",
                          flush=True)
                    failed_chunks.append((idx, rc, "empty output"))
                    continue
                idcol = "ID" if "ID" in cols else "Subject"
                for row in r:
                    rid = row[idcol].lstrip(">")
                    results[rid] = (row.get(score_col, ""), row.get("Prediction", ""))
            print(f"  chunk {idx} done ({len(results)} cumulative)", flush=True)

    scored = sum(1 for i in ids if i in results)
    failed_ids = [i for i in ids if i not in results]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["identifier", "score", "native_pred", "threshold_used"])
        for i in ids:
            if i in results:
                sc, pred = results[i]
                w.writerow([i, sc, pred, args.threshold])
            else:
                w.writerow([i, "", "", args.threshold])

    dt = time.time() - t0
    print(f"[{args.fasta}] scored={scored} failed={len(failed_ids)} "
          f"failed_chunks={len(failed_chunks)} elapsed={dt:.1f}s -> {args.out}",
          flush=True)
    if failed_ids:
        print("FAILED_IDS:", ",".join(failed_ids[:50]),
              ("..." if len(failed_ids) > 50 else ""), flush=True)


if __name__ == "__main__":
    main()
