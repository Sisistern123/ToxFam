"""Build the ToxDL 2.0 contamination set + contamination-excluded clean subset.

ToxDL 2.0 is ToxProt-trained, so it overlaps our UniProt KW-0800 test positives.
This script measures that overlap (the headline "65.4% of test toxins seen"
figure) and writes the clean-subset ground truth used by `compare.py
--labels-dir _shared_clean`.

Inputs (ToxDL 2.0 must be cloned at tools/ToxDL2 — github.com/shzhulin/ToxDL2 @ a265475):
  tools/ToxDL2/data/protein_sequences/train.fasta   ToxDL2 training sequences (headers = accessions)
  tools/ToxDL2/data/domain_data/valid.domain        ToxDL2 validation accessions
  benchmark/test_set/_shared/test_labels.csv         our 9,779 test ground truth (build_harness.py --shared-only)

Outputs:
  benchmark/test_set/_shared/toxdl2_seen_in_train.txt   the contaminated (seen) test ids
  benchmark/test_set/_shared_clean/test_labels.csv      test labels MINUS seen ids
  benchmark/test_set/_shared_clean/val_labels.csv       full val labels (unchanged; for Youden thresholding)

ToxDL 2.0's *test* set is deliberately NOT counted as contamination — only what the
model was trained on (train + valid).
"""
import csv, os, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]            # students/ToxFam
TOXDL2 = ROOT / "tools/ToxDL2/data"
SHARED = ROOT / "benchmark/test_set/_shared"
CLEAN = ROOT / "benchmark/test_set/_shared_clean"
CLEAN.mkdir(parents=True, exist_ok=True)


def fasta_accs(path):
    return {line[1:].split()[0].split("\t")[0].strip()
            for line in open(path) if line.startswith(">")}


def domain_accs(path):
    # valid.domain is FASTA-like: each protein is a ">ACC" header line followed
    # by interleaved data lines (sequence, domain count, 256-d embedding vector).
    # ONLY the header lines are accessions — parsing the data lines as accessions
    # silently over-collects. Verified against ToxDL2 @ a265475 (931 valid accs).
    # The non-empty assert in main() catches a future format change loudly.
    accs = set()
    for line in open(path):
        if line.startswith(">"):
            accs.add(line[1:].split()[0])
    return accs


def main():
    train_accs = fasta_accs(TOXDL2 / "protein_sequences/train.fasta")
    valid_accs = domain_accs(TOXDL2 / "domain_data/valid.domain")
    assert train_accs, f"No accessions parsed from {TOXDL2/'protein_sequences/train.fasta'}"
    assert valid_accs, f"No accessions parsed from {TOXDL2/'domain_data/valid.domain'}"
    seen = train_accs | valid_accs
    print(f"ToxDL2 train+valid accessions: {len(seen)} "
          f"(train={len(train_accs)}, valid={len(valid_accs)})")

    test_all, test_tox = set(), set()
    rows = list(csv.DictReader(open(SHARED / "test_labels.csv")))
    assert rows, f"{SHARED/'test_labels.csv'} has no data rows (run build_harness.py --shared-only)"
    for r in rows:
        test_all.add(r["identifier"])
        if r["is_toxic"] == "1":
            test_tox.add(r["identifier"])

    contaminated = test_all & seen
    contaminated_tox = test_tox & seen
    print(f"test: {len(test_all)} proteins, {len(test_tox)} toxins")
    print(f"CONTAMINATION (test ∩ ToxDL2 train+valid):")
    print(f"  proteins: {len(contaminated)} ({100*len(contaminated)/len(test_all):.1f}%)")
    print(f"  toxins:   {len(contaminated_tox)} ({100*len(contaminated_tox)/len(test_tox):.1f}% of {len(test_tox)})")
    print(f"  clean subset: {len(test_all)-len(contaminated)} proteins, "
          f"{len(test_tox)-len(contaminated_tox)} toxins")

    (SHARED / "toxdl2_seen_in_train.txt").write_text("\n".join(sorted(contaminated)))

    with open(CLEAN / "test_labels.csv", "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            if r["identifier"] not in contaminated:
                w.writerow(r)
    shutil.copy(SHARED / "val_labels.csv", CLEAN / "val_labels.csv")   # full val, for Youden thresholding
    print(f"wrote {SHARED/'toxdl2_seen_in_train.txt'} and {CLEAN}/")


if __name__ == "__main__":
    main()
