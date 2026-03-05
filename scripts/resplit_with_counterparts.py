"""Re-split training data with counterparts integrated into the nontox pool.

Takes the existing training_data.csv (already SP-trimmed + 90% clustered),
adds 287 new counterpart sequences, and re-runs only the 30% identity-aware
splitting so counterparts are distributed across train/val/test.
"""

import pandas as pd
from pathlib import Path

from toxfam._paths import get_project_root, processed_dir, raw_dir, intermediate_dir
from toxfam.data._fasta import write_fasta
from toxfam.data.preprocessing import identity_aware_splits, build_train_all_members

proj = get_project_root()
proc = processed_dir()
bench_dir = proj / "benchmark"
bench_hbi_dir = bench_dir / "HBI"

# 1. Load existing training data (already preprocessed representatives)
print("Loading existing training_data.csv ...")
base = pd.read_csv(proc / "training_data.csv")
print(f"  Base: {len(base)} sequences, {base['Protein families'].nunique()} families")

# 2. Load counterparts
cp_csv = raw_dir() / "nontox_counterparts" / "counterparts.csv"
cp_df = pd.read_csv(cp_csv)
cp_df = cp_df[["identifier", "Sequence", "Protein families", "organism_id"]].copy()
cp_df.rename(columns={"organism_id": "Organism (ID)"}, inplace=True)

# 3. Deduplicate: keep only counterparts not already in base
existing_ids = set(base["identifier"])
new_cp = cp_df[~cp_df["identifier"].isin(existing_ids)].copy()
print(f"  Counterparts: {len(cp_df)} total, {len(cp_df) - len(new_cp)} already in base, {len(new_cp)} new")

# 4. Combine (drop the old Split column — we're re-splitting)
base_no_split = base.drop(columns=["Split"])
combined = pd.concat([base_no_split, new_cp], ignore_index=True)
print(f"  Combined: {len(combined)} sequences")

# 5. Re-run identity-aware splitting
print("\nRunning identity-aware splitting (30% clustering) ...")
train_df, val_df, test_df = identity_aware_splits(combined, base_seq_id=0.3)
train_df["Split"] = "train"
val_df["Split"] = "val"
test_df["Split"] = "test"

training_data = pd.concat([train_df, val_df, test_df], ignore_index=True)

# 6. Save
training_data.to_csv(proc / "training_data.csv", index=False)
print(f"\nSaved training_data.csv: {len(training_data)} sequences")
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# 7. Check counterpart distribution
cp_ids = set(new_cp["identifier"])
in_train = training_data[(training_data["identifier"].isin(cp_ids)) & (training_data["Split"] == "train")]
in_val = training_data[(training_data["identifier"].isin(cp_ids)) & (training_data["Split"] == "val")]
in_test = training_data[(training_data["identifier"].isin(cp_ids)) & (training_data["Split"] == "test")]
print(f"\nNew counterpart distribution:")
print(f"  Train: {len(in_train)}, Val: {len(in_val)}, Test: {len(in_test)}")

# 8. Rebuild benchmark files
train_all_csv = bench_hbi_dir / "train_all_df.csv"
# For build_train_all_members we need the full original data with cluster members
# Since we only have representatives, train_all = train_df for this run
bench_hbi_dir.mkdir(parents=True, exist_ok=True)
train_df.to_csv(bench_hbi_dir / "train_all_df.csv", index=False)
write_fasta(train_df, bench_hbi_dir / "train_all_members.fasta")
bench_dir.mkdir(parents=True, exist_ok=True)
test_df.to_csv(bench_dir / "test_data.csv", index=False)
write_fasta(test_df, bench_dir / "test_data.fasta")
val_df.to_csv(bench_dir / "val_data.csv", index=False)
write_fasta(val_df, bench_dir / "val_data.fasta")

print("\nDone.")
