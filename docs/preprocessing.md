# Preprocessing Pipeline

The preprocessing pipeline (`uv run toxfam preprocess`) transforms raw UniProt data into clustered, split-ready datasets for model training.

```bash
uv run toxfam preprocess [--min-seq-id 0.9]
```

## Data Flow

```
data/raw/0800.tsv  ─┐
                    ├─→ Load & clean labels ─→ SignalP6 ─→ MMseqs2 cluster ─→ Identity-aware split
data/raw/nontox.tsv ┘                                       (per family)       ├─→ train reps
                                                             @ 90% identity    ├─→ val reps
                                                                               ├─→ test reps
                                                                               └─→ train all members (for HBI)
```

## Step 1 — Load raw data

**Input:** Two TSV files in `data/raw/` (downloaded via `toxfam download-data`). These are frozen UniProt exports (date cutoff 2026-03-03); see [README — Raw data](../README.md#raw-data) for the exact queries.

- `0800.tsv` — toxin proteins (UniProt keyword KW-0800)
- `nontox.tsv` — non-toxin proteins

**What happens:**

1. **Toxin data:** Loads the TSV, drops rows without a `Protein families` annotation, renames `Entry` → `identifier`.
2. **Family label cleanup:**
   - Takes only the first family from multi-family annotations (splits on `;` then `,`).
   - Renames ambiguous superfamily names (e.g. `"I1 superfamily"` → `"Conotoxin I1 superfamily"`).
   - Merges related families via regex (e.g. all `Conotoxin*` → `"Conotoxin family"`, all `*phospholipase*` → `"Phospholipase family"`).
   - Collapses families with <10 members into an `"other"` class.
3. **Non-toxin data:** Loads the TSV, renames `Entry` → `identifier`, removes the top 1% longest sequences (outlier filter), labels all as `"nontox"`.

**Output:** Two DataFrames (`tox`, `nontox`) + FASTA files written to `data/intermediate/fasta/`.

## Step 2 — SignalP6 signal peptide removal

**Requires:** SignalP6 set up in `tools/signalp6/` (see [SignalP6 Setup](signalp6_setup.md)) only if there are uncached sequences. If the SP6 cache was downloaded via `toxfam download-data`, SignalP6 is not needed.

**What happens:**

1. Loads a per-sequence cache (`data/intermediate/sp6/sp6_cache.json`) that maps each sequence's MD5 hash to its mature sequence (or `null` if no signal peptide was detected). On first run, the cache is bootstrapped from any existing monolithic SP6 output files in `data/intermediate/sp6/{tox,nontox}/`.
2. Only sequences not in the cache are sent to SignalP6 (run via the isolated `tools/signalp6` uv project). The cache is updated after each batch.
3. For proteins where SignalP6 detects a signal peptide with score > 0.8, the sequence is replaced with the mature (signal-peptide-removed) version.
4. Proteins without a detected signal peptide keep their original sequence.

The SP6 cache can be downloaded alongside other processed data via `uv run toxfam download-data`, which avoids needing to install SignalP6 for most workflows.

**Output:** Updated `tox`/`nontox` DataFrames + FASTA files written to `data/intermediate/fasta/` (`tox_noSP.fasta`, `nontox_noSP.fasta`). The two DataFrames are then concatenated into a single combined DataFrame.

## Step 3 — MMseqs2 per-family clustering

**Purpose:** Reduce redundancy within each family by clustering at a sequence identity threshold (default 90%).

**What happens:**

1. Groups the combined dataset by `Protein families`.
2. For each family separately:
   - Writes a per-family FASTA to `data/intermediate/mmseqs/{family}/input.fasta`.
   - Runs `mmseqs easy-cluster` at the configured `--min-seq-id` (default 0.9).
   - Cluster results are stored alongside the input in `data/intermediate/mmseqs/{family}/`.
3. Collects all representative sequences from every family's `cluster_rep_seq.fasta`.
4. Re-applies the <10 member threshold (families that dropped below 10 reps after clustering get collapsed to `"other"` again).

**Output:** Two DataFrames:

- `rep_df_all` — all representative sequences (toxin + non-toxin)
- `rep_df_tox` — toxin reps only

Both are saved as CSV + FASTA to `data/intermediate/mmseqs/representatives/`.

## Step 4 — Identity-aware train/val/test splits

**Problem:** Random stratified splitting allows proteins with >30% sequence identity to appear in both train and test, inflating metrics through data leakage.

**Solution:** The `identity_aware_splits()` function uses an adaptive cluster-then-split approach:

### 4a. Global clustering at 30% identity

All representative sequences are written to a single FASTA and clustered with `mmseqs easy-cluster --min-seq-id 0.3`. This produces meta-clusters of proteins sharing >30% identity.

### 4b. Cluster-level stratified splitting

Entire meta-clusters are assigned to train/val/test (70/15/15) using `MultilabelStratifiedShuffleSplit` at the cluster level. Each cluster's label is the union of its members' family labels. This ensures no sequence in val/test has >30% identity to any training sequence.

### 4c. Adaptive relaxation for under-represented families

After initial assignment, families stuck entirely in one split (all members in one tight 30% cluster) are handled:

1. Re-cluster just that family's members at 40%, then 50%, 60%, 70%
2. Stop at the first threshold that produces ≥2 sub-clusters
3. Assign sub-clusters to different splits
4. Families that remain a single cluster even at 70% go entirely to train

### 4d. Output

A summary is printed showing how many families required each threshold:
```
Split threshold summary:
  30%: 34 families
  40%: 1 families
  50%: 2 families
  70%: 1 families
```

**Output files:**

| File                                    | Description                                             |
| --------------------------------------- | ------------------------------------------------------- |
| `data/processed/training_data.csv`      | Combined CSV with `Split` column (`train`/`val`/`test`) |
| `benchmark/test_data.csv` + `.fasta`    | Test split                                              |
| `benchmark/val_data.csv` + `.fasta`     | Validation split                                        |
| `benchmark/HBI/train_all_df.csv`        | All members of train clusters                           |
| `benchmark/HBI/train_all_members.fasta` | FASTA for HBI baseline                                  |
