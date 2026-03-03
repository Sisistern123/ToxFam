# Preprocessing Pipeline

The preprocessing pipeline (`uv run toxfam preprocess`) transforms raw UniProt data into clustered, split-ready datasets for model training.

```bash
uv run toxfam preprocess [--no-signalp6] [--min-seq-id 0.9]
```

## Data Flow

```
data/raw/0800.tsv  ─┐
                     ├─→ Load & clean labels ─→ SignalP6 ─→ MMseqs2 cluster ─→ Stratified split
data/raw/nontox.tsv ┘                                       (per family)       ├─→ train reps
                                                             @ 90% identity    ├─→ val reps
                                                                               ├─→ test reps
                                                                               └─→ train all members (for HBI)
```

## Step 1 — Load raw data

**Input:** Two TSV files from UniProt in `data/raw/`:

- `0800.tsv` — toxin proteins (UniProt taxon 0800)
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

**Controlled by:** `--run-signalp6` (default) / `--no-signalp6`

**Requires:** SignalP6 set up in `tools/signalp6/` (see [SignalP6 Setup](signalp6_setup.md)). If not installed, the pipeline skips this step with a warning.

**What happens:**

1. If SignalP6 output already exists in `data/intermediate/sp6/{tox,nontox}/`, the cached results are used.
2. Otherwise, runs SignalP6 via `scripts/run_signalp6.sh` using the isolated `tools/signalp6` uv project.
3. For proteins where SignalP6 detects a signal peptide with score > 0.8, the sequence is replaced with the mature (signal-peptide-removed) version.
4. Proteins without a detected signal peptide keep their original sequence.

**Output:** Updated `tox`/`nontox` DataFrames + FASTA files written to `data/intermediate/fasta/` (`tox_noSP.fasta`, `nontox_noSP.fasta`). The two DataFrames are then concatenated into a single combined DataFrame.

## Step 3 — MMseqs2 per-family clustering

**Purpose:** Reduce redundancy within each family by clustering at a sequence identity threshold (default 90%).

**What happens:**

1. Groups the combined dataset by `Protein families`.
2. For each family separately:
   - Writes a per-family FASTA to `data/intermediate/families/{family}.fasta`.
   - Runs `mmseqs easy-cluster` at the configured `--min-seq-id` (default 0.9).
   - Results go to `data/intermediate/mmseqs/{family}/`.
3. Collects all representative sequences from every family's `cluster_rep_seq.fasta`.
4. Re-applies the <10 member threshold (families that dropped below 10 reps after clustering get collapsed to `"other"` again).

**Output:** Two DataFrames:

- `rep_df_all` — all representative sequences (toxin + non-toxin)
- `rep_df_tox` — toxin reps only

Both are saved as CSV + FASTA to `data/intermediate/representatives/`.

## Step 4 — Multilabel-stratified train/val/test splits

**What happens:**

1. Binarizes family labels using `MultiLabelBinarizer`.
2. First split: 70% train / 30% val+test using `MultilabelStratifiedShuffleSplit` (seed=42).
3. Second split: from the 30%, splits 50/50 into 15% val / 15% test.
4. Builds a **train-all-members** set: expands train representative sequences back to all cluster members using MMseqs2 cluster membership files. This is used for the HBI (homology-based inference) baseline benchmark.

**Output files:**

| File | Description |
|---|---|
| `data/processed/training_data.csv` | Combined CSV with `Split` column (`train`/`val`/`test`) |
| `benchmark/test_data.csv` + `.fasta` | Test split |
| `benchmark/val_data.csv` + `.fasta` | Validation split |
| `benchmark/HBI/train_all_df.csv` | All members of train clusters |
| `benchmark/HBI/train_all_members.fasta` | FASTA for HBI baseline |
