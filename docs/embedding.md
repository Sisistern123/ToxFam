# Embedding Generation

The embedding step (`uv run toxfam embed`) converts protein sequences from a FASTA file into fixed-length numerical vectors using the ProtT5 protein language model. Each protein produces a 1024-dimensional embedding stored in an HDF5 file.

```bash
uv run toxfam embed [-i <input.fasta>] [-o <output.h5>] [--force]
```

## How it works

1. **Read FASTA** — Loads all sequences and sorts them longest-first.
2. **Skip existing** — Opens the output H5 (if it exists) and filters out sequences that are already embedded. If all sequences are present, exits immediately without loading the model.
3. **Load model** — Downloads/loads the ProtT5-XL-U50 encoder (~2.4 GB) from HuggingFace.
4. **Batch & embed** — Groups sequences into batches respecting both a max-sequences limit and a max-residues limit (to control memory usage). For each batch:
   - Tokenizes sequences (adding spaces between residues, replacing non-standard amino acids U/Z/O with X)
   - Runs the ProtT5 encoder
   - Mean-pools over residue positions to get one 1024-dim vector per protein
   - Writes each embedding to the H5 file and flushes to disk

## CLI options

| Option | Default | Description |
|---|---|---|
| `-i`, `--input` | `data/intermediate/representatives/all.fasta` | Input FASTA file |
| `-o`, `--output` | `data/processed/embeddings.h5` | Output HDF5 file |
| `--force` | off | Overwrite the H5 file instead of resuming |
| `--model-dir` | HuggingFace cache | Custom cache directory for the model |
| `--model-name` | `Rostlab/prot_t5_xl_half_uniref50-enc` | HuggingFace model identifier |
| `--max-residues` | 4000 | Max total residues per batch |
| `--max-batch` | 100 | Max sequences per batch |

## Resume support

By default, the H5 file is opened in append mode. If a previous run was interrupted (crash, Ctrl+C, timeout), re-running the same command skips all already-embedded sequences and continues from where it left off. The model is only loaded if there is work to do.

Use `--force` to discard the existing file and re-embed everything from scratch.

## Performance

- **CUDA** — The model runs in float16 with `torch.amp.autocast` for faster inference and lower VRAM usage. Embeddings are always stored as float32 for downstream compatibility.
- **MPS (Apple Silicon)** — Supported, runs in float32.
- **CPU** — Supported, runs in float32. Significantly slower.
- **Batching** — Sequences are sorted longest-first so that the most memory-intensive batch runs first. If it fits, all subsequent batches will too. The `--max-residues` and `--max-batch` limits control batch sizes; smaller values reduce peak memory at the cost of throughput.
- **Flush** — Results are flushed to disk after every batch, so a crash only loses the in-progress batch.

## Output format

The HDF5 file contains one dataset per protein, keyed by identifier:

```python
import h5py

with h5py.File("data/processed/embeddings.h5", "r") as f:
    emb = f["P01234"][:]  # shape: (1024,), dtype: float32
```

## Where it fits in the pipeline

```
preprocessing ─→ data/intermediate/representatives/all.fasta
                              │
                    uv run toxfam embed
                              │
                       data/processed/embeddings.h5
                              │
                    uv run toxfam train configs/standard.yaml
```

The training step reads `embeddings.h5` via `ToxDataset` (`src/toxfam/data/dataset.py`), which looks up each protein's embedding by identifier.
