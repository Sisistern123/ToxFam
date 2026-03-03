# SignalP6 Setup

SignalP 6.0 predicts signal peptides in protein sequences. ToxFam uses it during preprocessing to remove signal peptides from toxin and non-toxin sequences before clustering and training.

SignalP6 requires **Python 3.10** and **PyTorch <2.0**, which conflict with ToxFam's own dependencies. To handle this, SignalP6 runs in an isolated uv project under `tools/signalp6/`.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) (already required for ToxFam)
- Python 3.10 available on your system (uv will auto-download it if needed)

## Step 1: Obtain SignalP 6.0

SignalP 6.0 is distributed by DTU Health Technology. You need the Python package with model weights.

**Academic users:**
1. Go to <https://services.healthtech.dtu.dk/services/SignalP-6.0/>
2. Navigate to the Downloads section
3. Download the `signalp-6-package` archive (choose the `slow-sequential` model for best accuracy)

**Non-academic users:**
- Contact DTU at health-software@dtu.dk for licensing

**From GitHub (source code only, no model weights):**
- <https://github.com/fteufel/signalp-6.0>
- You still need to obtain model weight files separately

## Step 2: Install into ToxFam

Place the unpacked `signalp-6-package` directory under `tools/signalp6/bin/`:

```bash
# If you downloaded a tar.gz:
tar xzf signalp6*.tar.gz
mv signalp-6-package tools/signalp6/bin/

# Or symlink from an existing installation:
ln -s /path/to/existing/signalp-6-package tools/signalp6/bin/signalp-6-package
```

The directory structure should look like:

```
tools/signalp6/
├── pyproject.toml
├── .python-version        # Pins Python 3.10
├── .gitignore
└── bin/
    └── signalp-6-package/
        ├── signalp/       # Python package
        ├── models/        # Model weight files
        ├── setup.py
        ├── requirements.txt
        └── README.md
```

## Step 3: Install model weights

If the model weights are not already in `signalp-6-package/models/`, copy them there:

```bash
# Copy into the models directory
cp -r /path/to/downloaded/models/* tools/signalp6/bin/signalp-6-package/models/
```

The `slow-sequential` mode (used by ToxFam) requires:
- `models/sequential_models_signalp6/` (6 cross-validation models + viterbi decoder)

For `fast` mode (lower accuracy, lower memory):
- `models/distilled_model_signalp6.pt`

## Step 4: Create the environment

```bash
cd tools/signalp6
uv sync
```

This creates a `.venv` with Python 3.10, PyTorch 1.13, and SignalP6 installed.

## Step 5: Verify

```bash
# From the project root:
uv run --project tools/signalp6 signalp6 --version
# Expected: SignalP 6.0 Signal peptide prediction tool 6.0h
```

## Usage

### Via the ToxFam preprocessing pipeline

SignalP6 runs automatically during preprocessing:

```bash
uv run toxfam preprocess
```

SignalP6 is required. If it is not installed, the pipeline will fail with an error.

### Standalone

Run SignalP6 directly on any FASTA file:

```bash
uv run --project tools/signalp6 signalp6 \
    --fastafile input.fasta \
    --output_dir output/ \
    --organism eukarya \
    --mode slow-sequential \
    --format none
```

Key options:
- `--organism`: `eukarya` (limits to Sec/SPI predictions) or `other`
- `--mode`: `slow-sequential` (recommended, used by ToxFam) or `fast` (lower accuracy)
- `--format`: `none` (summary only), `txt`, `png`, `eps`, or `all`
- `--bsize`: batch size (default 10, tune for memory)
- `--model_dir`: custom model weights path (if not in default location)

### Output files

SignalP6 produces these files in the output directory:

| File | Description |
|---|---|
| `prediction_results.txt` | Tab-delimited predictions with SP type and cleavage site |
| `processed_entries.fasta` | Sequences with signal peptides removed |
| `output.gff3` | Signal peptide positions in GFF3 format |
| `region_output.gff3` | Signal peptide region details |
| `output.json` | Complete results in JSON |

## How ToxFam uses SignalP6

During preprocessing (`src/toxfam/data/preprocessing.py`):

1. SignalP6 runs on `tox.fasta` and `nontox.fasta` (from `data/intermediate/fasta/`)
2. Results are cached in `data/intermediate/sp6/{tox,nontox}/`
3. For proteins with a predicted signal peptide (confidence > 0.8), the sequence is replaced with the mature protein (signal peptide removed)
4. Proteins without a signal peptide keep their original sequence

## Troubleshooting

**"signalp-6-package not found"**
- Ensure the package is at `tools/signalp6/bin/signalp-6-package/`
- Check that `signalp/__init__.py` exists inside it

**Python version errors**
- SignalP6 requires Python >=3.10,<3.11
- Run `uv python install 3.10` if uv can't find Python 3.10

**PyTorch errors**
- SignalP6 is incompatible with PyTorch >=2.0
- The `pyproject.toml` pins `numpy<2` and PyTorch is pinned <2 via signalp6's `requirements.txt`

**GPU support**
- By default, SignalP6 runs on CPU
- To convert to GPU: `uv run --project tools/signalp6 signalp6_convert_models gpu`
- To revert: `uv run --project tools/signalp6 signalp6_convert_models cpu`

## References

- SignalP 6.0 paper: Teufel et al., *Nature Biotechnology* (2022)
- Web service: <https://services.healthtech.dtu.dk/services/SignalP-6.0/>
- GitHub: <https://github.com/fteufel/signalp-6.0>
