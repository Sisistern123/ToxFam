# CPP (Comparative Physicochemical Profiling) Features

## Overview

CPP features provide physicochemical descriptors that distinguish toxic from non-toxic protein sequences. They are generated using the AAanalysis library and concatenated onto ProtT5 embeddings during training.

## Pipeline

1. **Input**: `training_data.csv` with `is_toxic` labels and protein `Sequence` column
2. **Processing**: AAanalysis computes comparative physicochemical profiles between toxic and non-toxic sequence sets
3. **Feature selection**: Top `n_filter` most discriminating features are selected (default: 100)
4. **Output**: HDF5 file (`data/intermediate/cpp/cpp_features.h5`) keyed by protein identifier

## CLI Usage

```bash
# Generate CPP features
uv run toxfam cpp --training-csv data/processed/training_data.csv

# With custom output and feature count
uv run toxfam cpp --training-csv data/processed/training_data.csv -o data/intermediate/cpp/custom.h5 --n-filter 200
```

## Integration with Training

CPP features are concatenated onto ProtT5 embeddings in `ToxDataset.__getitem__()`, making the effective input dimension `embedding_dim + cpp_dim` (e.g., 1024 + 100 = 1124).

### Config

```yaml
training_strategy: "binary"
cpp_h5_path: "data/intermediate/cpp/cpp_features.h5"
cpp_dim: 100
embedding_dim: 1024  # ProtT5 dimension (CPP is added on top)
```

The `effective_embedding_dim` property on `TrainConfig` automatically computes the correct input dimension:

```python
config.effective_embedding_dim  # Returns 1124 when cpp_h5_path is set
```

### Pre-built Config

```bash
uv run toxfam train configs/binary_cpp.yaml
```

## Output Format

- **File**: HDF5 with one dataset per protein
- **Key**: protein identifier (e.g., `"P12345"`)
- **Value**: 1D float32 array of length `cpp_dim`

## Dependencies

CPP feature generation requires [AAanalysis](https://github.com/breimanntools/aaanalysis), installed as an editable local dependency. See project setup for details.
