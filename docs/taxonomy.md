# Taxonomy Pipeline

The taxonomy step (`uv run toxfam taxonomy`) converts NCBI taxon IDs into fixed-length binary vectors that encode membership in 56 predefined animal taxa. These vectors are an optional input for the `combined` training strategy.

```bash
uv run toxfam taxonomy [--input-csv <csv>] [--input-h5 <h5>] [--output-h5 <h5>]
```

## How it works

1. **Read taxon IDs** — Loads the training CSV (produced by `toxfam preprocess`) and reads the `Organism (ID)` column, which contains NCBI taxon IDs already present in the raw UniProt data.
2. **Resolve lineage** — For each unique taxon ID, uses [taxopy](https://github.com/apcamargo/taxopy) to look up the full NCBI taxonomy lineage (root, domain, kingdom, phylum, class, order, family, genus, species). The NCBI taxonomy database is cached locally in `~/.cache/taxopy_db/` and auto-refreshed weekly.
3. **Encode binary vectors** — For each protein, checks whether any of its lineage ranks match one of the 56 predefined taxa in the `TAXA` list. This produces a 56-dimensional binary vector (1 = taxon present in lineage, 0 = absent).
4. **Write HDF5** — Iterates over every protein in the embeddings H5 and writes the corresponding binary vector (or an all-zero vector for unmatched proteins) to the output H5 file.

The pipeline is fully offline — it reads taxon IDs directly from the CSV rather than fetching them from UniProt's REST API.

## CLI options

| Option         | Default                                                 | Description                                      |
| -------------- | ------------------------------------------------------- | ------------------------------------------------ |
| `--input-csv`  | `data/processed/training_data.csv`                      | Training CSV with `Organism (ID)` column         |
| `--input-h5`   | `data/processed/embeddings.h5`                          | Embeddings H5 (determines which proteins to emit)|
| `--output-h5`  | `data/intermediate/taxonomy/binary_taxonomy_vectors.h5` | Output H5 for binary taxonomy vectors            |

## The 56 predefined taxa

The `TAXA` list (defined in `src/toxfam/data/taxonomy.py`) covers the major animal lineages represented in the toxin dataset:

```
Porifera, Calcarea, Demospongiae, Hexactinellida, Ctenophora, Nuda,
Tentaculata, Cnidaria, Metazoa, Spiralia (Gnathifera), Rotifera,
Annelida, Glyceridae, Bryozoa, Gymnolaemata, Phylactolaemata,
Stenolaemata, Mollusca, Bivalvia, Cephalopoda, Octopoda,
Neogastropoda, Conoidea, Polyplacophora, Scaphopoda, Echinodermata,
Asteroidea, Crinoidea, Echinoidea, Holothuroidea, Ophiuroidea,
Panarthropoda, Onychophora, Tardigrada, Myriapoda, Chilopoda,
Diplopoda, Arachnida, Araneae, Pseudoscorpiones, Scorpiones,
Insecta, Hymenoptera, Chordata, Aves, Chondrichthyes, Dasyatidae,
Actinopterygii, Scorpaenoidei, Trachinidae, Reptilia, Squamata,
Toxicofera, Mammalia, Solenodontidae, Soricidae
```

These taxa are not mutually exclusive — a single protein can match multiple taxa at different levels of the hierarchy (e.g. both `Araneae` and `Arachnida`).

## Taxopy database caching

On first run, taxopy downloads the NCBI taxonomy dump files (`nodes.dmp`, `names.dmp`, `merged.dmp`) to `~/.cache/taxopy_db/`. Subsequent runs reuse this cache. If the cache is older than one week, a background refresh is attempted; if the refresh fails, the stale cache is used as a fallback.

Override the cache directory with the `PROTSPACE_TAXDB_DIR` environment variable:

```bash
PROTSPACE_TAXDB_DIR=/path/to/taxdb uv run toxfam taxonomy
```

## Output format

The HDF5 file contains one dataset per protein, keyed by identifier:

```python
import h5py

with h5py.File("data/intermediate/taxonomy/binary_taxonomy_vectors.h5", "r") as f:
    vec = f["P01234"][:]  # shape: (56,), dtype: float32, values: 0.0 or 1.0
```

## Where it fits in the pipeline

```
preprocessing ─→ data/processed/training_data.csv  (contains Organism (ID))
                              │
                   uv run toxfam embed
                              │
                    data/processed/embeddings.h5
                              │
                   uv run toxfam taxonomy
                              │
              data/intermediate/taxonomy/binary_taxonomy_vectors.h5
                              │
                   uv run toxfam train configs/combined.yaml
```

The training step reads `binary_taxonomy_vectors.h5` via `ToxDataset` when the `combined` strategy is selected. The `standard` strategy does not require taxonomy vectors.
