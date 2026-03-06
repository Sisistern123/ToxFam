# Taxonomy Pipeline

The taxonomy step (`uv run toxfam taxonomy`) converts NCBI taxon IDs into fixed-length multi-hot vectors that encode membership in 50 predefined animal taxa. Multiple taxa can be active per protein because the predefined taxa span different levels of the taxonomic hierarchy. These vectors are an optional input for the `combined` training strategy.

```bash
uv run toxfam taxonomy [--input-csv <csv>] [--input-h5 <h5>] [--output-h5 <h5>]
```

## How it works

1. **Read taxon IDs** — Loads the training CSV (produced by `toxfam preprocess`) and reads the `Organism (ID)` column, which contains NCBI taxon IDs already present in the raw UniProt data.
2. **Resolve lineage** — For each unique taxon ID, uses [taxopy](https://github.com/apcamargo/taxopy) to look up the **full NCBI taxonomy lineage**, collecting all ancestor names across every rank (including clades, superclasses, infraorders, etc. — not just the 8 standard ranks). The NCBI taxonomy database is cached locally in `~/.cache/taxopy_db/` and auto-refreshed weekly.
3. **Encode multi-hot vectors** — For each protein, checks whether any of its full lineage names match one of the 50 predefined taxa in the `TAXA` list. This produces a 50-dimensional multi-hot vector (1 = taxon present in lineage, 0 = absent). Multiple positions can be 1 simultaneously. Full-lineage matching is essential because NCBI places many venom-relevant taxa at non-standard ranks (e.g. `Toxicofera` is a clade, `Actinopterygii` is a superclass, `Sauropsida` is a clade).
4. **Write HDF5** — Iterates over every protein in the embeddings H5 and writes the corresponding multi-hot vector (or an all-zero vector for unmatched proteins) to the output H5 file with gzip compression. The `TAXA` list is stored as an HDF5 attribute for self-documenting output.

The pipeline is fully offline — it reads taxon IDs directly from the CSV rather than fetching them from UniProt's REST API.

## CLI options

| Option         | Default                                                 | Description                                      |
| -------------- | ------------------------------------------------------- | ------------------------------------------------ |
| `--input-csv`  | `data/processed/training_data.csv`                      | Training CSV with `Organism (ID)` column         |
| `--input-h5`   | `data/processed/embeddings.h5`                          | Embeddings H5 (determines which proteins to emit)|
| `--output-h5`  | `data/processed/taxonomy_vectors.h5` | Output H5 for multi-hot taxonomy vectors         |

## The 50 predefined taxa

The `TAXA` list (defined in `src/toxfam/data/taxonomy.py`) covers the major venomous and toxin-producing animal lineages represented in the dataset. Taxa were selected to be data-driven: every taxon has at least one protein in the training data, no taxon is constant (100%), and near-redundant parent-child pairs (>90% overlap) are collapsed to the parent.

```
Porifera, Ctenophora, Tentaculata, Cnidaria, Anthozoa, Spiralia,
Rotifera, Annelida, Glyceridae, Bryozoa, Gymnolaemata, Mollusca,
Bivalvia, Cephalopoda, Octopoda, Gastropoda, Neogastropoda, Conidae,
Polyplacophora, Echinodermata, Asteroidea, Echinoidea, Holothuroidea,
Panarthropoda, Onychophora, Tardigrada, Myriapoda, Chilopoda,
Arachnida, Araneae, Theraphosidae, Scorpiones, Buthidae, Insecta,
Hymenoptera, Chordata, Chondrichthyes, Dasyatidae, Actinopterygii,
Amphibia, Anura, Sauropsida, Aves, Squamata, Serpentes, Viperidae,
Crotalinae, Elapidae, Mammalia, Soricidae
```

Notable design decisions:
- **Full-lineage matching** — taxa at any NCBI rank (clade, superclass, infraorder, etc.) are matched, not just standard ranks
- **`Sauropsida`** instead of `Reptilia` — NCBI does not use "Reptilia"; Sauropsida is the clade containing reptiles and birds
- **`Spiralia`** instead of `Spiralia (Gnathifera)` — NCBI has these as separate nodes; `Spiralia` has broader coverage
- **Venom-relevant additions** — `Viperidae`, `Elapidae`, `Crotalinae`, `Serpentes`, `Buthidae`, `Theraphosidae`, `Conidae`, `Anthozoa`, `Amphibia`, `Anura`, `Gastropoda`
- **Removed** — `Metazoa` (always 1, zero information), near-redundant children (`Demospongiae` ≈ `Porifera`, `Conoidea` ≈ `Neogastropoda`, `Toxicofera` ≈ `Squamata`), and 12 taxa with no representation in the dataset

These taxa are not mutually exclusive — a single protein can match multiple taxa at different levels of the hierarchy (e.g. both `Araneae` and `Arachnida`).

## Taxopy database caching

On first run, taxopy downloads the NCBI taxonomy dump files (`nodes.dmp`, `names.dmp`, `merged.dmp`) to `~/.cache/taxopy_db/`. Subsequent runs reuse this cache. If the cache is older than one week, a background refresh is attempted; if the refresh fails, the stale cache is used as a fallback.

Override the cache directory with the `PROTSPACE_TAXDB_DIR` environment variable:

```bash
PROTSPACE_TAXDB_DIR=/path/to/taxdb uv run toxfam taxonomy
```

## Output format

The HDF5 file contains one dataset per protein (gzip-compressed), keyed by identifier. The `TAXA` list is stored as the `taxa` attribute on the root group.

```python
import h5py

with h5py.File("data/processed/taxonomy_vectors.h5", "r") as f:
    taxa_names = list(f.attrs["taxa"])    # the 50 taxon names, in order
    vec = f["P01234"][:]                  # shape: (50,), dtype: float32, values: 0.0 or 1.0
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
              data/processed/taxonomy_vectors.h5
                              │
                   uv run toxfam train configs/combined.yaml
```

The training step reads `taxonomy_vectors.h5` via `ToxDataset` when the `combined` strategy is selected. A dimension mismatch between the H5 vectors and `tax_dim` in the config is caught at startup with a clear error. The `standard` strategy does not require taxonomy vectors.
