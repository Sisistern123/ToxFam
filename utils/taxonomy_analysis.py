import pandas as pd
import h5py
import numpy as np
import ast

# =============================================================================
# Configuration
# =============================================================================

NORMALIZE_TAXONOMY = True  # for the numeric taxonomy pipeline

# Single CSV with:
# - identifier
# - tax_array (IDs)
# - domain, kingdom, phylum, class, order, family, genus, species
TAX_CSV = "../data/tax/training_tax.csv"
# ProtT5 embeds
INPUT_H5 = "../data/protspace/training_data.h5"

# H5 output with embeddings + numeric taxonomy (your original behaviour)
NUMERIC_OUTPUT_H5 = "../data/tax/normed_training_data_with_tax.h5"

# H5 output that will contain ONLY the one-hot/binary taxonomy vectors
BINARY_OUTPUT_H5 = "../data/tax/binary_taxonomy_vectors.h5"

# List of taxa in the final binary array (order = column order)
TAXA = [
    "Porifera",
    "Calcarea",
    "Demospongiae",
    "Hexactinellida",
    "Ctenophora",
    "Nuda",
    "Tentaculata",
    "Cnidaria",
    "Metazoa",
    "Spiralia (Gnathifera)",
    "Rotifera",
    "Annelida",
    "Glyceridae",
    "Bryozoa",
    "Gymnolaemata",
    "Phylactolaemata",
    "Stenolaemata",
    "Mollusca",
    "Bivalvia",
    "Cephalopoda",
    "Octopoda",
    "Neogastropoda",
    "Conoidea",
    "Polyplacophora",
    "Scaphopoda",
    "Echinodermata",
    "Asteroidea",
    "Crinoidea",
    "Echinoidea",
    "Holothuroidea",
    "Ophiuroidea",
    "Panarthropoda",
    "Onychophora",
    "Tardigrada",
    "Myriapoda",
    "Chilopoda",
    "Diplopoda",
    "Arachnida",
    "Araneae",
    "Pseudoscorpiones",
    "Scorpiones",
    "Insecta",
    "Hymenoptera",
    "Chordata",
    "Aves",
    "Chondrichthyes",
    "Dasyatidae",
    "Actinopterygii",
    "Scorpaenoidei",
    "Trachinidae",
    "Reptilia",
    "Squamata",
    "Toxicofera",
    "Mammalia",
    "Solenodontidae",
    "Soricidae",
]

# =============================================================================
# OPTION 1: your original script in a function (numeric tax + append to embeddings)
# =============================================================================

def run_numeric_taxonomy_pipeline(
    tax_csv_path: str = TAX_CSV,
    input_path: str = INPUT_H5,
    output_path: str = NUMERIC_OUTPUT_H5,
    normalize: bool = NORMALIZE_TAXONOMY,
):
    """Original behaviour: append numeric taxonomy vectors to embeddings."""
    # Load taxonomy data
    tax = pd.read_csv(tax_csv_path)

    # Parse tax_array strings to lists (tax IDs)
    tax["tax_array"] = tax["tax_array"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Create dict: identifier -> tax_array
    tax_dict = dict(zip(tax["identifier"], tax["tax_array"]))

    # Determine taxonomy vector length (e.g. 6)
    if len(tax_dict) > 0:
        example_array = next(iter(tax_dict.values()))
        tax_vec_len = len(example_array)
    else:
        tax_vec_len = 0

    # Optional normalization
    if normalize and tax_vec_len > 0:
        all_tax_values = []
        for tax_array in tax_dict.values():
            all_tax_values.extend(tax_array)

        tax_min = min(all_tax_values)
        tax_max = max(all_tax_values)

        print(f"Taxonomy value range: [{tax_min}, {tax_max}]")
        print("Will normalize to range: [-2, 2]\n")

        def normalize_taxonomy(tax_array):
            """Normalize taxonomy values from their original range to [-2, 2]."""
            tax_array = np.array(tax_array, dtype=np.float32)
            normalized = (tax_array - tax_min) / (tax_max - tax_min)  # [0, 1]
            normalized = normalized * 4 - 2  # [-2, 2]
            return normalized

    else:
        print("Taxonomy normalization: DISABLED\n")
        normalize_taxonomy = None

    print(f"Loaded {len(tax_dict)} numeric taxonomy entries")

    # Open the original H5 file and create a new one with appended taxonomy
    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        total_entries = len(f_in.keys())
        matched = 0
        unmatched = 0
        unmatched_ids = []

        for i, protein_id in enumerate(f_in.keys()):
            # Original embedding (e.g. (1024,))
            embedding = f_in[protein_id][:]

            # Get taxonomy array for this protein
            if protein_id in tax_dict:
                tax_array = tax_dict[protein_id]

                # Apply normalization if enabled
                if normalize_taxonomy is not None:
                    tax_array = normalize_taxonomy(tax_array)
                else:
                    tax_array = np.array(tax_array, dtype=embedding.dtype)

                # CONCAT HERE: embeddings + numeric taxonomy
                combined = np.concatenate([embedding, tax_array])
                matched += 1
            else:
                # If no taxonomy found, append zeros of same length
                tax_array = np.zeros(tax_vec_len, dtype=embedding.dtype)
                combined = np.concatenate([embedding, tax_array])
                unmatched += 1
                unmatched_ids.append(protein_id)

            # Write to new file
            f_out.create_dataset(protein_id, data=combined)

            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1}/{total_entries} entries...")

        print(f"\nProcessing complete! (numeric taxonomy)")
        print(f"Total entries: {total_entries}")
        print(f"Matched with taxonomy: {matched}")
        print(f"Unmatched (filled with zeros): {unmatched}")
        print(f"Normalization applied: {normalize and (normalize_taxonomy is not None)}")

        if unmatched > 0:
            print(f"\nFirst 10 unmatched IDs: {unmatched_ids[:10]}")

    print("\n" + "=" * 70)
    print("Numeric taxonomy pipeline finished.")
    print("=" * 70)


# =============================================================================
# OPTION 2: one-hot / binary taxonomy as a SEPARATE input (NO concatenation)
# =============================================================================

def build_binary_tax_dict(
    detailed_tax_csv_path: str = TAX_CSV,
    id_col: str = "identifier",
):
    """
    Build a dict: identifier -> np.array of 0/1 (len = len(TAXA)),
    based on domain/kingdom/phylum/class/order/family/genus/species columns.
    """
    df = pd.read_csv(detailed_tax_csv_path)

    tax_cols = [
        "domain",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]

    missing = [c for c in tax_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing taxonomy columns in {detailed_tax_csv_path}: {missing}")

    # Normalize case/whitespace
    for c in tax_cols:
        df[c] = df[c].astype(str).str.strip().str.lower()

    taxa_norm = [t.strip().lower() for t in TAXA]

    # Create one 0/1 column per taxon
    for original_name, norm_name in zip(TAXA, taxa_norm):
        df[original_name] = (df[tax_cols] == norm_name).any(axis=1).astype(np.float32)

    # Build dict: identifier -> 0/1 vector
    tax_dict = {}
    for _, row in df.iterrows():
        identifier = row[id_col]
        tax_array = row[TAXA].to_numpy(dtype=np.float32)
        tax_dict[identifier] = tax_array

    print(f"Built binary taxonomy dict for {len(tax_dict)} identifiers")
    print(f"Binary taxonomy vector length: {len(TAXA)}")
    return tax_dict


def run_binary_taxonomy_pipeline(
    detailed_tax_csv_path: str = TAX_CSV,
    id_col: str = "identifier",
    input_path: str = INPUT_H5,
    output_path: str = BINARY_OUTPUT_H5,
):
    """
    For each protein in INPUT_H5, create a one-hot / binary taxonomy vector
    (length len(TAXA)) and write it to a SEPARATE H5 file.

    - The embeddings in INPUT_H5 are NOT modified.
    - The output H5 contains ONLY the binary taxonomy vectors, keyed by protein_id.
    """
    tax_dict = build_binary_tax_dict(detailed_tax_csv_path, id_col=id_col)
    vec_len = len(TAXA)

    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        total_entries = len(f_in.keys())
        matched = 0
        unmatched = 0
        unmatched_ids = []

        for i, protein_id in enumerate(f_in.keys()):
            # Use keys from the embeddings file to know which proteins exist.
            if protein_id in tax_dict:
                vec = tax_dict[protein_id].astype(np.float32)
                matched += 1
            else:
                # If this protein has no taxonomy row, use an all-zero vector
                vec = np.zeros(vec_len, dtype=np.float32)
                unmatched += 1
                unmatched_ids.append(protein_id)

            # Store ONLY the taxonomy vector, no concatenation
            f_out.create_dataset(protein_id, data=vec)

            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1}/{total_entries} entries...")

        print(f"\nProcessing complete! (binary one-hot taxonomy)")
        print(f"Total entries (proteins): {total_entries}")
        print(f"Matched with taxonomy: {matched}")
        print(f"Unmatched (all-zero vector): {unmatched}")

        if unmatched > 0:
            print(f"\nFirst 10 unmatched IDs: {unmatched_ids[:10]}")

    print("\n" + "=" * 70)
    print("Binary taxonomy pipeline finished.")
    print(f"Output file (only one-hot vectors): {output_path}")
    print("=" * 70)


# =============================================================================
# Entry point: choose which option to run
# =============================================================================

if __name__ == "__main__":
    # OPTION 1: original script behaviour (embeddings + numeric tax IDs)
    # run_numeric_taxonomy_pipeline()

    # OPTION 2: one-hot encoded binary arrays as separate input
    run_binary_taxonomy_pipeline()
