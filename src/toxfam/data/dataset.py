from __future__ import annotations

import math
import os

import h5py
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


# Organism (taxon) IDs for known venomous lineages.
# Covers major venomous clades: snakes (Serpentes 8570), spiders (Araneae 6893),
# scorpions (Scorpiones 6855), cone snails (Conidae 6490), jellyfish (Cubozoa 6082),
# bees/wasps (Aculeata 7434), centipedes (Chilopoda 7537).
_VENOMOUS_TAXA_IDS: frozenset[int] = frozenset({
    8570, 6893, 6855, 6490, 6082, 7434, 7537,
})


class ToxDataset(Dataset):
    """PyTorch Dataset for toxin-classification tasks that reads embeddings
    from multiple HDF5 files with LRU caching.

    Optionally concatenates auxiliary feature vectors (taxonomy, CPP, HBI)
    and scalar features (sequence length, venom indicator) to the embedding.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        h5_paths: list[str] | str,
        *,
        label_encoder: LabelEncoder | None = None,
        is_train: bool = True,
        label_col: str = "Protein families",
        cache_size: int = 3,
        tax_h5_path: str | None = None,
        cpp_h5_path: str | None = None,
        hbi_h5_path: str | None = None,
        include_length: bool = False,
        include_venom_indicator: bool = False,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.cache_size = cache_size
        self.include_length = include_length
        self.include_venom_indicator = include_venom_indicator

        if is_train:
            self.le = LabelEncoder()
            self.df[f"{label_col}_encoded"] = self.le.fit_transform(self.df[label_col])
        else:
            if label_encoder is None:
                raise ValueError("label_encoder must be provided when is_train=False")
            self.le = label_encoder
            self.df[f"{label_col}_encoded"] = self.le.transform(self.df[label_col])

        self.num_classes = len(self.le.classes_)

        if isinstance(h5_paths, str) and os.path.isdir(h5_paths):
            self.h5_paths = sorted(
                os.path.join(h5_paths, fn)
                for fn in os.listdir(h5_paths)
                if fn.endswith(".h5")
            )
        else:
            self.h5_paths = list(h5_paths)

        if not self.h5_paths:
            raise ValueError("No HDF5 files found.")

        self._open_cache: dict[str, h5py.File] = {}
        self._lru: list[str] = []

        # Auxiliary feature H5 files
        self.tax_h5 = h5py.File(tax_h5_path, "r") if tax_h5_path else None
        self.cpp_h5 = h5py.File(cpp_h5_path, "r") if cpp_h5_path else None
        self.hbi_h5 = h5py.File(hbi_h5_path, "r") if hbi_h5_path else None

    def _get_file_handle(self, path: str) -> h5py.File:
        if path in self._open_cache:
            self._lru.remove(path)
            self._lru.append(path)
            return self._open_cache[path]

        h5f = h5py.File(path, "r")
        self._open_cache[path] = h5f
        self._lru.append(path)

        if len(self._open_cache) > self.cache_size:
            oldest = self._lru.pop(0)
            try:
                self._open_cache[oldest].close()
            except Exception:
                pass
            del self._open_cache[oldest]
        return h5f

    def _find_embedding(self, protein_id: str):
        for path in self.h5_paths:
            h5f = self._get_file_handle(path)
            if protein_id in h5f:
                return h5f[protein_id][:]
        raise KeyError(f"Protein ID '{protein_id}' not found in any HDF5 file.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        protein_id = row["identifier"]
        embedding = self._find_embedding(protein_id)
        label = row[f"{self.label_col}_encoded"]

        emb_tensor = torch.tensor(embedding, dtype=torch.float32)

        # Concatenate auxiliary features
        aux_parts: list[torch.Tensor] = []

        if self.cpp_h5 is not None and protein_id in self.cpp_h5:
            aux_parts.append(
                torch.tensor(self.cpp_h5[protein_id][:], dtype=torch.float32)
            )

        if self.hbi_h5 is not None and protein_id in self.hbi_h5:
            aux_parts.append(
                torch.tensor(self.hbi_h5[protein_id][:], dtype=torch.float32)
            )

        if self.include_length:
            seq = row.get("Sequence", "")
            seq_len = len(seq) if isinstance(seq, str) else 0
            aux_parts.append(
                torch.tensor([math.log2(max(seq_len, 1))], dtype=torch.float32)
            )

        if self.include_venom_indicator:
            # Check if organism belongs to a known venomous taxon
            org_id = int(row.get("Organism (ID)", 0))
            is_venomous = 1.0 if org_id in _VENOMOUS_TAXA_IDS else 0.0
            aux_parts.append(torch.tensor([is_venomous], dtype=torch.float32))

        if aux_parts:
            emb_tensor = torch.cat([emb_tensor] + aux_parts, dim=0)

        # If taxonomy vectors are provided, return as tuple for MultiInputMLP
        if self.tax_h5 is not None:
            if protein_id not in self.tax_h5:
                raise KeyError(f"Protein '{protein_id}' not found in taxonomy H5.")
            tax_vec = self.tax_h5[protein_id][:]
            tax_tensor = torch.tensor(tax_vec, dtype=torch.float32)
            return (emb_tensor, tax_tensor), label
        else:
            return emb_tensor, label

    def close(self):
        for h5f in self._open_cache.values():
            try:
                h5f.close()
            except Exception:
                pass
        self._open_cache.clear()
        self._lru.clear()

        for h5 in (self.tax_h5, self.cpp_h5, self.hbi_h5):
            if h5 is not None:
                try:
                    h5.close()
                except Exception:
                    pass
        self.tax_h5 = None
        self.cpp_h5 = None
        self.hbi_h5 = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def analyze_data_splits(df: pd.DataFrame):
    """Return three DataFrames split by the ``Split`` column."""
    allowed = {"train", "val", "test"}
    if not set(df["Split"]).issubset(allowed):
        raise ValueError(f"Unexpected split names found: {set(df['Split']) - allowed}")

    train_df = df[df["Split"] == "train"].reset_index(drop=True)
    val_df = df[df["Split"] == "val"].reset_index(drop=True)
    test_df = df[df["Split"] == "test"].reset_index(drop=True)

    return train_df, val_df, test_df
