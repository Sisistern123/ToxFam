from __future__ import annotations

import os
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class ToxDataset(Dataset):
    """PyTorch Dataset for toxin-classification tasks that reads embeddings
    from multiple HDF5 files with LRU caching."""

    # Known venomous taxa (order/family level) for the venom indicator feature
    VENOMOUS_TAXA_IDS: set[int] = {
        8570,    # Serpentes (snakes)
        6893,    # Scorpiones (scorpions)
        6854,    # Araneae (spiders)
        6960,    # Hymenoptera (bees, wasps, ants)
        6101,    # Cnidaria (jellyfish, anemones, corals)
        6447,    # Mollusca/Gastropoda (cone snails via Conus)
        61985,   # Chilopoda (centipedes)
        8504,    # Helodermatidae (Gila monster, beaded lizard)
        9400,    # Soricidae (venomous shrews)
        31922,   # Octopoda (blue-ringed octopus etc.)
    }

    def __init__(
        self,
        df: pd.DataFrame,
        h5_paths: List[str] | str,
        *,
        label_encoder: Optional[LabelEncoder] = None,
        is_train: bool = True,
        label_col: str = "Protein families",
        cache_size: int = 3,
        tax_h5_path: str | None = None,
        cpp_h5_path: str | None = None,
        hbi_h5_path: str | None = None,
        handcrafted_h5_path: str | None = None,
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

        self._open_cache: Dict[str, h5py.File] = {}
        self._lru: List[str] = []

        self.tax_h5 = None
        if tax_h5_path is not None:
            self.tax_h5 = h5py.File(tax_h5_path, "r")

        self.cpp_h5 = None
        if cpp_h5_path is not None:
            self.cpp_h5 = h5py.File(cpp_h5_path, "r")

        self.hbi_h5 = None
        if hbi_h5_path is not None:
            self.hbi_h5 = h5py.File(hbi_h5_path, "r")

        self.handcrafted_h5 = None
        if handcrafted_h5_path is not None:
            self.handcrafted_h5 = h5py.File(handcrafted_h5_path, "r")

        # Pre-compute venom indicator if needed
        if self.include_venom_indicator and "Organism (ID)" in self.df.columns:
            self._venom_indicators = self._compute_venom_indicators()
        else:
            self._venom_indicators = None

    def _compute_venom_indicators(self) -> np.ndarray:
        """Pre-compute binary venom indicator from organism taxonomy IDs."""
        indicators = np.zeros(len(self.df), dtype=np.float32)
        for i, org_id in enumerate(self.df["Organism (ID)"]):
            try:
                org_id_int = int(org_id)
                # Check if organism is in a known venomous taxon
                # For simplicity, check direct match; a full taxopy lookup
                # would be more thorough but slower
                if org_id_int in self.VENOMOUS_TAXA_IDS:
                    indicators[i] = 1.0
            except (ValueError, TypeError):
                pass
        return indicators

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

        # Concatenate CPP features if available
        if self.cpp_h5 is not None and protein_id in self.cpp_h5:
            cpp_vec = self.cpp_h5[protein_id][:]
            embedding = np.concatenate([embedding, cpp_vec])

        # Concatenate HBI features if available
        if self.hbi_h5 is not None and protein_id in self.hbi_h5:
            hbi_vec = self.hbi_h5[protein_id][:]
            embedding = np.concatenate([embedding, hbi_vec])

        # Concatenate handcrafted features if available
        if self.handcrafted_h5 is not None and protein_id in self.handcrafted_h5:
            hc_vec = self.handcrafted_h5[protein_id][:]
            embedding = np.concatenate([embedding, hc_vec])

        # Append log2(length) feature
        if self.include_length:
            seq = row.get("Sequence", "")
            length = len(str(seq)) if pd.notna(seq) else 0
            length_feat = np.array(
                [np.log2(max(length, 1))], dtype=np.float32
            )
            embedding = np.concatenate([embedding, length_feat])

        # Append venomous organism indicator
        if self.include_venom_indicator and self._venom_indicators is not None:
            venom_feat = np.array(
                [self._venom_indicators[index]], dtype=np.float32
            )
            embedding = np.concatenate([embedding, venom_feat])

        if self.tax_h5 is not None:
            if protein_id not in self.tax_h5:
                raise KeyError(f"Protein '{protein_id}' not found in taxonomy H5.")
            tax_vec = self.tax_h5[protein_id][:]
            emb_tensor = torch.tensor(embedding, dtype=torch.float32)
            tax_tensor = torch.tensor(tax_vec, dtype=torch.float32)
            return (emb_tensor, tax_tensor), label
        else:
            return torch.tensor(embedding, dtype=torch.float32), label

    def close(self):
        for h5f in self._open_cache.values():
            try:
                h5f.close()
            except Exception:
                pass
        self._open_cache.clear()
        self._lru.clear()

        if self.tax_h5 is not None:
            try:
                self.tax_h5.close()
            except Exception:
                pass
            self.tax_h5 = None

        if self.cpp_h5 is not None:
            try:
                self.cpp_h5.close()
            except Exception:
                pass
            self.cpp_h5 = None

        if self.hbi_h5 is not None:
            try:
                self.hbi_h5.close()
            except Exception:
                pass
            self.hbi_h5 = None

        if self.handcrafted_h5 is not None:
            try:
                self.handcrafted_h5.close()
            except Exception:
                pass
            self.handcrafted_h5 = None

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
