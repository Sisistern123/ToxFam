import os
from typing import List, Optional, Dict

import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ToxDataset(Dataset):
    """PyTorch ``Dataset`` for toxin‑classification tasks that can read embeddings
    distributed across **multiple** HDF5 files.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the columns ``Entry`` (UniProt/sequence ID),
        the categorical label column (default: ``"Protein families"``), and a
        column named ``Split`` whose values are ``"train"``, ``"val"`` or
        ``"test"``.
    h5_paths : list[str] | str
        Either a list with absolute / relative paths to the HDF5 files *or* a
        path to a directory that only contains the relevant ``*.h5`` files.
    label_encoder : sklearn.preprocessing.LabelEncoder | None
        Pass an already‑fitted instance when *not* in training mode so that
        labels are encoded identically in all splits.
    is_train : bool, default True
        If *True*, a new ``LabelEncoder`` is fitted; otherwise ``label_encoder``
        is mandatory.
    label_col : str, default "Protein families"
        Name of the column that contains the *human‑readable* labels.
    cache_size : int, default 3
        Number of HDF5 file handles to keep open concurrently. When the limit
        is exceeded the least‑recently‑used handle is closed.

    Notes
    -----
    Each HDF5 file must contain **one group per protein ID**, matching the
    values in the DataFrame's ``Entry`` column, and each group must be an array
    with the embedding (e.g. shape ``(1024,)`` or ``(n_layers, n_features)``).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        h5_paths: List[str] | str,
        *,
        label_encoder: Optional[LabelEncoder] = None,
        is_train: bool = True,
        label_col: str = "Protein families",
        cache_size: int = 3,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.cache_size = cache_size

        # ----- label encoding -------------------------------------------------
        if is_train:
            self.le = LabelEncoder()
            self.df[f"{label_col}_encoded"] = self.le.fit_transform(
                self.df[label_col]
            )
        else:
            if label_encoder is None:
                raise ValueError("label_encoder must be provided when is_train=False")
            self.le = label_encoder
            self.df[f"{label_col}_encoded"] = self.le.transform(self.df[label_col])

        self.num_classes = len(self.le.classes_)

        # ----- handle HDF5 files ---------------------------------------------
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

        # lazy‑open cache: {path: h5py.File}
        self._open_cache: Dict[str, h5py.File] = {}
        self._lru: List[str] = []  # keep order of usage

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------
    def _get_file_handle(self, path: str) -> h5py.File:
        """Return an *open* h5py.File handle respecting the LRU cache."""
        if path in self._open_cache:
            # mark as recently used
            self._lru.remove(path)
            self._lru.append(path)
            return self._open_cache[path]

        # need to open new handle
        h5f = h5py.File(path, "r")
        self._open_cache[path] = h5f
        self._lru.append(path)

        # evict least‑recently‑used if cache is full
        if len(self._open_cache) > self.cache_size:
            oldest = self._lru.pop(0)
            try:
                self._open_cache[oldest].close()
            except Exception:
                pass
            del self._open_cache[oldest]
        return h5f

    def _find_embedding(self, protein_id: str):
        """Retrieve the embedding array for *protein_id* by scanning the HDF5 files."""
        for path in self.h5_paths:
            h5f = self._get_file_handle(path)
            if protein_id in h5f:
                return h5f[protein_id][:]
        raise KeyError(f"Protein ID '{protein_id}' not found in any HDF5 file.")

    # ---------------------------------------------------------------------
    # PyTorch Dataset API
    # ---------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401, PLE, N802
        return len(self.df)

    def __getitem__(self, index: int):  # noqa: D401, PLE, N802
        row = self.df.iloc[index]
        protein_id = row["Entry"]
        embedding = self._find_embedding(protein_id)
        label = row[f"{self.label_col}_encoded"]
        return torch.tensor(embedding, dtype=torch.float32), label

    # ---------------------------------------------------------------------
    # cleanup
    # ---------------------------------------------------------------------
    def close(self):  # noqa: D401, PLE, N802
        """Close **all** open HDF5 handles."""
        for h5f in self._open_cache.values():
            try:
                h5f.close()
            except Exception:
                pass
        self._open_cache.clear()
        self._lru.clear()

    def __del__(self):  # noqa: D401, PLE, N802
        try:
            self.close()
        except Exception:
            pass


# -------------------------------------------------------------------------
# Utility to split DataFrame ------------------------------------------------
# -------------------------------------------------------------------------

def analyze_data_splits(df: pd.DataFrame):
    """Return three DataFrames split by the ``Split`` column.

    Examples
    --------
    >>> train_df, val_df, test_df = analyze_data_splits(main_df)
    """
    allowed = {"train", "val", "test"}
    if not set(df["Split"]).issubset(allowed):
        raise ValueError(f"Unexpected split names found: {set(df['Split']) - allowed}")

    train_df = df[df["Split"] == "train"].reset_index(drop=True)
    val_df = df[df["Split"] == "val"].reset_index(drop=True)
    test_df = df[df["Split"] == "test"].reset_index(drop=True)

    return train_df, val_df, test_df
