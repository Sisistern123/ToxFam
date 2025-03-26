import torch
import h5py
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class ToxDataset(Dataset):
    def __init__(self, df, h5_path, label_encoder=None, is_train=True, label_col='Protein families'):
        """
        Dataset for toxin protein data

        Args:
            df: DataFrame with protein data
            h5_path: Path to h5 file with embeddings
            label_encoder: Optional pre-fitted LabelEncoder
            is_train: Whether this is a training dataset
            label_col: Column name in df for labels
        """
        self.df = df.reset_index(drop=True)
        self.h5f = h5py.File(h5_path, 'r')
        self.label_col = label_col

        if is_train:
            self.le = LabelEncoder()
            self.df[label_col + '_encoded'] = self.le.fit_transform(self.df[label_col])
        else:
            self.le = label_encoder
            self.df[label_col + '_encoded'] = self.le.transform(self.df[label_col])

        self.num_classes = len(self.le.classes_)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        protein_id = row['Entry']
        embedding = self.h5f[protein_id][:]
        label = row[self.label_col + '_encoded']
        return torch.tensor(embedding, dtype=torch.float32), label

    def close(self):
        """Explicitly close the h5 file"""
        if hasattr(self, "h5f") and self.h5f is not None:
            try:
                self.h5f.close()
            except Exception:
                pass
            self.h5f = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def analyze_data_splits(df):
    """
    Analyze data splits for train/val/test and return split DataFrames.
    (Missing-label information will be computed per label later.)

    Args:
        df: Main DataFrame with split indicators

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Convert split indicator columns to bool
    df["Train_Cluster_Rep"] = df["Train_Cluster_Rep"].astype(bool)
    df["Val_Cluster_Rep"] = df["Val_Cluster_Rep"].astype(bool)
    df["Test_Cluster_Rep"] = df["Test_Cluster_Rep"].astype(bool)

    train_df = df[df["Train_Cluster_Rep"] == True]
    val_df = df[df["Val_Cluster_Rep"] == True]
    test_df = df[df["Test_Cluster_Rep"] == True]

    return train_df, val_df, test_df
