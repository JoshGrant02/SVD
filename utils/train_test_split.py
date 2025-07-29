import pickle
from dataclasses import dataclass
import pandas as pd
import os
from scipy.sparse._csr import csr_matrix

@dataclass
class DatasetDivision:
    dataset_name: str
    df_train: pd.DataFrame
    df_test_seen: pd.DataFrame
    df_test_unseen: pd.DataFrame
    X_train: csr_matrix
    X_test_seen: csr_matrix
    X_test_unseen: csr_matrix

    def save_to_pickle(self, base_dir: str ="."):
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, f"{self.dataset_name}_tts.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_pickle(cls, dataset_name: str, base_dir: str =".") -> "DatasetDivision":
        path = os.path.join(base_dir, f"{dataset_name}_tts.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)

@dataclass
class DatasetDivisionOld:
    dataset_name: str
    df_train: pd.DataFrame
    X_train: csr_matrix
    X_test: csr_matrix
    df_test_unseen: pd.DataFrame

    def save_to_pickle(self, base_dir: str ="."):
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, f"{self.dataset_name}_tts_old.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_pickle(cls, dataset_name: str, base_dir: str =".") -> "DatasetDivision":
        path = os.path.join(base_dir, f"{dataset_name}_tts_old.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)
