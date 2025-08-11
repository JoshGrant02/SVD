import pickle
from dataclasses import dataclass
import pandas as pd
import os
from scipy.sparse._csr import csr_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from scipy import sparse


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


def divide_dataset(df: pd.DataFrame, dataset_name: str):
    df = df[~df["user_id"].isnull()].copy()
    df["item_id"], unique_item_ids = pd.factorize(df["item_id"])
    n_items = len(unique_item_ids)

    # Train Test Split
    print("Performing TTS")
    gss = GroupShuffleSplit(n_splits = 1, train_size=0.75, random_state=12)
    groups = gss.split(df, groups=df["user_id"])
    train_indices = []
    test_indices = []
    for fold_train_indices, fold_test_indices in groups:
        train_indices = fold_train_indices
        test_indices = fold_test_indices
    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()

    # Sparsify Train
    df_train["user_id"], unique_train_user_ids = pd.factorize(df_train["user_id"])
    n_training_users = len(unique_train_user_ids)
    X_train = sparse.csr_matrix(
        (df_train["rating"], (df_train["user_id"], df_train["item_id"])),
        shape=(n_training_users, n_items)
    )

    # Seen Unseen Split
    df_test["user_id"], unique_test_user_ids = pd.factorize(df_test["user_id"])
    n_testing_users = len(unique_test_user_ids)
    user_rating_counts = df_test.groupby("user_id")["rating"].count()
    users_with_one_rating = user_rating_counts[user_rating_counts == 1].index
    df_test = df_test[~df_test["user_id"].isin(users_with_one_rating)]
    df_test_seen, df_test_unseen = train_test_split(df_test, test_size=0.25, stratify=df_test["user_id"])

    # Sparsify Seen
    X_test_seen = sparse.csr_matrix(
        (df_test_seen["rating"], (df_test_seen["user_id"], df_test_seen["item_id"])),
        shape=(n_testing_users, n_items)
    )

    X_test_unseen = sparse.csr_matrix(
        (df_test_unseen["rating"], (df_test_unseen["user_id"], df_test_unseen["item_id"])),
        shape=(n_testing_users, n_items)
    )

    dataset_division = DatasetDivision(
        dataset_name=dataset_name,
        df_train=df_train,
        df_test_seen=df_test_seen,
        df_test_unseen=df_test_unseen,
        X_train=X_train,
        X_test_seen=X_test_seen,
        X_test_unseen=X_test_unseen
    )
    dataset_division.save_to_pickle()
    return dataset_division


def sparse_matrix_to_df(sparse_matrix: csr_matrix) -> pd.DataFrame:
    coo = sparse_matrix.tocoo()
    return pd.DataFrame({
        "user_id": coo.row,
        "item_id": coo.col,
        "rating": coo.data
    })


def df_to_sparse_matrix(df: pd.DataFrame) -> csr_matrix:
    df["item_id"], unique_items = pd.factorize(df["item_id"])
    n_items = len(unique_items)
    df["user_id"], unique_users = pd.factorize(df["user_id"])
    n_users = len(unique_users)
    sparse_matrix = csr_matrix(
        (df["rating"], (df["user_id"], df["item_id"])),
        shape=(n_users, n_items)
    )
    return sparse_matrix
