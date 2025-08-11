from sklearn.decomposition import TruncatedSVD
from train_test_split import DatasetDivision
import pandas as pd
from scipy.stats import linregress
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, vstack
import concurrent.futures as cf


def mean_center_sparse(sparse_matrix: csr_matrix | csc_matrix, means: np.ndarray = None, counts: np.ndarray = None) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Mean-center each <primary-index> of a sparse CSR/CSP matrix (non-zero entries only)."""
    matrix: csr_matrix | csc_matrix = sparse_matrix.copy().astype(float)

    if isinstance(matrix, csc_matrix):
        secondary_axis = 1
        primary_axis = 0
    else:
        secondary_axis = 0
        primary_axis = 1

    if means is None or counts is None:
        # Compute column sums and counts of non-zero entries
        sums = matrix.sum(axis=primary_axis).A1
        counts = np.diff(matrix.indptr)

        with np.errstate(divide='ignore', invalid='ignore'):
            means = np.divide(sums, counts, where=counts != 0)

    # Subtract column mean from each non-zero in that column
    for i in range(matrix.shape[secondary_axis]):
        start = matrix.indptr[i]
        end = matrix.indptr[i + 1]
        if counts[i] > 0:
            matrix.data[start:end] = matrix.data[start:end].astype(float) - means[i]

    return matrix.tocsr(), means, counts


def global_mean_center_sparse(sparse_matrix: csr_matrix, mean: float = None) -> tuple[csr_matrix, float]:
    """Mean-center a sparse CSR matrix globally (all non-zero entries)."""
    matrix = sparse_matrix.copy()

    if mean is None:
        mean = matrix.data.mean()

    matrix.data = matrix.data.astype(float) - mean
    return matrix, mean


def lookup_predicted_ratings(df_unseen: pd.DataFrame, X_pred: np.ndarray) -> pd.Series:
    predicted_ratings = df_unseen.apply(
        lambda row: X_pred[int(row["user_id"]), int(row["item_id"])], axis=1
    )
    return predicted_ratings


def perform_svd_prediction(dataset: DatasetDivision, use_seen_in_svd: bool = False):
    # Combine seen data if applicable
    if use_seen_in_svd:
        X_train = vstack((dataset.X_train,dataset.X_test_seen))
    else:
        X_train = dataset.X_train

    for center_mechanism in ["global", "none"]:
        # Center data
        if center_mechanism == "user":  # User-centered
            train_dataset, _, _ = mean_center_sparse(X_train)
            test_dataset, _, _ = mean_center_sparse(dataset.X_test_seen)
        elif center_mechanism == "item":  # Item-centered
            train_dataset, item_means, item_counts = mean_center_sparse(X_train.tocsc())
            test_dataset, _, _ = mean_center_sparse(dataset.X_test_seen.tocsc(), item_means, item_counts)
        elif center_mechanism == "global":  # Global mean-centered
            train_dataset, global_mean = global_mean_center_sparse(X_train)
            test_dataset, _ = global_mean_center_sparse(dataset.X_test_seen, global_mean)
        else:
            train_dataset = X_train
            test_dataset = dataset.X_test_seen

        for k in [5, 10, 20, 50, 100]:
            truncated_svd = TruncatedSVD(n_components=k, random_state=42)
            # Fit
            truncated_svd.fit(train_dataset)

            # Predict
            X_reduced = truncated_svd.transform(test_dataset)

            def inverse_transform_batch(data_batch):
                return truncated_svd.inverse_transform(data_batch)

            batch_size = 10000
            n_rows = X_reduced.shape[0]
            batches = [X_reduced[i:min(i + batch_size, n_rows)] for i in range(0, n_rows, batch_size)]
            with cf.ThreadPoolExecutor() as executor:
                test_pred_batches = list(executor.map(inverse_transform_batch, batches))

            test_pred = np.vstack(test_pred_batches)

            # Report
            prediction_name = f"pred_rating_{center_mechanism}_k{k}"
            dataset.df_test_unseen[prediction_name] = lookup_predicted_ratings(
                dataset.df_test_unseen, test_pred
            )
            stats = linregress(dataset.df_test_unseen["rating"], dataset.df_test_unseen[prediction_name])
            print(f"{prediction_name:<23} R² = {stats.rvalue ** 2:.4f}")

            # Checkpoint results
            dataset.save_to_pickle()
