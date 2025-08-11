import unittest
from scipy.sparse import csr_matrix, csc_matrix
import numpy as np
import pandas as pd

from svd_helpers import (
    mean_center_sparse,
    global_mean_center_sparse,
    lookup_predicted_ratings
)


class TestSVDHelpers(unittest.TestCase):
    def test__mean_center_sparse__csr_user_averages(self):
        test_matrix = csr_matrix([[1, 0, 3], [0, 4, 0], [5, 0, 6]])
        centered_matrix, means, counts = mean_center_sparse(test_matrix)
        assert np.allclose(means, np.array([2, 4, 5.5]), 0.001)
        assert np.allclose(counts, np.array([2, 1, 2]), 0.001)
        assert np.allclose(centered_matrix.toarray(), np.array([[-1, 0, 1],[0, 0, 0],[-0.5, 0, 0.5]]), 0.001)

    def test__mean_center_sparse__csc_item_averages(self):
        test_matrix = csc_matrix([[1, 0, 3], [0, 4, 0], [5, 0, 6]])
        centered_matrix, means, counts = mean_center_sparse(test_matrix)
        assert np.allclose(means, np.array([3, 4, 4.5]), 0.001)
        assert np.allclose(counts, np.array([2, 1, 2]), 0.001)
        assert np.allclose(centered_matrix.toarray(), np.array([[-2, 0, -1.5],[0, 0, 0],[2, 0, 1.5]]), 0.001)

    def test_global_mean_center_sparse(self):
        test_matrix = csc_matrix([[1, 0, 3], [0, 4, 0], [5, 0, 6]])
        centered_matrix, mean = global_mean_center_sparse(test_matrix)
        assert np.isclose(mean, 3.8, 0.001)
        assert np.allclose(centered_matrix.toarray(), np.array([[-2.8, 0, -0.8],[0, 0.2, 0],[1.2, 0, 2.2]]), 0.001)

    def test_lookup_predicted_ratings(self):
        unseen_df = pd.DataFrame({
            "user_id": [0, 1, 2],
            "item_id": [2, 1, 0]
        })
        test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        predicted_ratings = lookup_predicted_ratings(unseen_df, test_matrix)
        assert np.allclose(predicted_ratings.values, np.array([3, 5, 7]), 0.001)

if __name__ == "__main__":
    unittest.main()
