from typing import List, Optional, Union

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy import sparse as sps

from .base import BaseRecommender


class IALSRecommender(BaseRecommender):
    def __init__(
        self,
        df_interaction: pd.DataFrame,
        items: pd.DataFrame,
        test_size: Union[float, int],
        random_split: bool = False,
        factors: int = 64,
        regularization: float = 0.05,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(df_interaction, items, test_size, random_split)
        self.ials = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            random_state=random_state,
        )
        self.train_mat = self.bm25(self.train_mat)

    def bm25(self, X: sps.coo_matrix, K1: int = 100, B: float = 0.8) -> sps.csr_matrix:
        """Weighs each col of a sparse matrix X  by BM25 weighting.
        - `Taken from nearest_neighbours.py of implicit
          <https://github.com/benfred/implicit/blob/main/implicit/nearest_neighbours.py>`_
        """

        X = X.T
        N = float(X.shape[0])
        idf = np.log(N) - np.log1p(np.bincount(X.col))
        row_sums = np.ravel(X.sum(axis=1))
        average_length = row_sums.mean()
        length_norm = (1.0 - B) + B * row_sums / average_length
        X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
        return X.T.tocsr()

    def _fit(self) -> None:
        self.ials.fit(2 * self.train_mat)

    def recommend(self, user_ids: np.ndarray) -> tuple:
        ids, scores = self.ials.recommend(
            user_ids, self.train_mat[user_ids], N=self.n_items
        )
        id_list: List = [list(id) for id in ids]
        return (id_list, scores)

    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        user_features: Optional[sps.csr_matrix] = None,
        item_features: Optional[sps.csr_matrix] = None,
    ) -> np.ndarray:
        user_factors = self.ials.user_factors[user_ids]
        item_factors = self.ials.item_factors[item_ids]
        predict_array: np.ndarray = np.asarray(
            [user @ item for user, item in zip(user_factors, item_factors)]
        )
        return predict_array
