from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import AUC_at_k, precision_at_k
from scipy import sparse as sps

from .base import BaseRecommender


class IALSRecommender(BaseRecommender):
    def __init__(
        self,
        df_interaction: pd.DataFrame,
        items: pd.DataFrame,
        test_size: Union[float, int],
        random_split: bool = False,
        factors: int = 300,
        regularization: float = 0.03,
        alpha: float = 0.6,
        iterations: int = 10,
        random_state: Optional[int] = 42,
        toppop_mask: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(df_interaction, items, test_size, random_split, toppop_mask)
        self.ials = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            random_state=random_state,
            calculate_training_loss=True,
        )
        self.train_mat = self.bm25(self.train_mat)

    def bm25(self, X: sps.coo_matrix, K1: int = 100, B: float = 0.8) -> sps.csr_matrix:
        r"""Weighs each col of a sparse matrix X by BM25 weighting.
        Taken from `nearest_neighbours.py of implicit <https://github.com/benfred/implicit/blob/main/implicit/nearest_neighbours.py>`_
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

    def recommend(self, user_ids: List[int]) -> tuple:
        ids, scores = self.ials.recommend(
            user_ids, self.train_mat[user_ids], N=self.n_items
        )
        id_list: List = [list(id) for id in ids]
        return (id_list, scores)

    def recommend_single(self, user_token: str, top_k: int = 100) -> np.ndarray:
        if user_token in self.userDict:
            user_id = self.userDict[user_token]
            ids, _ = self.ials.recommend(user_id, self.train_mat[user_id], N=top_k)
            return np.asarray(ids)
        else:
            return self.toppop

    def auc_score(self, top_k: int = 100) -> float:
        return float(AUC_at_k(self.ials, self.train_mat, self.test_mat, K=top_k))

    def precision_at_top_k(self, top_k: int = 100) -> float:
        return float(precision_at_k(self.ials, self.train_mat, self.test_mat, K=top_k))

    def get_item_factors(self) -> np.ndarray:
        return np.asarray(self.ials.item_factors)

    def get_user_factors(self) -> np.ndarray:
        return np.asarray(self.ials.user_factors)

    def mask_items(self, keep_row: np.ndarray) -> None:
        mask = np.ones(self.ials.item_factors.shape[0], dtype=bool)
        mask[keep_row] = False
        self.ials.item_factors[mask] = 0

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

    def rerank_preprocess(
        self, user_id: int, truncate_at: int, category_col: str, embedding_col: str
    ) -> Tuple:
        item_clean = self.clean_items()
        category = item_clean[category_col].to_list()
        embedding = np.stack(item_clean[embedding_col])

        org_rank = self.recommend([user_id])[0][0]
        org_scores = self.recommend([user_id])[1][0]

        relevance_scores = org_scores[:truncate_at]
        org_select = org_rank[:truncate_at]
        similarity_scores = embedding[org_select]
        similarity_matrix = similarity_scores @ similarity_scores.T

        return (org_select, category, relevance_scores, similarity_matrix)
