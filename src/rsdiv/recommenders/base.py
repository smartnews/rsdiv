from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
from scipy import sparse as sps
from sklearn.model_selection import train_test_split

R = TypeVar("R", bound="BaseRecommender")


class BaseRecommender(metaclass=ABCMeta):
    df_interaction: pd.DataFrame
    items: pd.DataFrame
    test_size: Optional[float]
    user_features: Optional[pd.DataFrame]
    item_features: Optional[pd.DataFrame]

    def __init__(
        self,
        df_interaction: pd.DataFrame,
        items: pd.DataFrame,
        test_size: Optional[float],
        user_features: Optional[pd.DataFrame] = None,
        item_features: Optional[pd.DataFrame] = None,
    ) -> None:
        self.n_users, self.n_items = df_interaction.max()[:2]
        self.df_interaction = self.get_interaction(df_interaction)
        self.items = items
        self.user_features = user_features
        self.item_features = item_features
        self.test_size = test_size
        self.train_mat, self.test_mat = self.process_interaction()

    def get_interaction(self, df_interaction: pd.DataFrame) -> pd.DataFrame:
        dataframe = df_interaction.iloc[:, :3]
        dataframe.columns = pd.Index(["userId", "itemId", "interaction"])
        return dataframe

    def process_interaction(self) -> Tuple[sps.coo_matrix, sps.coo_matrix]:
        dataframe = self.df_interaction
        train, test = train_test_split(
            dataframe, test_size=self.test_size, random_state=42
        )
        train_mat = sps.coo_matrix(
            (train.interaction, (train.userId - 1, train.itemId - 1)),
            (self.n_users, self.n_items),
            "int32",
        )
        test_mat = sps.coo_matrix(
            (test.interaction, (test.userId - 1, test.itemId - 1)),
            (self.n_users, self.n_items),
            "int32",
        )
        return train_mat, test_mat

    def fit(self: R) -> R:
        self._fit()
        return self

    @abstractmethod
    def _fit(self) -> None:
        raise NotImplementedError("_fit must be implemented.")

    @abstractmethod
    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        user_features: Optional[sps.csr_matrix],
        item_features: Optional[sps.csr_matrix],
    ) -> np.ndarray:
        raise NotImplementedError("predict must be implemented.")

    def predict_for_userId(self, user_id: int) -> np.ndarray:
        user_ids: np.ndarray = np.full(self.n_items, user_id - 1)
        item_ids: np.ndarray = np.arange(self.n_items)
        prediction = self.predict(
            user_ids, item_ids, self.user_features, self.item_features
        )
        return prediction

    def predict_for_userId_unseen(self, user_id: int) -> np.ndarray:
        seen = (
            self.df_interaction[self.df_interaction["userId"] == user_id]["itemId"] - 1
        )
        prediction = self.predict_for_userId(user_id)
        prediction[seen] = -np.inf
        return prediction

    def predict_top_n_unseen(self, user_id: int, top_n: int) -> Dict[int, float]:
        prediction = self.predict_for_userId_unseen(user_id)
        argpartition = np.argpartition(-prediction, top_n)
        result_args = argpartition[:top_n]
        return {key + 1: prediction[key] for key in result_args}

    def predict_top_n_item(self, user_id: int, top_n: int) -> pd.DataFrame:
        prediction = self.predict_top_n_unseen(user_id, top_n)
        candidates: pd.DataFrame = pd.DataFrame.from_dict(prediction.items())
        candidates.columns = pd.Index(["itemId", "scores"])
        candidates = candidates.sort_values(
            by="scores", ascending=False, ignore_index=True
        )
        return candidates.merge(self.items, how="left", on="itemId")
