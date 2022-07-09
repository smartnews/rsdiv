from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
from scipy import sparse as sps
from sklearn.model_selection import train_test_split

R = TypeVar("R", bound="BaseRecommender")


class BaseRecommender(metaclass=ABCMeta):
    df_interaction: pd.DataFrame
    user_features: Optional[pd.DataFrame]
    item_features: Optional[pd.DataFrame]
    test_size: Optional[float]

    def __init__(
        self, df_interaction: pd.DataFrame, test_size: Optional[float]
    ) -> None:
        self.n_users, self.n_items = df_interaction.max()[:2]
        self.df_interaction = df_interaction
        self.test_size = test_size
        self.train_mat, self.test_mat = self.process_interaction()

    def process_interaction(self) -> Tuple[sps.coo_matrix, sps.coo_matrix]:
        dataframe = self.df_interaction.iloc[:, :3]
        dataframe.columns = ["userId", "itemId", "interaction"]
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
