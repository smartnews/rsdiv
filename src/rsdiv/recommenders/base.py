from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps
from sklearn.model_selection import train_test_split

R = TypeVar("R", bound="BaseRecommender")


class BaseRecommender(metaclass=ABCMeta):
    """Defines a common interface for all recommendation models

    Args:
        df_interaction (pd.DataFrame): user/item interaction for train/test.
        item (pd.DataFrame): side information for items.
        test_size (float|int): indicates whether and how to do the test.
        random_split (bool): random split or not.
        user_features (pd.DataFrame): user feature columns.
        item_features (pd.DataFrame): item feature columns.

    """

    df_interaction: pd.DataFrame
    items: pd.DataFrame
    test_size: Union[float, int]
    random_split: bool
    user_features: Optional[pd.DataFrame]
    item_features: Optional[pd.DataFrame]

    def __init__(
        self,
        df_interaction: pd.DataFrame,
        items: Optional[pd.DataFrame],
        test_size: Union[float, int],
        random_split: bool,
        user_features: Optional[pd.DataFrame] = None,
        item_features: Optional[pd.DataFrame] = None,
        toppop_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.df_interaction, self.userDict, self.itemDict = self.get_interaction(
            df_interaction
        )
        self.n_users, self.n_items = self.df_interaction.max()[:2] + 1
        self.items = items
        self.user_features = user_features
        self.item_features = item_features
        self.test_size = test_size
        self.random_split = random_split
        self.train_mat, self.test_mat = self.process_interaction()
        self.toppop: np.ndarray = self._get_toppop(toppop_mask)

    def get_interaction(
        self, df_interaction: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """The converter for input dataframe

        Args:
            df_interaction(pd.DataFrame): user/item interaction matrix.
                columns should be ["userId", "itemId", "interaction"]

        """

        dataframe = df_interaction.iloc[:, :3]
        dataframe.columns = pd.Index(["userId", "itemId", "interaction"])
        user_cat = pd.Categorical(dataframe["userId"])
        item_cat = pd.Categorical(dataframe["itemId"])
        dataframe["userId"] = user_cat.codes
        dataframe["itemId"] = item_cat.codes
        userDict = dict(enumerate(user_cat.categories))
        itemDict = dict(enumerate(item_cat.categories))
        return dataframe, userDict, itemDict

    def process_interaction(self) -> Tuple[sps.coo_matrix, Optional[sps.coo_matrix]]:
        dataframe = self.df_interaction

        if self.random_split:
            dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)

        self.df_interaction = dataframe

        if self.test_size != 0:
            train, test = train_test_split(
                dataframe, test_size=self.test_size, shuffle=False
            )
            test_mat = sps.coo_matrix(
                (test.interaction, (test.userId, test.itemId)),
                (self.n_users, self.n_items),
                "int32",
            )
        else:
            train = dataframe
            test_mat = None
        train_mat = sps.coo_matrix(
            (train.interaction, (train.userId, train.itemId)),
            (self.n_users, self.n_items),
            "int32",
        )
        return train_mat, test_mat

    def clean_items(self) -> pd.DataFrame:
        invmap = {v: k for k, v in self.itemDict.items()}
        self.items["encodes"] = self.items["itemId"].apply(
            lambda x: self._encode(x, invmap)
        )
        clean_items = self.items.sort_values(by=["encodes"]).dropna()
        clean_items["encodes"] = clean_items["encodes"].apply(int)
        return clean_items

    def _encode(self, id: int, invmap: Dict) -> Optional[int]:
        try:
            code = invmap[id]
            return int(code)
        except:
            return None

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

    def _get_toppop(self, toppop_mask: Optional[np.ndarray]) -> np.ndarray:
        scores = np.asarray(self.train_mat.sum(axis=0)).ravel()
        if toppop_mask:
            mask = np.zeros(scores.shape[0], dtype=bool)
            mask[toppop_mask] = True
            scores[mask] = 0
        return (-scores).argsort().argsort()

    def predict_for_userId(self, user_id: int) -> np.ndarray:
        user_ids: np.ndarray = np.full(self.n_items, user_id)
        item_ids: np.ndarray = np.arange(self.n_items)
        prediction = self.predict(
            user_ids, item_ids, self.user_features, self.item_features
        )
        return prediction

    def predict_for_userId_unseen(self, user_id: int) -> np.ndarray:
        seen = self.df_interaction[self.df_interaction["userId"] == user_id]["itemId"]
        prediction = self.predict_for_userId(user_id)
        prediction[seen] = -np.inf
        return prediction

    def predict_top_n_unseen(self, user_id: int, top_n: int) -> Dict[int, float]:
        prediction = self.predict_for_userId_unseen(user_id)
        argpartition = np.argpartition(-prediction, top_n)
        result_args = argpartition[:top_n]
        return {key: prediction[key] for key in result_args}

    def predict_top_n_item(self, user_id: int, top_n: int) -> pd.DataFrame:
        prediction = self.predict_top_n_unseen(user_id, top_n)
        candidates: pd.DataFrame = pd.DataFrame.from_dict(prediction.items())
        candidates.columns = pd.Index(["itemId", "scores"])
        candidates = candidates.sort_values(
            by="scores", ascending=False, ignore_index=True
        )
        return candidates.merge(self.items, how="left", on="itemId")
