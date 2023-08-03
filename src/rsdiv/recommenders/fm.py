from typing import Optional, Union
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from scipy import sparse as sps
from .base import BaseRecommender


class FMRecommender(BaseRecommender):
    """
    FM recommender based on `LightFM`.

    Args:
        interaction (pd.DataFrame): user-item interaction matrix.
        items (Optional[pd.DataFrame]): item side information.
        no_components (int): the dimensions of user/item embeddings.
        item_alpha (float): L2 penalty on item features.
        user_alpha (float): L2 penalty on user features.
    """

    def __init__(
        self,
        interaction,
        items: Optional[sps.csr_matrix],
        test_size: Union[float, int] = 0.3,
        random_split: bool = True,
        no_components: int = 10,
        item_alpha: float = 0,
        user_alpha: float = 0,
        loss: str = "bpr",
        epochs: int = 30,
    ):
        super().__init__(interaction, items, test_size, random_split)
        self.epochs = epochs
        self.fm = LightFM(
            no_components=no_components,
            item_alpha=item_alpha,
            user_alpha=user_alpha,
            loss=loss,
            random_state=42,
        )

    def _fit(self):
        self.fm.fit(self.train_mat, epochs=self.epochs)

    def predict(
        self,
        user_ids,
        item_ids,
        user_features: Optional[sps.csr_matrix] = None,
        item_features: Optional[sps.csr_matrix] = None,
    ):
        return self.fm.predict(user_ids, item_ids, user_features, item_features)

    def precision_at_top_k(self, top_k: int = 5):
        return precision_at_k(self.fm, self.test_mat, k=top_k).mean()
