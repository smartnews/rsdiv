from typing import Optional
import numpy as np
import pandas as pd


class RankProduct:
    """Rank product is a biologically motivated test to combine the lists to a comprehensive rank."""

    def __init__(self, multi_scores: pd.DataFrame) -> None:
        """Constructor of rank product calculator.

        Args:
            multi_scores (pd.DataFrame):
                The multi-task scores, each row corresponds to an item.
                For the values, the larger the more preferable.
        """
        self.multi_scores = multi_scores

    def get_rp_values(self, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Get the list of rank product values for each item.

        Args:
            weights (Optional[np.ndarray], optional):
                Weights for each column. The larger the more important.
                Defaults to None. If not given, a uniformed weight will be set.

        Returns:
            np.ndarray:
                The rank product scores for each item, the larger the more preferable.
                Laplacian smoothing is applied to avoid zeros.
        """
        if weights is None:
            weights = np.ones(len(self.multi_scores.columns))

        ranks = self.multi_scores.apply(lambda x: x.rank(method="min")).apply(lambda x: x**weights)
        rp_scores = np.prod(ranks, axis=1) ** (1.0 / weights.sum())

        return rp_scores.values
