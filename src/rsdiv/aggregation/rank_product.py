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
        num_weights: int = len(self.multi_scores)
        if weights:
            power = weights.reshape(num_weights, -1)
        else:
            power = np.ones([num_weights, 1])
        ranks: np.ndarray = np.argsort(np.argsort(self.multi_scores)) + 1
        ranks = ranks**power
        rp_scores = np.asarray(ranks.prod(axis=0) ** (1.0 / power.sum()))
        return rp_scores
