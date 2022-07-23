from typing import Sequence

import numpy as np
import numpy.ma as ma
from soupsieve import select

from .base import BaseReranker


class MaximalMarginalRelevance(BaseReranker):
    def __init__(self, lbd: float):
        self.lbd = lbd

    def rerank(
        self, quality_scores: np.ndarray, similarity_scores: np.ndarray, k: int
    ) -> Sequence[int]:
        selected_ind = list(range(k))  # logic to be implemented here\
        return selected_ind
