from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        quality_scores: np.ndarray,
        k: int,
        *,
        similarity_scores: np.ndarray,
        embeddings: np.ndarray,
    ) -> Sequence[int]:
        raise NotImplementedError("Rerank method not implemented!")
