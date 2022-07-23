from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        quality_scores: np.ndarray,
        *,
        similarity_scores: Optional[np.ndarray],
        embeddings: Optional[np.ndarray],
        k: int,
    ) -> Sequence[int]:
        raise NotImplementedError("Rerank method not implemented!")
