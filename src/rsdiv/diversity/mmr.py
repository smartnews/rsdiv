from typing import Optional, Sequence

import numpy as np
import numpy.ma as ma

from .base import BaseReranker


class MaximalMarginalRelevance(BaseReranker):
    """Improve the diversity with Maximal Marginal Relevance algorithm."""

    def __init__(self, lbd: float):
        assert 0 <= lbd <= 1, "lbd should be within the interval [0, 1]!"
        self.lbd = lbd

    def rerank(
        self,
        quality_scores: np.ndarray,
        k: int,
        *,
        similarity_scores: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
    ) -> Sequence[int]:
        assert k > 0, "k must be larger than 0!"
        n = quality_scores.shape[0]
        k = min(k, n)
        new_selection = np.argmax(quality_scores).item()
        selected_ind = [new_selection]
        ma_similarity_scores = ma.array(similarity_scores, mask=True)

        ma_similarity_scores.mask[:, new_selection] = False
        ma_similarity_scores[new_selection, new_selection] = ma.masked

        quality_scores = ma.array(quality_scores)
        quality_scores[new_selection] = ma.masked

        for _ in range(k - 1):
            scores = self.lbd * quality_scores - (1.0 - self.lbd) * np.max(
                ma_similarity_scores, axis=1
            )
            new_selection = np.argmax(scores).item()
            quality_scores[new_selection] = ma.masked

            ma_similarity_scores.mask[:, new_selection] = False
            ma_similarity_scores[new_selection, :] = ma.masked
            ma_similarity_scores[selected_ind, new_selection] = ma.masked

            selected_ind.append(new_selection)
        return selected_ind
