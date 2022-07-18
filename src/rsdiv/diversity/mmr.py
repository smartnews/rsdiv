from typing import Sequence

import numpy as np
import numpy.ma as ma

from .base import BaseReranker


class MaximalMarginalRelevance(BaseReranker):
    def __init__(self, lbd: float):
        self.lbd = lbd

    def rerank(
        self, quality_scores: np.ndarray, similarity_scores: np.ndarray, k: int
    ) -> Sequence[int]:
        n = quality_scores.shape[0]
        k = min(k, n)
        new_selection = np.argmax(quality_scores)
        selected_ind = [new_selection]
        similarity_scores = ma.array(similarity_scores, mask=True)

        similarity_scores.mask[:, new_selection] = False
        similarity_scores[new_selection, new_selection] = ma.masked

        quality_scores = ma.array(quality_scores)
        quality_scores[new_selection] = ma.masked

        for _ in range(k - 1):
            scores = self.lbd * quality_scores - (1.0 - self.lbd) * np.max(
                similarity_scores, axis=1
            )
            new_selection = np.argmax(scores).item()
            quality_scores[new_selection] = ma.masked

            similarity_scores.mask[:, new_selection] = False
            similarity_scores[new_selection, :] = ma.masked
            similarity_scores[selected_ind, new_selection] = ma.masked

            selected_ind.append(new_selection)
        return selected_ind
