from typing import Optional, Sequence

import numpy as np

from .base import BaseReranker

norm = np.linalg.norm


class SlidingSpectrumDecomposition(BaseReranker):
    def __init__(self, gamma: float):
        self.gamma = gamma

    def rerank(
        self,
        quality_scores: np.ndarray,
        *,
        similarity_scores: None = None,
        embeddings: np.ndarray,
        k: int,
    ) -> Sequence[int]:
        selection = np.argmax(quality_scores).item()
        ret = [selection]
        volume = self.gamma * norm(embeddings[selection])
        for _ in range(k - 1):
            selected_emb = embeddings[selection]
            selected_emb /= norm(selected_emb)
            embeddings -= np.outer(embeddings @ selected_emb, selected_emb)
            norms = norm(embeddings, axis=1)
            norms *= volume
            scores = norms + quality_scores
            scores[ret] = -np.inf
            selection = np.argmax(scores)
            ret.append(selection)
            volume = norms[selection]
        return ret
