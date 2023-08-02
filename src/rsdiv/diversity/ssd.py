from typing import Optional, Sequence

import numpy as np

from .base import BaseReranker

norm = np.linalg.norm


class SlidingSpectrumDecomposition(BaseReranker):
    """Improve the diversity with Sliding Spectrum Decomposition algorithm."""

    def __init__(self, gamma: float):
        assert gamma >= 0, "gamma should be >= 0!"
        self.gamma = gamma

    def _adjust_embeddings(self, embeddings: np.ndarray, selected_emb: np.ndarray):
        selected_norm = norm(selected_emb)
        if selected_norm > 1e-7:  # treat new selection as 0 vector if it's too small
            selected_emb /= selected_norm
            np.subtract(embeddings, np.outer(embeddings @ selected_emb, selected_emb), out=embeddings)

    def rerank(
        self,
        quality_scores: np.ndarray,
        k: int,
        *,
        similarity_scores: Optional[np.ndarray] = None,
        embeddings: np.ndarray,
        inplace: bool = False
    ) -> Sequence[int]:
        assert k > 0, "k must be larger than 0!"
        selection = np.argmax(quality_scores).item()
        ret = [selection]
        volume = self.gamma * norm(embeddings[selection])
        norms = norm(embeddings, axis=1)
        for _ in range(k - 1):
            selected_emb = embeddings[selection]
            self._adjust_embeddings(embeddings, selected_emb)
            norms = norm(embeddings, axis=1)  # update norms after adjusting embeddings
            scaled_norms = norms.copy()
            scaled_norms *= volume
            scores = scaled_norms + quality_scores
            scores[ret] = -np.inf
            selection = np.argmax(scores).item()
            ret.append(selection)
            volume = norms[selection]
        return ret
