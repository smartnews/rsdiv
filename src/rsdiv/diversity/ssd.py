from typing import Optional, Sequence

import numpy as np

from .base import BaseReranker

norm = np.linalg.norm


class SlidingSpectrumDecomposition(BaseReranker):
    """Improve the diversity with Sliding Spectrum Decomposition algorithm."""

    def __init__(self, gamma: float):
        assert gamma >= 0, "gamma should be >= 0!"
        self.gamma = gamma

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
        if not inplace:
            embeddings = embeddings.copy()
        selection = np.argmax(quality_scores).item()
        ret = [selection]
        volume = self.gamma * norm(embeddings[selection])
        for _ in range(k - 1):
            selected_emb = embeddings[selection]
            selected_norm = norm(selected_emb)
            if (
                selected_norm > 1e-7
            ):  # treat new selection as 0 vector if it's too small
                selected_emb /= selected_norm
                embeddings -= np.outer(embeddings @ selected_emb, selected_emb)
            norms = norm(embeddings, axis=1)
            norms *= volume
            scores = norms + quality_scores
            scores[ret] = -np.inf
            selection = np.argmax(scores).item()
            ret.append(selection)
            volume = norms[selection]
        return ret
