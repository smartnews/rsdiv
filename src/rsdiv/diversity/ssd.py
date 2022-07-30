from typing import Optional, Sequence

import numpy as np

from .base import BaseReranker

norm = np.linalg.norm


class SlidingSpectrumDecomposition(BaseReranker):
    def __init__(self, gamma: float):
        assert gamma >= 0, "gamma should be >= 0!"
        self.gamma = gamma

    def rerank(
        self,
        quality_scores: np.ndarray,
        k: int,
        *,
        similarity_scores: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray],
        inplace: bool = False
    ) -> Sequence[int]:
        assert k > 0, "k must be larger than 0!"
        selection = np.argmax(quality_scores).item()
        ret = [selection]
        if not inplace and embeddings:
            embeddings = embeddings.copy()
            volume = self.gamma * norm(embeddings[selection])
            for _ in range(k - 1):
                selected_emb = embeddings[selection]
                selected_emb /= norm(selected_emb)
                embeddings -= np.outer(embeddings @ selected_emb, selected_emb)
                norms = norm(embeddings, axis=1)
                norms *= volume
                scores = norms + quality_scores
                scores[ret] = -np.inf
                selection = int(np.argmax(scores))
                ret.append(selection)
                volume = norms[selection]
        return ret
