from typing import Callable, Hashable, Sequence, TypeVar

import numpy as np

T = TypeVar("T", bound=Hashable)


class RankingMetrics:
    """Ranking metrics to evaluate the recommended quality: DCG/nDCG/MAP."""

    @staticmethod
    def DCG(recommended_scores: np.ndarray) -> float:
        score: float = np.sum(
            recommended_scores / np.log2(np.arange(2, len(recommended_scores) + 2))
        ).item()
        return score

    @classmethod
    def nDCG(
        cls,
        item2relevance: Callable[[T], float],
        relevant_items: Sequence[T],
        recommended_items: Sequence[T],
        position: int,
        exponential: bool = False,
    ) -> float:
        """Normalized Discounted Cumulative Gain."""
        assert position <= len(relevant_items) and position <= len(recommended_items)

        top_scores = np.array(tuple(map(item2relevance, relevant_items)))
        top_scores.sort()
        top_scores = top_scores[: -position - 1 : -1]

        recommended_scores = np.array(
            tuple(map(item2relevance, recommended_items[:position]))
        )

        if exponential:
            top_scores = 2**top_scores - 1
            recommended_scores = 2**recommended_scores - 1
        idcg = cls.DCG(top_scores)
        dcg = cls.DCG(recommended_scores)
        return dcg / idcg

    @staticmethod
    def mean_average_precision(
        recommended_relevance: np.ndarray, position: int
    ) -> float:
        if len(recommended_relevance.shape) == 1:
            recommended_relevance = recommended_relevance.reshape((1, -1))
        assert (
            position <= recommended_relevance.shape[1]
        ), "position should be smaller than the number of items recommended!"
        recommended_relevance = recommended_relevance[:, :position]
        cum_sum = np.cumsum(recommended_relevance, axis=1)

        ap = cum_sum * recommended_relevance / np.arange(1, position + 1)
        map: float = np.mean(ap).item()
        return map
