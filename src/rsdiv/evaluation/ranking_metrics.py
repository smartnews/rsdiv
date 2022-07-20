from typing import Callable, Hashable, Sequence, TypeVar

import numpy as np

ItemNameType = TypeVar("ItemNameType", bound=Hashable)


class RankingMetrics:
    @staticmethod
    def DCG(recommended_scores: np.ndarray) -> float:
        score: float = np.sum(
            recommended_scores / np.log2(np.arange(2, len(recommended_scores) + 2))
        ).item()
        return score

    @staticmethod
    def nDCG(
        item2relevance: Callable[[ItemNameType], float],
        relevant_items: Sequence[ItemNameType],
        recommended_items: Sequence[ItemNameType],
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
        idcg = RankingMetrics.DCG(top_scores)
        dcg = RankingMetrics.DCG(recommended_scores)
        return dcg / idcg
