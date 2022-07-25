from typing import Dict, Sequence

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytest import mark, skip

from rsdiv.evaluation import RankingMetrics


class TestNDCG:
    rng = np.random.default_rng(15213)

    @mark.parametrize("item_size", [10, 100, 500])
    @mark.parametrize("relevant_size", [10, 100, 500])
    @mark.parametrize("position", [10, 100, 500])
    @mark.parametrize("exponential", [True, False])
    def test_equality(
        self, item_size: int, relevant_size: int, position: int, exponential: bool
    ) -> None:
        if position > min(item_size, relevant_size) or relevant_size > item_size:
            skip()
        scores = self.rng.random(item_size)
        sorted_indices = np.argsort(scores)
        relevant_id = sorted_indices[: -relevant_size - 1 : -1]
        assert_allclose(
            RankingMetrics.nDCG(
                scores.__getitem__, relevant_id, relevant_id, position, exponential
            ),
            1,
        )

    @mark.parametrize("item_size", [10, 100, 500])
    @mark.parametrize("relevant_size", [10, 100, 500])
    @mark.parametrize("recommend_size", [10, 100, 500])
    @mark.parametrize("position", [10, 100, 500])
    @mark.parametrize("exponential", [True, False])
    def test_range(
        self,
        item_size: int,
        relevant_size: int,
        recommend_size: int,
        position: int,
        exponential: bool,
    ) -> None:
        if (
            position > min([item_size, relevant_size, recommend_size])
            or not max(relevant_size, recommend_size) <= item_size
        ):
            skip()
        scores = self.rng.random(item_size)
        sorted_indices = np.argsort(scores)
        relevant_ind = sorted_indices[: -relevant_size - 1 : -1]
        self.rng.shuffle(relevant_ind)
        recommend_id = self.rng.choice(
            np.arange(item_size), recommend_size, replace=False
        )
        assert (
            0
            <= RankingMetrics.nDCG(
                scores.__getitem__, relevant_ind, recommend_id, position, exponential
            )
            <= 1
        )

    @mark.parametrize("position", range(1, 4))
    @mark.parametrize("exponential", [True, False])
    def test_no_relevant(
        self,
        position: int,
        exponential: bool,
        relevant_ind: Sequence[int] = (11, 7, 2, 3),
        recommend_ind: Sequence[int] = (4, 9, 8, 2, 1, 90),
    ) -> None:
        scores = self.rng.random(len(relevant_ind))
        item2score: Dict[int, float] = dict(zip(relevant_ind, scores))
        assert_allclose(
            RankingMetrics.nDCG(
                lambda idx: item2score.get(idx, 0),
                relevant_ind,
                recommend_ind,
                position,
                exponential,
            ),
            0,
        )
