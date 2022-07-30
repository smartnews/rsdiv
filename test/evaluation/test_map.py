from collections import defaultdict
from typing import Dict, Sequence

import numpy as np
from numpy.testing import assert_allclose
from pytest import mark, skip

from rsdiv.evaluation import RankingMetrics


class TestMAP:
    rng = np.random.default_rng(42)

    @mark.parametrize("user_size", [1, 10, 50, 100, 500])
    @mark.parametrize("recommend_size", [10, 50, 100, 500])
    @mark.parametrize("position", [10, 50, 100, 500])
    def test_range(self, user_size: int, recommend_size: int, position: int) -> None:
        if position > recommend_size:
            skip()
        recommend_relevance = self.rng.integers(
            0,
            2,
            (user_size, recommend_size) if user_size != 1 else recommend_size,
            dtype=int,
        )
        assert (
            0
            <= RankingMetrics.mean_average_precision(recommend_relevance, position)
            <= 1
        )

    @mark.parametrize("user_size", [1, 10, 50, 100, 500])
    @mark.parametrize("recommend_size", [10, 50, 100, 500])
    @mark.parametrize("position", [10, 50, 100, 500])
    def test_all_ones(self, user_size: int, recommend_size: int, position: int) -> None:
        if position > recommend_size:
            skip()
        recommend_relevance = np.ones(
            (user_size, recommend_size) if user_size != 1 else recommend_size,
            dtype=int,
        )
        assert_allclose(
            RankingMetrics.mean_average_precision(recommend_relevance, position), 1
        )

    @mark.parametrize("user_size", [1, 10, 50, 100, 500])
    @mark.parametrize("recommend_size", [10, 50, 100, 500])
    @mark.parametrize("position", [10, 50, 100, 500])
    def test_all_zeros(
        self, user_size: int, recommend_size: int, position: int
    ) -> None:
        if position > recommend_size:
            skip()
        recommend_relevance = np.zeros(
            (user_size, recommend_size) if user_size != 1 else recommend_size,
            dtype=int,
        )
        assert_allclose(
            RankingMetrics.mean_average_precision(recommend_relevance, position), 0
        )

    @mark.parametrize("user_size", [10, 50, 100, 500])
    @mark.parametrize("recommend_size", [10, 50, 100, 500])
    def test_random_swap(self, user_size: int, recommend_size: int) -> None:
        recommend_relevance = self.rng.integers(
            0,
            2,
            (user_size, recommend_size) if user_size != 1 else recommend_size,
            dtype=int,
        )
        row = self.rng.integers(0, (user_size, recommend_size), 1, dtype=int)
        old_map = RankingMetrics.mean_average_precision(
            recommend_relevance, recommend_size
        )
        arg_ones_ind = np.argwhere(recommend_relevance[row].flatten() == 1).flatten()
        arg_zeros_ind = np.argwhere(recommend_relevance[row].flatten() == 0).flatten()

        first_one = arg_ones_ind[0]
        first_zero = arg_zeros_ind[0]

        recommend_relevance[row, [first_zero, first_one]] = recommend_relevance[
            row, [first_one, first_zero]
        ]
        new_map = RankingMetrics.mean_average_precision(
            recommend_relevance, recommend_size
        )

        if first_one < first_zero:
            assert new_map < old_map
        else:
            assert new_map > old_map
