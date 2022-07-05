from itertools import chain
from typing import Hashable, Iterable, Sequence, Union

import numpy as np
import pandas as pd


class DiversityMetrics:
    @staticmethod
    def _get_histogram(
        items: Union[Iterable[Hashable], Iterable[Sequence[Hashable]]],
    ) -> np.ndarray:
        if isinstance(next(iter(items)), Sequence):
            items = chain(*items)
        flatten_items = list(items)
        return np.asarray(pd.Series(flatten_items).value_counts())

    @staticmethod
    def _gini_coefficient(categories_histogram: np.ndarray, sort: bool = True) -> float:
        if sort:
            categories_histogram = np.sort(categories_histogram)[::-1]
        count: int = categories_histogram.shape[0]
        area: float = categories_histogram @ np.arange(1, count + 1)
        area /= categories_histogram.sum() * count
        return 1 - 2 * area + 1 / count

    @staticmethod
    def _effective_catalog_size(
        categories_histogram: np.ndarray, sort: bool = True
    ) -> float:
        pmf = categories_histogram / categories_histogram.sum()
        if sort:
            pmf.sort()
            pmf = pmf[::-1]
        ecs: float = pmf @ np.arange(1, categories_histogram.shape[0] + 1) * 2 - 1
        return ecs

    @classmethod
    def gini_coefficient(
        cls,
        items: Union[Iterable[Hashable], Iterable[Sequence[Hashable]]],
    ) -> float:
        return cls._gini_coefficient(cls._get_histogram(items))

    @classmethod
    def effective_catalog_size(
        cls, items: Union[Iterable[Hashable], Iterable[Sequence[Hashable]]]
    ) -> float:
        return cls._effective_catalog_size(cls._get_histogram(items))
