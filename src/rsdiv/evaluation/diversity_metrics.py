from itertools import chain
from typing import Hashable, Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy


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

    @classmethod
    def shannon_index(
        cls,
        items: Union[Iterable[Hashable], Iterable[Sequence[Hashable]]],
        base: Optional[float] = None,
    ) -> float:
        ent: float = entropy(cls._get_histogram(items), base=base)
        return ent

    @classmethod
    def get_lorenz_curve(
        cls, items: Union[Iterable[Hashable], Iterable[Sequence[Hashable]]]
    ) -> None:
        categories_histogram = cls._get_histogram(items)[::-1]
        scaled_prefix_sum = categories_histogram.cumsum() / categories_histogram.sum()
        lorenz_curve: np.ndarray = np.insert(scaled_prefix_sum, 0, 0)
        _, ax = plt.subplots()
        x_axis: np.ndarray = np.linspace(0.0, 1.0, lorenz_curve.size)
        ax.fill_between(x_axis, 0, lorenz_curve, alpha=0.3)
        ax.fill_between(x_axis, lorenz_curve, x_axis, alpha=0.3)
        plt.plot(x_axis, lorenz_curve)
        plt.savefig("Lorenz.png")
