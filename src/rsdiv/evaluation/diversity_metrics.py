import numpy as np


def gini_coefficient(x: np.ndarray, sort: bool = True) -> float:
    if sort:
        x = np.sort(x)[::-1]
    count: int = x.shape[0]
    b: float = x @ np.arange(1, count + 1)
    b /= x.sum() * count
    return 1 - 2 * b + 1 / count


def effective_catalog_size(x: np.ndarray, sort: bool = True) -> float:
    pmf = x / x.sum()
    if sort:
        pmf.sort()
        pmf = pmf[::-1]
    ecs: float = pmf @ np.arange(1, x.shape[0] + 1) * 2 - 1
    return ecs
