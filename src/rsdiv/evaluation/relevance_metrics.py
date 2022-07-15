from abc import ABC, abstractmethod
from typing import Literal, Tuple

import numpy as np


class RelevanceMetricsBase(ABC):
    @staticmethod
    @abstractmethod
    def get_similarity_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        raise NotImplementedError("get_similarity_scores must be implemented.")

    @classmethod
    def _get_partition(
        cls,
        query: np.ndarray,
        candidates: np.ndarray,
        kind: Literal["most", "least"],
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert k > 0, f"top should be larger than 0."
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        scores = cls.get_similarity_scores(query, candidates)
        if kind == "least":
            partition_point = k - 1
            slice_ = slice(None, k)
        else:
            partition_point = -k
            slice_ = slice(-k, None)
        indices = np.argpartition(scores, partition_point, axis=1)[:, slice_]
        scores = np.take_along_axis(scores, indices, axis=1)
        return np.squeeze(indices), np.squeeze(scores)

    @classmethod
    def most_similar(
        cls, query: np.ndarray, candidates: np.ndarray, top: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        return cls._get_partition(query, candidates, "most", top)

    @classmethod
    def least_similar(
        cls, query: np.ndarray, candidates: np.ndarray, top: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        return cls._get_partition(query, candidates, "least", top)


class CosineRelevanceMetric(RelevanceMetricsBase):
    @staticmethod
    def get_similarity_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.squeeze(
                query
                @ candidates.T
                / np.linalg.norm(
                    query, axis=0 if len(query.shape) == 1 else 1, keepdims=True
                )
                / np.linalg.norm(
                    candidates,
                    axis=0 if len(candidates.shape) == 1 else 1,
                    keepdims=True,
                ).T
            )
        )


class InnerProductRelevanceMetric(RelevanceMetricsBase):
    @staticmethod
    def get_similarity_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        return np.asarray(query @ candidates.T)
