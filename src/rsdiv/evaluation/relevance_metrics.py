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
        return indices.squeeze(), scores.squeeze()

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
    """Relevance metric based on cosine distance."""

    @staticmethod
    def get_similarity_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        eps = 1e-8  # Small constant to avoid division by zero
        return (
            query
            @ candidates.T
            / (
                np.linalg.norm(
                    query, axis=0 if len(query.shape) == 1 else 1, keepdims=True
                )
                + eps
            )
            / (
                np.linalg.norm(
                    candidates,
                    axis=0 if len(candidates.shape) == 1 else 1,
                    keepdims=True,
                ).T
                + eps
            )
        ).squeeze()


class InnerProductRelevanceMetric(RelevanceMetricsBase):
    """Relevance metric based on inner-product."""

    @staticmethod
    def get_similarity_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        return (query @ candidates.T).squeeze()
