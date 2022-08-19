import pickle as pkl
import pkgutil
from typing import Dict, List, Optional

import numpy as np

from .base import BaseEmbedder


class FastTextEmbedder(BaseEmbedder):
    """Embedding extracted from `fastText` for Movielens dataset."""

    EMB_PATH: Optional[bytes] = pkgutil.get_data(
        "rsdiv.embedding", "cc.en.300.movielens.pkl"
    )
    if EMB_PATH:
        MAPPER: Dict[str, np.ndarray] = pkl.loads(EMB_PATH)

    @classmethod
    def _l2_norm(cls, vector: np.ndarray) -> float:
        """Internal method to calculate the normalize of a given vector.

        Args:
            vector (np.ndarray): target vector to be normalized.

        Returns:
            float: the l2 norm value for the given vector.
        """
        norm_val: float = np.sqrt(np.sum(vector**2))
        return norm_val

    @classmethod
    def embedding_norm(cls, org: str) -> np.ndarray:
        """Normalize a given vector.

        Args:
            org (str): target string to generate the embedding.

        Returns:
            np.ndarray: normalized vector.
        """
        vector: np.ndarray = cls.embedding_single(org)
        norm_val: float = cls._l2_norm(vector)
        if norm_val:
            embed: np.ndarray = vector * (1.0 / norm_val)
            return embed
        else:
            return vector

    @classmethod
    def embedding_list(cls, org: List[str]) -> np.ndarray:
        """Normalize a summation of a list of vectors.

        Args:
            org (List[str]): target list of strings.

        Returns:
            np.ndarray: normalized vector.
        """
        emb_list: np.ndarray = np.asarray([cls.embedding_norm(item) for item in org])
        emb_norm: np.ndarray = np.mean(emb_list, axis=0)
        return emb_norm
