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
    def embedding_norm(cls, org: str) -> np.ndarray:
        """Normalize a given vector.

        Args:
            org (str): target string to generate the embedding.

        Returns:
            np.ndarray: normalized vector.
        """
        vector: np.ndarray = cls.embedding_single(org)
        norm_val: float = np.linalg.norm(vector)
        if norm_val:
            embed: np.ndarray = vector / norm_val
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
        return np.mean([cls.embedding_norm(item) for item in org], axis=0)
