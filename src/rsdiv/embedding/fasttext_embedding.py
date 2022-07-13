import pickle as pkl
import pkgutil
from typing import Dict, List, Optional

import numpy as np

from .base import BaseEmbedder


class FastTextEmbedder(BaseEmbedder):
    EMB_PATH: Optional[bytes] = pkgutil.get_data(
        "rsdiv.embedding", "cc.en.300.movielens.pkl"
    )
    if EMB_PATH:
        MAPPER: Dict[str, np.ndarray] = pkl.loads(EMB_PATH)

    @classmethod
    def _l2_norm(cls, vector: np.ndarray) -> float:
        norm_val: float = np.sqrt(np.sum(vector**2))
        return norm_val

    @classmethod
    def embedding_norm(cls, org: str) -> np.ndarray:
        vector: np.ndarray = cls.embedding_single(org)
        norm_val: float = cls._l2_norm(vector)
        if norm_val:
            embed: np.ndarray = vector * (1.0 / norm_val)
            return embed
        else:
            return vector

    @classmethod
    def embedding_list(cls, org: List[str]) -> np.ndarray:
        emb_list: np.ndarray = np.asarray([cls.embedding_norm(item) for item in org])
        emb_norm: np.ndarray = np.mean(emb_list, axis=0)
        return emb_norm
