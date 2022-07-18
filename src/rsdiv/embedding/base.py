from abc import ABCMeta
from typing import Dict

import numpy as np


class BaseEmbedder(metaclass=ABCMeta):
    MAPPER: Dict[str, np.ndarray]

    @classmethod
    def embedding_single(cls, org: str) -> np.ndarray:
        return cls.MAPPER[org]
