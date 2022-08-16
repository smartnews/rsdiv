from abc import ABCMeta
from typing import Dict

import numpy as np


class BaseEmbedder(metaclass=ABCMeta):
    """Base embedding API for all pre-trained embedding."""

    MAPPER: Dict[str, np.ndarray]

    @classmethod
    def embedding_single(cls, org: str) -> np.ndarray:
        return cls.MAPPER[org]
