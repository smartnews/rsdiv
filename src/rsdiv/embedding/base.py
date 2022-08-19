from abc import ABCMeta
from typing import Dict

import numpy as np


class BaseEmbedder(metaclass=ABCMeta):
    """Base embedding API for all pre-trained embedding."""

    MAPPER: Dict[str, np.ndarray]

    @classmethod
    def embedding_single(cls, org: str) -> np.ndarray:
        """Base method to embed the single string.

        Args:
            org (str): target string to be embedded.

        Returns:
            np.ndarray: embedding vector for the given string.
        """
        return cls.MAPPER[org]
