from typing import List, Tuple

import numpy as np


class PMF:
    """Probability mass function for re-ranking matrix."""

    def __init__(self, rank_lists: List[List[str]], top_k: int) -> None:
        """Constructor for PMF instance.

        Args:
            rank_lists (List[List[str]]): Target ranking lists.
            top_k (int): Re-ranking for `top_k` items to obtain a “better” ordering.
        """
        self.rank_lists = rank_lists
        self.top_k = top_k
        self.initialize()

    @staticmethod
    def get_elements(rank_lists: List[List[str]]) -> List:
        """Get the list for unique elements.

        Args:
            rank_lists (List[List[str]]): Target ranking lists.

        Returns:
            List: List of unique elements from target.
        """
        return list(set(sum(rank_lists, [])))

    def initialize_uniform(self) -> Tuple[np.ndarray, List]:
        """Initialize the PMF matrix with uniform distribution.

        Returns:
            Tuple[np.ndarray, List]: Generated PMF and list of unique elements.
        """
        unique_elements = self.get_elements(self.rank_lists)
        num_elements: int = len(unique_elements)
        pmf: np.ndarray = np.ones([num_elements, self.top_k])
        pmf /= num_elements
        return pmf, unique_elements

    def initialize(self) -> None:
        """Initialize the PMF instance."""
        self.pmf, self.unique_elements = self.initialize_uniform()
