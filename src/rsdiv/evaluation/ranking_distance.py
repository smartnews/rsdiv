from typing import List

import numpy as np


class RankingDistance:
    """Distance between ordered lists."""

    @classmethod
    def set_measure(
        cls, source_list: List, target_list: List, truncate_at: int
    ) -> float:
        """Calculate the common elements for two ordered lists.

        Args:
            source_list (List): First ordered list.
            target_list (List): Second ordered list.
            truncate_at (int): Until which position the numbers would be considered.

        Returns:
            float: Ratio for the  number of common elements.
        """
        source_set = set(source_list[:truncate_at])
        target_set = set(target_list[:truncate_at])
        return len(source_set.intersection(target_set)) // truncate_at

    @classmethod
    def naive_set_based_measure(
        cls, source_list: List, target_list: List, truncate_at: int
    ) -> float:
        """Calculate the average common elements for two ordered lists.

        Args:
            source_list (List): First ordered list.
            target_list (List): Second ordered list.
            truncate_at (int): Until which position the numbers would be considered.

        Returns:
            float: The average number of common elements.
        """
        average_common: List = []
        for depth in range(1, truncate_at + 1):
            average_common.append(cls.set_measure(source_list, target_list, depth))
        return float(np.mean(average_common))

    @classmethod
    def rank_biased_overlap(
        cls, source_list: List, target_list: List, decay: float
    ) -> float:
        """Rank Biased Overlap (RBO) based on geometric series.

        Args:
            source_list (List): First ordered list.
            target_list (List): Second ordered list.
            decay (float): The values in geometric series decreases with the increasing depth.

        Returns:
            float: The average overlap for comparing ranked lists.
        """
        rbo: float = 0
        truncate_at = len(source_list)
        for depth in range(1, truncate_at + 1):
            weight = (1 - decay) * decay ** (depth - 1)
            rbo += weight * cls.set_measure(source_list, target_list, depth)
        return rbo
