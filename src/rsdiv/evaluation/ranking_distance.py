from typing import List

import numpy as np


class RankingDistance:
    """Distance between ordered lists."""

    @classmethod
    def common_at_k(
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
    def average_common_at_k(
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
        for num in range(1, truncate_at + 1):
            average_common.append(cls.common_at_k(source_list, target_list, num))
        return float(np.mean(average_common))
