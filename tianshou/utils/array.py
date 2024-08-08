from collections.abc import Sequence

import numpy as np


def bisect_left(arr: Sequence[float], x: float) -> float:
    """Assuming arr is sorted, return the largest element el of arr s.t. el < x."""
    el_index = int(np.searchsorted(arr, x, side="left")) - 1
    return arr[el_index]


def bisect_right(arr: Sequence[float], x: float) -> float:
    """Assuming arr is sorted, return the smallest element el of arr s.t. el > x."""
    el_index = int(np.searchsorted(arr, x, side="right"))
    return arr[el_index]
