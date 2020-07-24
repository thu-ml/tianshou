import numpy as np
from typing import Union, Optional


class SegmentTree:
    """Implementation of Segment Tree. The procedure is as follows:

    1. Find out the smallest n which safisfies ``size <= 2^n``, and let \
    ``bound = 2^n``. This is to ensure that all leaf nodes are in the same \
    depth inside the segment tree.
    2. Store the original value to leaf nodes in ``[bound:bound * 2]``, and \
    the union of elementary to internal nodes in ``[1:bound]``. The internal \
    node follows the rule: \
    ``value[i] = operation(value[i * 2], value[i * 2 + 1])``.
    3. Update a node takes O(log(bound)) time complexity.
    4. Query an interval [l, r] with the default operation takes O(log(bound))

    :param int size: the size of segment tree.
    :param str operation: the operation of segment tree. Choose one of "sum",
        "min" and "max", defaults to "sum".
    """

    def __init__(self, size: int,
                 operation: str = "sum") -> None:
        bound = 1
        while bound < size:
            bound <<= 1
        self._bound = bound
        assert operation in ["sum", "min", "max"], \
            f"Unknown operation {operation}."
        (self._op, self._init_value) = {
            "sum": (np.sum, 0.),
            "min": (np.min, np.inf),
            "max": (np.max, -np.inf),
        }[operation]
        self._value = np.zeros([bound << 1]) + self._init_value

    def __getitem__(self, index: int) -> float:
        """Return self[index]"""
        assert isinstance(index, int) and 0 <= index < self._bound
        return self._value[index + self._bound]

    def __setitem__(self, index: Union[int, np.ndarray],
                    value: Union[float, np.ndarray]) -> None:
        """Insert or overwrite a (or some) value(s) in this segment tree."""
        if isinstance(index, int) and isinstance(value, float):
            index, value = np.array([index]), np.array([value])
        assert isinstance(index, np.ndarray) and isinstance(value, np.ndarray)
        assert ((0 <= index) & (index < self._bound)).all()
        index += self._bound
        self._value[index] = value
        while index[0] > 1:
            index >>= 1
            self._value[index] = self._op(
                [self._value[index << 1], self._value[index << 1 | 1]], axis=0)

    def reduce(self, start: Optional[int] = 0,
               end: Optional[int] = None) -> float:
        """Return operation(value[start:end])."""
        if start == 0 and end is None:
            return self._value[1]
        if end is None:
            end = self._bound
        if end < 0:
            end += self._bound
        start, end = start + self._bound - 1, end + self._bound
        result = self._init_value
        while start ^ end ^ 1 != 0:
            if start % 2 == 0:
                result = self._op([result, self._value[start ^ 1]])
            if end % 2 == 1:
                result = self._op([result, self._value[end ^ 1]])
            start, end = start >> 1, end >> 1
        return result

    def get_prefix_sum_idx(
            self, value: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Return the index ``i`` which satisfies
        ``sum(value[:i]) <= value < sum(value[:i + 1])``.
        """
        assert self._op is np.sum
        single = False
        if not isinstance(value, np.ndarray):
            value = np.array([value])
            single = True
        assert (value <= self._value[1]).all()
        index = np.ones(value.shape, dtype=np.int)
        while index[0] < self._bound:
            index <<= 1
            direct = self._value[index] <= value
            value -= self._value[index] * direct
            index += direct
        index -= self._bound
        return index.item() if single else index
