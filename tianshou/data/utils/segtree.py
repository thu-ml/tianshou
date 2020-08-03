import numpy as np
from typing import Union, Optional
# from numba import njit


# numba version, 5x speed up
# with size=100000 and bsz=64
# first block (vectorized np): 0.0923 (now) -> 0.0251
# second block (for-loop): 0.2914 -> 0.0192 (future)
# @njit
def _get_prefix_sum_idx(value, bound, sums):
    index = np.ones(value.shape, dtype=np.int64)
    while index[0] < bound:
        index *= 2
        direct = sums[index] < value
        value -= sums[index] * direct
        index += direct
    # for _, s in enumerate(value):
    #     i = 1
    #     while i < bound:
    #         l = i * 2
    #         if sums[l] >= s:
    #             i = l
    #         else:
    #             s = s - sums[l]
    #             i = l + 1
    #     index[_] = i
    index -= bound
    return index


class SegmentTree:
    """Implementation of Segment Tree: store an array ``arr`` with size ``n``
    in a segment tree, support value update and fast query of ``min/max/sum``
    for the interval ``[left, right)`` in O(log n) time.

    The detailed procedure is as follows:

    1. Pad the array to have length of power of 2, so that leaf nodes in the\
    segment tree have the same depth.
    2. Store the segment tree in a binary heap.

    :param int size: the size of segment tree.
    :param str operation: the operation of segment tree. Choices are "sum",
        "min" and "max". Default: "sum".
    """

    def __init__(self, size: int,
                 operation: str = 'sum') -> None:
        bound = 1
        while bound < size:
            bound *= 2
        self._size = size
        self._bound = bound
        assert operation in ['sum', 'min', 'max'], \
            f'Unknown operation {operation}.'
        if operation == 'sum':
            self._op, self._init_value = np.add, 0.
        elif operation == 'min':
            self._op, self._init_value = np.minimum, np.inf
        else:
            self._op, self._init_value = np.maximum, -np.inf
        # assert isinstance(self._op, np.ufunc)
        self._value = np.full([bound * 2], self._init_value)

    def __len__(self):
        return self._size

    def __getitem__(self, index: Union[int, np.ndarray]
                    ) -> Union[float, np.ndarray]:
        """Return self[index]"""
        return self._value[index + self._bound]

    def __setitem__(self, index: Union[int, np.ndarray],
                    value: Union[float, np.ndarray]) -> None:
        """Duplicate values in ``index`` are handled by numpy: later index
        overwrites previous ones.

        ::

            >>> a = np.array([1, 2, 3, 4])
            >>> a[[0, 1, 0, 1]] = [4, 5, 6, 7]
            >>> print(a)
            [6 7 3 4]

        """
        # TODO numba njit version
        if isinstance(index, int):
            index = np.array([index])
        assert np.all(0 <= index) and np.all(index < self._size)
        if self._op is np.add:
            assert np.all(0. <= value)
        index = index + self._bound
        self._value[index] = value
        while index[0] > 1:
            index //= 2
            self._value[index] = self._op(
                self._value[index * 2], self._value[index * 2 + 1])

    def reduce(self, start: Optional[int] = 0,
               end: Optional[int] = None) -> float:
        """Return operation(value[start:end])."""
        # TODO numba njit version
        if start == 0 and end is None:
            return self._value[1]
        if end is None:
            end = self._size
        if end < 0:
            end += self._size
        # nodes in (start, end) should be aggregated
        start, end = start + self._bound - 1, end + self._bound
        result = self._init_value
        while end - start > 1:  # (start, end) interval is not empty
            if start % 2 == 0:
                result = self._op(result, self._value[start + 1])
            if end % 2 == 1:
                result = self._op(result, self._value[end - 1])
            start, end = start // 2, end // 2
        return result

    def get_prefix_sum_idx(
            self, value: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Return the minimum index for each ``v`` in ``value`` so that
        ``v <= sums[i]``, where sums[i] = \\sum_{j=0}^{i} arr[j].
        """
        assert self._op is np.add
        assert np.all(value >= 0.) and np.all(value < self._value[1])
        single = False
        if not isinstance(value, np.ndarray):
            value = np.array([value])
            single = True
        index = _get_prefix_sum_idx(value, self._bound, self._value)
        return index.item() if single else index
