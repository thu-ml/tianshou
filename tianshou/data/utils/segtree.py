import numpy as np
from typing import Union, Optional
# from numba import njit


class SegmentTree:
    """Implementation of Segment Tree. The procedure is as follows:

    1. Find out the smallest n which safisfies ``size <= 2^n``, and let \
    ``bound = 2^n``. This is to ensure that all leaf nodes are in the same \
    depth inside the segment tree.
    2. Store the original value to leaf nodes in ``[bound:bound * 2]``, and \
    the union of elementary to internal nodes in ``[1:bound]``. The internal \
    node follows the rule: \
    ``value[i] = operation(value[i * 2], value[i * 2 + 1])``.
    3. Update a (or some) node(s) takes O(log(bound)) time complexity.
    4. Query an interval [l, r] with the default operation takes O(log(bound))

    :param int size: the size of segment tree.
    :param str operation: the operation of segment tree. Choose one of "sum",
        "min" and "max", defaults to "sum".
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
        """Insert or overwrite a (or some) value(s) in this segment tree. The
        duplicate values are handled as numpy array, in other words, we only
        keep the last value and ignore the previous same value.
        """
        if isinstance(index, int) and isinstance(value, float):
            index, value = np.array([index]), np.array([value])
        assert ((0 <= index) & (index < self._size)).all()
        index = index + self._bound
        self._value[index] = value
        while index[0] > 1:
            index //= 2
            self._value[index] = self._op(
                self._value[index * 2], self._value[index * 2 + 1])

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
                result = self._op(result, self._value[start ^ 1])
            if end % 2 == 1:
                result = self._op(result, self._value[end ^ 1])
            start, end = start // 2, end // 2
        return result

    def get_prefix_sum_idx(
            self, value: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Return the index ``i`` which satisfies
        ``sum(value[:i]) <= value < sum(value[:i + 1])``.
        """
        assert self._op is np.add
        single = False
        if not isinstance(value, np.ndarray):
            value = np.array([value])
            single = True
        assert (value <= self._value[1]).all()
        index = np.ones(value.shape, dtype=np.int)
        index = self.__class__._get_prefix_sum_idx(
            index, value, self._bound, self._value)
        return index.item() if single else index

    # numba version, 5x speed up
    # with size=100000 and bsz=64
    # first block (vectorized np): 0.0923 (now) -> 0.0251
    # second block (for-loop): 0.2914 -> 0.0192 (future)
    # @njit
    def _get_prefix_sum_idx(index, scalar, bound, weight):
        while index[0] < bound:
            index *= 2
            direct = weight[index] <= scalar
            scalar -= weight[index] * direct
            index += direct
        # for _, s in enumerate(scalar):
        #     i = 1
        #     while i < bound:
        #         l = i * 2
        #         if weight[l] > s:
        #             i = l
        #         else:
        #             s = s - weight[l]
        #             i = l + 1
        #     index[_] = i
        return index - bound
