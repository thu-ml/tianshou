from typing import Any, List, Tuple, Union

import numpy as np

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.batch import _alloc_by_keys_diff, _create_value


class SimpleReplayBuffer(ReplayBuffer):
    """:class:`~tianshou.data.SimpleReplayBuffer` stores data generated from interaction \
    between the policy and environment.

    SimpleReplayBuffer adds a sequence of data by directly filling in samples. It \
    ignores sequence information in an episode.

    :param int size: the maximum size of replay buffer.
    """

    def __init__(
        self,
        size: int,
    ) -> None:
        self.maxsize = size
        self._meta: Batch = Batch()
        self.reset()

    def reset(self) -> None:
        """Clear all the data in replay buffer."""
        self._index = self._size = 0

    def unfinished_index(self) -> np.ndarray:
        """Return the index of unfinished episode."""
        return np.arange(self._size)[~self.done[:self._size]]

    def prev(self, index: Union[int, np.ndarray]) -> np.ndarray:
        """Return the input index."""
        return np.array(index)

    def next(self, index: Union[int, np.ndarray]) -> np.ndarray:
        """Return the input index."""
        return np.array(index)

    def update(self, buffer: "ReplayBuffer") -> np.ndarray:
        """Move the data from the given buffer to current buffer.

        Return the updated indices. If update fails, return an empty array.
        """
        if len(buffer) == 0 or self.maxsize == 0:
            return np.array([], int)
        self.add(buffer._meta)
        num_samples = len(buffer)
        to_indices = np.arange(self._index, self._index + num_samples) % self.maxsize
        return to_indices

    def _add_index(self, rew: Union[float, np.ndarray],
                   done: bool) -> Tuple[int, Union[float, np.ndarray], int, int]:
        """Deprecated."""
        raise NotImplementedError

    def add(
        self,
        batch: Batch,
    ) -> Tuple[int, int, int, int]:
        """Add a batch of data into SimpleReplayBuffer.

        :param Batch batch: the input data batch. Its keys must belong to the 7
            reserved keys, and "obs", "act", "rew", "done" is required.

        Return current_index and constants to keep compatability
        """
        # preprocess batch
        b = Batch()
        for key in set(self._reserved_keys).intersection(batch.keys()):
            b.__dict__[key] = batch[key]
        batch = b
        assert set(["obs", "act", "rew", "done"]).issubset(batch.keys())

        num_samples = len(batch)
        ptr = self._index
        indices = np.arange(self._index, self._index + num_samples) % self.maxsize
        self._size = min(self._size + num_samples, self.maxsize)
        self._index = (self._index + num_samples) % self.maxsize
        try:
            self._meta[indices] = batch
        except ValueError:
            stack = False
            batch.rew = batch.rew.astype(float)
            batch.done = batch.done.astype(bool)
            if self._meta.is_empty():
                self._meta = _create_value(  # type: ignore
                    batch, self.maxsize, stack
                )
            else:  # dynamic key pops up in batch
                _alloc_by_keys_diff(self._meta, batch, self.maxsize, stack)
            self._meta[indices] = batch
        return ptr, 0, 0, 0

    def sample_indices(self, batch_size: int) -> np.ndarray:
        """Get a random sample of index with size = batch_size.

        Return all available indices in the buffer if batch_size is 0; return an empty
        numpy array if batch_size < 0 or no available index can be sampled.
        """
        if batch_size > 0:
            return np.random.choice(self._size, batch_size)
        elif batch_size == 0:  # construct current available indices
            return np.concatenate(
                [np.arange(self._index, self._size),
                 np.arange(self._index)]
            )
        else:
            return np.array([], int)

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with size = batch_size.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.
        """
        indices = self.sample_indices(batch_size)
        return self[indices], indices

    def get(
        self,
        index: Union[int, List[int], np.ndarray],
        key: str,
        default_value: Any = None,
    ) -> Union[Batch, np.ndarray]:
        """Return self.key[index] or default_value.

        E.g., if you set ``key = "obs", stack_num = 4, index = t``, it returns the
        stacked result as ``[obs[t-3], obs[t-2], obs[t-1], obs[t]]``.

        :param index: the index for getting stacked data.
        :param str key: the key to get, should be one of the reserved_keys.
        :param default_value: if the given key's data is not found and default_value is
            set, return this default_value.
        """
        if key not in self._meta and default_value is not None:
            return default_value
        val = self._meta[key]
        try:
            return val[index]
        except IndexError as e:
            if not (isinstance(val, Batch) and val.is_empty()):
                raise e  # val != Batch()
            return Batch()

    def __getitem__(self, index: Union[slice, int, List[int], np.ndarray]) -> Batch:
        """Return a data batch: self[index]."""
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indices = self.sample_indices(0) if index == slice(None) \
                else self._indices[:len(self)][index]
        else:
            indices = index
        # raise KeyError first instead of AttributeError,
        # to support np.array([SimpleReplayBuffer()])
        return Batch(
            obs=self.obs[indices],
            act=self.act[indices],
            rew=self.rew[indices],
            done=self.done[indices],
            obs_next=self.obs_next[indices],
            info=self.get(indices, "info", Batch()),
            policy=self.get(indices, "policy", Batch()),
        )
