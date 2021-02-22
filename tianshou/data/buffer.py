import h5py
import torch
import numpy as np
from numba import njit
from typing import Any, Dict, List, Tuple, Union, Sequence, Optional

from tianshou.data.batch import _create_value
from tianshou.data import Batch, SegmentTree, to_numpy
from tianshou.data.utils.converter import to_hdf5, from_hdf5


def _alloc_by_keys_diff(
    meta: Batch, batch: Batch, size: int, stack: bool = True
) -> None:
    for key in batch.keys():
        if key in meta.keys():
            if isinstance(meta[key], Batch) and isinstance(batch[key], Batch):
                _alloc_by_keys_diff(meta[key], batch[key], size, stack)
            elif isinstance(meta[key], Batch) and meta[key].is_empty():
                meta[key] = _create_value(batch[key], size, stack)
        else:
            meta[key] = _create_value(batch[key], size, stack)


class ReplayBuffer:
    """:class:`~tianshou.data.ReplayBuffer` stores data generated from interaction \
    between the policy and environment.

    ReplayBuffer can be considered as a specialized form (or management) of Batch. It
    stores all the data in a batch with circular-queue style.

    For the example usage of ReplayBuffer, please check out Section Buffer in
    :doc:`/tutorials/concepts`.

    :param int size: the maximum size of replay buffer.
    :param int stack_num: the frame-stack sampling argument, should be greater than or
        equal to 1. Default to 1 (no stacking).
    :param bool ignore_obs_next: whether to store obs_next. Default to False.
    :param bool save_only_last_obs: only save the last obs/obs_next when it has a shape
        of (timestep, ...) because of temporal stacking. Default to False.
    :param bool sample_avail: the parameter indicating sampling only available index
        when using frame-stack sampling method. Default to False.
    """

    _reserved_keys = ("obs", "act", "rew", "done", "obs_next", "info", "policy")

    def __init__(
        self,
        size: int,
        stack_num: int = 1,
        ignore_obs_next: bool = False,
        save_only_last_obs: bool = False,
        sample_avail: bool = False,
        **kwargs: Any,  # otherwise PrioritizedVectorReplayBuffer will cause TypeError
    ) -> None:
        self.options: Dict[str, Any] = {
            "stack_num": stack_num,
            "ignore_obs_next": ignore_obs_next,
            "save_only_last_obs": save_only_last_obs,
            "sample_avail": sample_avail,
        }
        super().__init__()
        self.maxsize = size
        assert stack_num > 0, "stack_num should greater than 0"
        self.stack_num = stack_num
        self._indices = np.arange(size)
        self._save_obs_next = not ignore_obs_next
        self._save_only_last_obs = save_only_last_obs
        self._sample_avail = sample_avail
        self._meta: Batch = Batch()
        self.reset()

    def __len__(self) -> int:
        """Return len(self)."""
        return self._size

    def __repr__(self) -> str:
        """Return str(self)."""
        return self.__class__.__name__ + self._meta.__repr__()[5:]

    def __getattr__(self, key: str) -> Any:
        """Return self.key."""
        try:
            return self._meta[key]
        except KeyError as e:
            raise AttributeError from e

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Unpickling interface.

        We need it because pickling buffer does not work out-of-the-box
        ("buffer.__getattr__" is customized).
        """
        self.__dict__.update(state)
        # compatible with version == 0.3.1's HDF5 data format
        self._indices = np.arange(self.maxsize)

    def __setattr__(self, key: str, value: Any) -> None:
        """Set self.key = value."""
        assert (
            key not in self._reserved_keys
        ), "key '{}' is reserved and cannot be assigned".format(key)
        super().__setattr__(key, value)

    def save_hdf5(self, path: str) -> None:
        """Save replay buffer to HDF5 file."""
        with h5py.File(path, "w") as f:
            to_hdf5(self.__dict__, f)

    @classmethod
    def load_hdf5(cls, path: str, device: Optional[str] = None) -> "ReplayBuffer":
        """Load replay buffer from HDF5 file."""
        with h5py.File(path, "r") as f:
            buf = cls.__new__(cls)
            buf.__setstate__(from_hdf5(f, device=device))
        return buf

    def reset(self) -> None:
        """Clear all the data in replay buffer and episode statistics."""
        self.last_index = np.array([0])
        self._index = self._size = 0
        self._ep_rew, self._ep_len, self._ep_idx = 0.0, 0, 0

    def set_batch(self, batch: Batch) -> None:
        """Manually choose the batch you want the ReplayBuffer to manage."""
        assert len(batch) == self.maxsize and set(batch.keys()).issubset(
            self._reserved_keys
        ), "Input batch doesn't meet ReplayBuffer's data form requirement."
        self._meta = batch

    def unfinished_index(self) -> np.ndarray:
        """Return the index of unfinished episode."""
        last = (self._index - 1) % self._size if self._size else 0
        return np.array([last] if not self.done[last] and self._size else [], np.int)

    def prev(self, index: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        """Return the index of previous transition.

        The index won't be modified if it is the beginning of an episode.
        """
        index = (index - 1) % self._size
        end_flag = self.done[index] | (index == self.last_index[0])
        return (index + end_flag) % self._size

    def next(self, index: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        """Return the index of next transition.

        The index won't be modified if it is the end of an episode.
        """
        end_flag = self.done[index] | (index == self.last_index[0])
        return (index + (1 - end_flag)) % self._size

    def update(self, buffer: "ReplayBuffer") -> np.ndarray:
        """Move the data from the given buffer to current buffer.

        Return the updated indices. If update fails, return an empty array.
        """
        if len(buffer) == 0 or self.maxsize == 0:
            return np.array([], np.int)
        stack_num, buffer.stack_num = buffer.stack_num, 1
        from_indices = buffer.sample_index(0)  # get all available indices
        buffer.stack_num = stack_num
        if len(from_indices) == 0:
            return np.array([], np.int)
        to_indices = []
        for _ in range(len(from_indices)):
            to_indices.append(self._index)
            self.last_index[0] = self._index
            self._index = (self._index + 1) % self.maxsize
            self._size = min(self._size + 1, self.maxsize)
        to_indices = np.array(to_indices)
        if self._meta.is_empty():
            self._meta = _create_value(  # type: ignore
                buffer._meta, self.maxsize, stack=False)
        self._meta[to_indices] = buffer._meta[from_indices]
        return to_indices

    def _add_index(
        self, rew: Union[float, np.ndarray], done: bool
    ) -> Tuple[int, Union[float, np.ndarray], int, int]:
        """Maintain the buffer's state after adding one data batch.

        Return (index_to_be_modified, episode_reward, episode_length,
        episode_start_index).
        """
        self.last_index[0] = ptr = self._index
        self._size = min(self._size + 1, self.maxsize)
        self._index = (self._index + 1) % self.maxsize

        self._ep_rew += rew
        self._ep_len += 1

        if done:
            result = ptr, self._ep_rew, self._ep_len, self._ep_idx
            self._ep_rew, self._ep_len, self._ep_idx = 0.0, 0, self._index
            return result
        else:
            return ptr, self._ep_rew * 0.0, 0, self._ep_idx

    def add(
        self, batch: Batch, buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add a batch of data into replay buffer.

        :param Batch batch: the input data batch. Its keys must belong to the 7
            reserved keys, and "obs", "act", "rew", "done" is required.
        :param buffer_ids: to make consistent with other buffer's add function; if it
            is not None, we assume the input batch's first dimension is always 1.

        Return (current_index, episode_reward, episode_length, episode_start_index). If
        the episode is not finished, the return value of episode_length and
        episode_reward is 0.
        """
        # preprocess batch
        b = Batch()
        for key in set(self._reserved_keys).intersection(batch.keys()):
            b.__dict__[key] = batch[key]
        batch = b
        assert set(["obs", "act", "rew", "done"]).issubset(batch.keys())
        stacked_batch = buffer_ids is not None
        if stacked_batch:
            assert len(batch) == 1
        if self._save_only_last_obs:
            batch.obs = batch.obs[:, -1] if stacked_batch else batch.obs[-1]
        if not self._save_obs_next:
            batch.pop("obs_next", None)
        elif self._save_only_last_obs:
            batch.obs_next = (
                batch.obs_next[:, -1] if stacked_batch else batch.obs_next[-1]
            )
        # get ptr
        if stacked_batch:
            rew, done = batch.rew[0], batch.done[0]
        else:
            rew, done = batch.rew, batch.done
        ptr, ep_rew, ep_len, ep_idx = list(
            map(lambda x: np.array([x]), self._add_index(rew, done))
        )
        try:
            self._meta[ptr] = batch
        except ValueError:
            stack = not stacked_batch
            batch.rew = batch.rew.astype(np.float)
            batch.done = batch.done.astype(np.bool_)
            if self._meta.is_empty():
                self._meta = _create_value(  # type: ignore
                    batch, self.maxsize, stack)
            else:  # dynamic key pops up in batch
                _alloc_by_keys_diff(self._meta, batch, self.maxsize, stack)
            self._meta[ptr] = batch
        return ptr, ep_rew, ep_len, ep_idx

    def sample_index(self, batch_size: int) -> np.ndarray:
        """Get a random sample of index with size = batch_size.

        Return all available indices in the buffer if batch_size is 0; return an empty
        numpy array if batch_size < 0 or no available index can be sampled.
        """
        if self.stack_num == 1 or not self._sample_avail:  # most often case
            if batch_size > 0:
                return np.random.choice(self._size, batch_size)
            elif batch_size == 0:  # construct current available indices
                return np.concatenate(
                    [np.arange(self._index, self._size), np.arange(self._index)]
                )
            else:
                return np.array([], np.int)
        else:
            if batch_size < 0:
                return np.array([], np.int)
            all_indices = prev_indices = np.concatenate(
                [np.arange(self._index, self._size), np.arange(self._index)]
            )
            for _ in range(self.stack_num - 2):
                prev_indices = self.prev(prev_indices)
            all_indices = all_indices[prev_indices != self.prev(prev_indices)]
            if batch_size > 0:
                return np.random.choice(all_indices, batch_size)
            else:
                return all_indices

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with size = batch_size.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.
        """
        indices = self.sample_index(batch_size)
        return self[indices], indices

    def get(
        self,
        index: Union[int, np.integer, np.ndarray],
        key: str,
        default_value: Optional[Any] = None,
        stack_num: Optional[int] = None,
    ) -> Union[Batch, np.ndarray]:
        """Return the stacked result.

        E.g. [s_{t-3}, s_{t-2}, s_{t-1}, s_t], where s is self.key, t is the index.

        :param index: the index for getting stacked data (t in the example).
        :param str key: the key to get, should be one of the reserved_keys.
        :param default_value: if the given key's data is not found and default_value is
            set, return this default_value.
        :param int stack_num: the stack num (4 in the example). Default to
            self.stack_num.
        """
        if key not in self._meta and default_value is not None:
            return default_value
        val = self._meta[key]
        if stack_num is None:
            stack_num = self.stack_num
        try:
            if stack_num == 1:  # the most often case
                return val[index]
            stack: List[Any] = []
            if isinstance(index, list):
                indice = np.array(index)
            else:
                indice = index
            for _ in range(stack_num):
                stack = [val[indice]] + stack
                indice = self.prev(indice)
            if isinstance(val, Batch):
                return Batch.stack(stack, axis=indice.ndim)
            else:
                return np.stack(stack, axis=indice.ndim)
        except IndexError as e:
            if not (isinstance(val, Batch) and val.is_empty()):
                raise e  # val != Batch()
            return Batch()

    def __getitem__(self, index: Union[slice, int, np.integer, np.ndarray]) -> Batch:
        """Return a data batch: self[index].

        If stack_num is larger than 1, return the stacked obs and obs_next with shape
        (batch, len, ...).
        """
        if isinstance(index, slice):  # change slice to np array
            if index == slice(None):  # buffer[:] will get all available data
                index = self.sample_index(0)
            else:
                index = self._indices[:len(self)][index]
        # raise KeyError first instead of AttributeError,
        # to support np.array([ReplayBuffer()])
        obs = self.get(index, "obs")
        if self._save_obs_next:
            obs_next = self.get(index, "obs_next", Batch())
        else:
            obs_next = self.get(self.next(index), "obs", Batch())
        return Batch(
            obs=obs,
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            obs_next=obs_next,
            info=self.get(index, "info", Batch()),
            policy=self.get(index, "policy", Batch()),
        )


class PrioritizedReplayBuffer(ReplayBuffer):
    """Implementation of Prioritized Experience Replay. arXiv:1511.05952.

    :param float alpha: the prioritization exponent.
    :param float beta: the importance sample soft coefficient.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more detailed
        explanation.
    """

    def __init__(self, size: int, alpha: float, beta: float, **kwargs: Any) -> None:
        # will raise KeyError in PrioritizedVectorReplayBuffer
        # super().__init__(size, **kwargs)
        ReplayBuffer.__init__(self, size, **kwargs)
        assert alpha > 0.0 and beta >= 0.0
        self._alpha, self._beta = alpha, beta
        self._max_prio = self._min_prio = 1.0
        # save weight directly in this class instead of self._meta
        self.weight = SegmentTree(size)
        self.__eps = np.finfo(np.float32).eps.item()
        self.options.update(alpha=alpha, beta=beta)

    def init_weight(self, index: Union[int, np.ndarray]) -> None:
        self.weight[index] = self._max_prio ** self._alpha

    def update(self, buffer: ReplayBuffer) -> np.ndarray:
        indices = super().update(buffer)
        self.init_weight(indices)

    def add(
        self, batch: Batch, buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ptr, ep_rew, ep_len, ep_idx = super().add(batch, buffer_ids)
        self.init_weight(ptr)
        return ptr, ep_rew, ep_len, ep_idx

    def sample_index(self, batch_size: int) -> np.ndarray:
        if batch_size > 0 and len(self) > 0:
            scalar = np.random.rand(batch_size) * self.weight.reduce()
            return self.weight.get_prefix_sum_idx(scalar)
        else:
            return super().sample_index(batch_size)

    def get_weight(
        self, index: Union[slice, int, np.integer, np.ndarray]
    ) -> np.ndarray:
        """Get the importance sampling weight.

        The "weight" in the returned Batch is the weight on loss function to de-bias
        the sampling process (some transition tuples are sampled more often so their
        losses are weighted less).
        """
        # important sampling weight calculation
        # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
        # simplified formula: (p_j/p_min)**(-beta)
        return (self.weight[index] / self._min_prio) ** (-self._beta)

    def update_weight(
        self, index: np.ndarray, new_weight: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Update priority weight by index in this buffer.

        :param np.ndarray index: index you want to update weight.
        :param np.ndarray new_weight: new priority weight you want to update.
        """
        weight = np.abs(to_numpy(new_weight)) + self.__eps
        self.weight[index] = weight ** self._alpha
        self._max_prio = max(self._max_prio, weight.max())
        self._min_prio = min(self._min_prio, weight.min())

    def __getitem__(self, index: Union[slice, int, np.integer, np.ndarray]) -> Batch:
        batch = super().__getitem__(index)
        batch.weight = self.get_weight(index)
        return batch


class ReplayBufferManager(ReplayBuffer):
    """ReplayBufferManager contains a list of ReplayBuffer with exactly the same \
    configuration.

    These replay buffers have contiguous memory layout, and the storage space each
    buffer has is a shallow copy of the topmost memory.

    :param buffer_list: a list of ReplayBuffer needed to be handled.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more detailed
        explanation.
    """

    def __init__(self, buffer_list: List[ReplayBuffer]) -> None:
        self.buffer_num = len(buffer_list)
        self.buffers = np.array(buffer_list)
        offset, size = [], 0
        buffer_type = type(self.buffers[0])
        kwargs = self.buffers[0].options
        for buf in self.buffers:
            assert buf._meta.is_empty()
            assert isinstance(buf, buffer_type) and buf.options == kwargs
            offset.append(size)
            size += buf.maxsize
        self._offset = np.array(offset)
        self._extend_offset = np.array(offset + [size])
        self._lengths = np.zeros_like(offset)
        super().__init__(size=size, **kwargs)
        self._compile()

    def _compile(self) -> None:
        lens = last = index = np.array([0])
        offset = np.array([0, 1])
        done = np.array([False, False])
        _prev_index(index, offset, done, last, lens)
        _next_index(index, offset, done, last, lens)

    def __len__(self) -> int:
        return self._lengths.sum()

    def reset(self) -> None:
        self.last_index = self._offset.copy()
        self._lengths = np.zeros_like(self._offset)
        for buf in self.buffers:
            buf.reset()

    def _set_batch_for_children(self) -> None:
        for offset, buf in zip(self._offset, self.buffers):
            buf.set_batch(self._meta[offset:offset + buf.maxsize])

    def set_batch(self, batch: Batch) -> None:
        super().set_batch(batch)
        self._set_batch_for_children()

    def unfinished_index(self) -> np.ndarray:
        return np.concatenate([
            buf.unfinished_index() + offset
            for offset, buf in zip(self._offset, self.buffers)
        ])

    def prev(self, index: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        if isinstance(index, (list, np.ndarray)):
            return _prev_index(np.asarray(index), self._extend_offset,
                               self.done, self.last_index, self._lengths)
        else:
            return _prev_index(np.array([index]), self._extend_offset,
                               self.done, self.last_index, self._lengths)[0]

    def next(self, index: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        if isinstance(index, (list, np.ndarray)):
            return _next_index(np.asarray(index), self._extend_offset,
                               self.done, self.last_index, self._lengths)
        else:
            return _next_index(np.array([index]), self._extend_offset,
                               self.done, self.last_index, self._lengths)[0]

    def update(self, buffer: ReplayBuffer) -> np.ndarray:
        """The ReplayBufferManager cannot be updated by any buffer."""
        raise NotImplementedError

    def add(
        self, batch: Batch, buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add a batch of data into ReplayBufferManager.

        Each of the data's length (first dimension) must equal to the length of
        buffer_ids. By default buffer_ids is [0, 1, ..., buffer_num - 1].

        Return (current_index, episode_reward, episode_length, episode_start_index). If
        the episode is not finished, the return value of episode_length and
        episode_reward is 0.
        """
        # preprocess batch
        b = Batch()
        for key in set(self._reserved_keys).intersection(batch.keys()):
            b.__dict__[key] = batch[key]
        batch = b
        assert set(["obs", "act", "rew", "done"]).issubset(batch.keys())
        if self._save_only_last_obs:
            batch.obs = batch.obs[:, -1]
        if not self._save_obs_next:
            batch.pop("obs_next", None)
        elif self._save_only_last_obs:
            batch.obs_next = batch.obs_next[:, -1]
        # get index
        if buffer_ids is None:
            buffer_ids = np.arange(self.buffer_num)
        ptrs, ep_lens, ep_rews, ep_idxs = [], [], [], []
        for batch_idx, buffer_id in enumerate(buffer_ids):
            ptr, ep_rew, ep_len, ep_idx = self.buffers[buffer_id]._add_index(
                batch.rew[batch_idx], batch.done[batch_idx]
            )
            ptrs.append(ptr + self._offset[buffer_id])
            ep_lens.append(ep_len)
            ep_rews.append(ep_rew)
            ep_idxs.append(ep_idx + self._offset[buffer_id])
            self.last_index[buffer_id] = ptr + self._offset[buffer_id]
            self._lengths[buffer_id] = len(self.buffers[buffer_id])
        ptrs = np.array(ptrs)
        try:
            self._meta[ptrs] = batch
        except ValueError:
            batch.rew = batch.rew.astype(np.float)
            batch.done = batch.done.astype(np.bool_)
            if self._meta.is_empty():
                self._meta = _create_value(  # type: ignore
                    batch, self.maxsize, stack=False)
            else:  # dynamic key pops up in batch
                _alloc_by_keys_diff(self._meta, batch, self.maxsize, False)
            self._set_batch_for_children()
            self._meta[ptrs] = batch
        return ptrs, np.array(ep_rews), np.array(ep_lens), np.array(ep_idxs)

    def sample_index(self, batch_size: int) -> np.ndarray:
        if batch_size < 0:
            return np.array([], np.int)
        if self._sample_avail and self.stack_num > 1:
            all_indices = np.concatenate([
                buf.sample_index(0) + offset
                for offset, buf in zip(self._offset, self.buffers)
            ])
            if batch_size == 0:
                return all_indices
            else:
                return np.random.choice(all_indices, batch_size)
        if batch_size == 0:  # get all available indices
            sample_num = np.zeros(self.buffer_num, np.int)
        else:
            buffer_idx = np.random.choice(
                self.buffer_num, batch_size, p=self._lengths / self._lengths.sum()
            )
            sample_num = np.bincount(buffer_idx, minlength=self.buffer_num)
            # avoid batch_size > 0 and sample_num == 0 -> get child's all data
            sample_num[sample_num == 0] = -1

        return np.concatenate([
            buf.sample_index(bsz) + offset
            for offset, buf, bsz in zip(self._offset, self.buffers, sample_num)
        ])


class PrioritizedReplayBufferManager(PrioritizedReplayBuffer, ReplayBufferManager):
    """PrioritizedReplayBufferManager contains a list of PrioritizedReplayBuffer with \
    exactly the same configuration.

    These replay buffers have contiguous memory layout, and the storage space each
    buffer has is a shallow copy of the topmost memory.

    :param buffer_list: a list of PrioritizedReplayBuffer needed to be handled.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer`,
        :class:`~tianshou.data.ReplayBufferManager`, and
        :class:`~tianshou.data.PrioritizedReplayBuffer` for more detailed explanation.
    """

    def __init__(self, buffer_list: Sequence[PrioritizedReplayBuffer]) -> None:
        ReplayBufferManager.__init__(self, buffer_list)  # type: ignore
        kwargs = buffer_list[0].options
        for buf in buffer_list:
            del buf.weight
        PrioritizedReplayBuffer.__init__(self, self.maxsize, **kwargs)


class VectorReplayBuffer(ReplayBufferManager):
    """VectorReplayBuffer contains n ReplayBuffer with the same size.

    It is used for storing data frame from different environments yet keeping the order
    of time.

    :param int total_size: the total size of VectorReplayBuffer.
    :param int buffer_num: the number of ReplayBuffer it uses, which are under the same
        configuration.

    Other input arguments (stack_num/ignore_obs_next/save_only_last_obs/sample_avail)
    are the same as :class:`~tianshou.data.ReplayBuffer`.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` and
        :class:`~tianshou.data.ReplayBufferManager` for more detailed explanation.
    """

    def __init__(self, total_size: int, buffer_num: int, **kwargs: Any) -> None:
        assert buffer_num > 0
        size = int(np.ceil(total_size / buffer_num))
        buffer_list = [ReplayBuffer(size, **kwargs) for _ in range(buffer_num)]
        super().__init__(buffer_list)


class PrioritizedVectorReplayBuffer(PrioritizedReplayBufferManager):
    """PrioritizedVectorReplayBuffer contains n PrioritizedReplayBuffer with same size.

    It is used for storing data frame from different environments yet keeping the order
    of time.

    :param int total_size: the total size of PrioritizedVectorReplayBuffer.
    :param int buffer_num: the number of PrioritizedReplayBuffer it uses, which are
        under the same configuration.

    Other input arguments (alpha/beta/stack_num/ignore_obs_next/save_only_last_obs/
    sample_avail) are the same as :class:`~tianshou.data.PrioritizedReplayBuffer`.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` and
        :class:`~tianshou.data.PrioritizedReplayBufferManager` for more detailed
        explanation.
    """

    def __init__(self, total_size: int, buffer_num: int, **kwargs: Any) -> None:
        assert buffer_num > 0
        size = int(np.ceil(total_size / buffer_num))
        buffer_list = [
            PrioritizedReplayBuffer(size, **kwargs) for _ in range(buffer_num)
        ]
        super().__init__(buffer_list)


class CachedReplayBuffer(ReplayBufferManager):
    """CachedReplayBuffer contains a given main buffer and n cached buffers, \
    cached_buffer_num * ReplayBuffer(size=max_episode_length).

    The memory layout is: ``| main_buffer | cached_buffers[0] | cached_buffers[1] | ...
    | cached_buffers[cached_buffer_num - 1]``.

    The data is first stored in cached buffers. When an episode is terminated, the data
    will move to the main buffer and the corresponding cached buffer will be reset.

    :param ReplayBuffer main_buffer: the main buffer whose ``.update()`` function
        behaves normally.
    :param int cached_buffer_num: number of ReplayBuffer needs to be created for cached
        buffer.
    :param int max_episode_length: the maximum length of one episode, used in each
        cached buffer's maxsize.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` or
        :class:`~tianshou.data.ReplayBufferManager` for more detailed explanation.
    """

    def __init__(
        self,
        main_buffer: ReplayBuffer,
        cached_buffer_num: int,
        max_episode_length: int,
    ) -> None:
        assert cached_buffer_num > 0 and max_episode_length > 0
        assert type(main_buffer) == ReplayBuffer
        kwargs = main_buffer.options
        buffers = [main_buffer] + [
            ReplayBuffer(max_episode_length, **kwargs)
            for _ in range(cached_buffer_num)
        ]
        super().__init__(buffer_list=buffers)
        self.main_buffer = self.buffers[0]
        self.cached_buffers = self.buffers[1:]
        self.cached_buffer_num = cached_buffer_num

    def add(
        self, batch: Batch, buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add a batch of data into CachedReplayBuffer.

        Each of the data's length (first dimension) must equal to the length of
        buffer_ids. By default the buffer_ids is [0, 1, ..., cached_buffer_num - 1].

        Return (current_index, episode_reward, episode_length, episode_start_index)
        with each of the shape (len(buffer_ids), ...), where (current_index[i],
        episode_reward[i], episode_length[i], episode_start_index[i]) refers to the
        cached_buffer_ids[i]th cached buffer's corresponding episode result.
        """
        if buffer_ids is None:
            buffer_ids = np.arange(1, 1 + self.cached_buffer_num)
        else:  # make sure it is np.ndarray
            buffer_ids = np.asarray(buffer_ids) + 1
        ptr, ep_rew, ep_len, ep_idx = super().add(batch, buffer_ids=buffer_ids)
        # find the terminated episode, move data from cached buf to main buf
        updated_ptr, updated_ep_idx = [], []
        done = batch.done.astype(np.bool_)
        for buffer_idx in buffer_ids[done]:
            index = self.main_buffer.update(self.buffers[buffer_idx])
            if len(index) == 0:  # unsuccessful move, replace with -1
                index = [-1]
            updated_ep_idx.append(index[0])
            updated_ptr.append(index[-1])
            self.buffers[buffer_idx].reset()
            self._lengths[0] = len(self.main_buffer)
            self._lengths[buffer_idx] = 0
            self.last_index[0] = index[-1]
            self.last_index[buffer_idx] = self._offset[buffer_idx]
        ptr[done] = updated_ptr
        ep_idx[done] = updated_ep_idx
        return ptr, ep_rew, ep_len, ep_idx


@njit
def _prev_index(
    index: np.ndarray,
    offset: np.ndarray,
    done: np.ndarray,
    last_index: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    index = index % offset[-1]
    prev_index = np.zeros_like(index)
    for start, end, cur_len, last in zip(offset[:-1], offset[1:], lengths, last_index):
        mask = (start <= index) & (index < end)
        cur_len = max(1, cur_len)
        if np.sum(mask) > 0:
            subind = index[mask]
            subind = (subind - start - 1) % cur_len
            end_flag = done[subind + start] | (subind + start == last)
            prev_index[mask] = (subind + end_flag) % cur_len + start
    return prev_index


@njit
def _next_index(
    index: np.ndarray,
    offset: np.ndarray,
    done: np.ndarray,
    last_index: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    index = index % offset[-1]
    next_index = np.zeros_like(index)
    for start, end, cur_len, last in zip(offset[:-1], offset[1:], lengths, last_index):
        mask = (start <= index) & (index < end)
        cur_len = max(1, cur_len)
        if np.sum(mask) > 0:
            subind = index[mask]
            end_flag = done[subind] | (subind == last)
            next_index[mask] = (subind - start + 1 - end_flag) % cur_len + start
    return next_index
