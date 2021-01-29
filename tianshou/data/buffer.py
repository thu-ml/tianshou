import h5py
import torch
import warnings
import numpy as np
from numbers import Number
from typing import Any, Dict, List, Tuple, Union, Optional

from tianshou.data.batch import _create_value
from tianshou.data import Batch, SegmentTree, to_numpy
from tianshou.data.utils.converter import to_hdf5, from_hdf5


class ReplayBuffer:
    """:class:`~tianshou.data.ReplayBuffer` stores data generated from \
    interaction between the policy and environment.

    ReplayBuffer can be considered as a specialized form (or management) of
    Batch. It stores all the data in a batch with circular-queue style.

    For the example usage of ReplayBuffer, please check out Section Buffer in
    :doc:`/tutorials/concepts`.

    :param int size: the maximum size of replay buffer.
    :param int stack_num: the frame-stack sampling argument, should be greater
        than or equal to 1, defaults to 1 (no stacking).
    :param bool ignore_obs_next: whether to store obs_next, defaults to False.
    :param bool save_only_last_obs: only save the last obs/obs_next when it has
        a shape of (timestep, ...)  because of temporal stacking, defaults to
        False.
    :param bool sample_avail: the parameter indicating sampling only available
        index when using frame-stack sampling method, defaults to False.
    """

    _reserved_keys = ("obs", "act", "rew", "done",
                      "obs_next", "info", "policy")

    def __init__(
        self,
        size: int,
        stack_num: int = 1,
        ignore_obs_next: bool = False,
        save_only_last_obs: bool = False,
        sample_avail: bool = False,
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
        assert key not in self._reserved_keys, (
            "key '{}' is reserved and cannot be assigned".format(key))
        super().__setattr__(key, value)

    def save_hdf5(self, path: str) -> None:
        """Save replay buffer to HDF5 file."""
        with h5py.File(path, "w") as f:
            to_hdf5(self.__dict__, f)

    @classmethod
    def load_hdf5(
        cls, path: str, device: Optional[str] = None
    ) -> "ReplayBuffer":
        """Load replay buffer from HDF5 file."""
        with h5py.File(path, "r") as f:
            buf = cls.__new__(cls)
            buf.__setstate__(from_hdf5(f, device=device))
        return buf

    def reset(self) -> None:
        """Clear all the data in replay buffer and episode statistics."""
        self._index = self._size = 0
        self._episode_length, self._episode_reward = 0, 0.0

    def set_batch(self, batch: Batch) -> None:
        """Manually choose the batch you want the ReplayBuffer to manage."""
        assert len(batch) == self.maxsize and \
            set(batch.keys()).issubset(self._reserved_keys), \
            "Input batch doesn't meet ReplayBuffer's data form requirement."
        self._meta = batch

    def unfinished_index(self) -> np.ndarray:
        """Return the index of unfinished episode."""
        last = (self._index - 1) % self._size if self._size else 0
        return np.array(
            [last] if not self.done[last] and self._size else [], np.int)

    def prev(self, index: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        """Return the index of previous transition.

        The index won't be modified if it is the beginning of an episode.
        """
        index = (index - 1) % self._size
        end_flag = self.done[index] | np.isin(index, self.unfinished_index())
        return (index + end_flag) % self._size

    def next(self, index: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        """Return the index of next transition.

        The index won't be modified if it is the end of an episode.
        """
        end_flag = self.done[index] | np.isin(index, self.unfinished_index())
        return (index + (1 - end_flag)) % self._size

    def update(self, buffer: "ReplayBuffer") -> None:
        """Move the data from the given buffer to current buffer."""
        if len(buffer) == 0 or self.maxsize == 0:
            return
        stack_num, buffer.stack_num = buffer.stack_num, 1
        save_only_last_obs = self._save_only_last_obs
        self._save_only_last_obs = False
        indices = buffer.sample_index(0)  # get all available indices
        for i in indices:
            self.add(**buffer[i])  # type: ignore
        buffer.stack_num = stack_num
        self._save_only_last_obs = save_only_last_obs

    def _buffer_allocator(self, key: List[str], value: Any) -> None:
        """Allocate memory on buffer._meta for new (key, value) pair."""
        data = self._meta
        for k in key[:-1]:
            data = data[k]
        data[key[-1]] = _create_value(value, self.maxsize)

    def _add_to_buffer(self, name: str, inst: Any) -> None:
        try:
            value = self._meta.__dict__[name]
        except KeyError:
            self._buffer_allocator([name], inst)
            value = self._meta[name]
        if isinstance(inst, (torch.Tensor, np.ndarray)):
            if inst.shape != value.shape[1:]:
                raise ValueError(
                    "Cannot add data to a buffer with different shape with key"
                    f" {name}, expect {value.shape[1:]}, given {inst.shape}."
                )
        try:
            value[self._index] = inst
        except KeyError:  # inst is a dict/Batch
            for key in set(inst.keys()).difference(value.keys()):
                self._buffer_allocator([name, key], inst[key])
            self._meta[name][self._index] = inst

    def add(
        self,
        obs: Any,
        act: Any,
        rew: Union[Number, np.number, np.ndarray],
        done: Union[Number, np.number, np.bool_],
        obs_next: Any = None,
        info: Optional[Union[dict, Batch]] = {},
        policy: Optional[Union[dict, Batch]] = {},
        **kwargs: Any,
    ) -> Tuple[int, Union[float, np.ndarray]]:
        """Add a batch of data into replay buffer.

        Return (episode_length, episode_reward) if one episode is terminated,
        otherwise return (0, 0.0).
        """
        assert isinstance(
            info, (dict, Batch)
        ), "You should return a dict in the last argument of env.step()."
        if self._save_only_last_obs:
            obs = obs[-1]
        self._add_to_buffer("obs", obs)
        self._add_to_buffer("act", act)
        # make sure the data type of reward is float instead of int
        # but rew may be np.ndarray, so that we cannot use float(rew)
        rew = rew * 1.0  # type: ignore
        self._add_to_buffer("rew", rew)
        self._add_to_buffer("done", bool(done))  # done should be a bool scalar
        if self._save_obs_next:
            if obs_next is None:
                obs_next = Batch()
            elif self._save_only_last_obs:
                obs_next = obs_next[-1]
            self._add_to_buffer("obs_next", obs_next)
        self._add_to_buffer("info", info)
        self._add_to_buffer("policy", policy)

        if self.maxsize > 0:
            self._size = min(self._size + 1, self.maxsize)
            self._index = (self._index + 1) % self.maxsize
        else:  # TODO: remove this after deleting ListReplayBuffer
            self._size = self._index = self._size + 1

        self._episode_reward += rew
        self._episode_length += 1

        if done:
            result = self._episode_length, self._episode_reward
            self._episode_length, self._episode_reward = 0, 0.0
            return result
        else:
            return 0, self._episode_reward * 0.0

    def sample_index(self, batch_size: int) -> np.ndarray:
        """Get a random sample of index with size = batch_size.

        Return all available indices in the buffer if batch_size is 0; return
        an empty numpy array if batch_size < 0 or no available index can be
        sampled.
        """
        if self.stack_num == 1 or not self._sample_avail:  # most often case
            if batch_size > 0:
                return np.random.choice(self._size, batch_size)
            elif batch_size == 0:  # construct current available indices
                return np.concatenate([
                    np.arange(self._index, self._size),
                    np.arange(self._index)])
            else:
                return np.array([], np.int)
        else:
            if batch_size < 0:
                return np.array([], np.int)
            all_indices = prev_indices = np.concatenate([
                np.arange(self._index, self._size), np.arange(self._index)])
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
        stack_num: Optional[int] = None,
    ) -> Union[Batch, np.ndarray]:
        """Return the stacked result.

        E.g. [s_{t-3}, s_{t-2}, s_{t-1}, s_t], where s is self.key, t is the
        index.
        """
        if stack_num is None:
            stack_num = self.stack_num
        val = self._meta[key]
        try:
            if stack_num == 1:  # the most often case
                return val[index]
            stack: List[Any] = []
            indice = np.asarray(index)
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

    def __getitem__(
        self, index: Union[slice, int, np.integer, np.ndarray]
    ) -> Batch:
        """Return a data batch: self[index].

        If stack_num is larger than 1, return the stacked obs and obs_next with
        shape (batch, len, ...).
        """
        if isinstance(index, slice):  # change slice to np array
            index = self._indices[:len(self)][index]
        # raise KeyError first instead of AttributeError, to support np.array
        obs = self.get(index, "obs")
        if self._save_obs_next:
            obs_next = self.get(index, "obs_next")
        else:
            obs_next = self.get(self.next(index), "obs")
        return Batch(
            obs=obs,
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            obs_next=obs_next,
            info=self.get(index, "info"),
            policy=self.get(index, "policy"),
        )


class ListReplayBuffer(ReplayBuffer):
    """List-based replay buffer.

    The function of :class:`~tianshou.data.ListReplayBuffer` is almost the same
    as :class:`~tianshou.data.ReplayBuffer`. The only difference is that
    :class:`~tianshou.data.ListReplayBuffer` is based on list. Therefore,
    it does not support advanced indexing, which means you cannot sample a
    batch of data out of it. It is typically used for storing data.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more detailed
        explanation.
    """

    def __init__(self, **kwargs: Any) -> None:
        warnings.warn("ListReplayBuffer will be replaced in version 0.4.0.")
        super().__init__(size=0, ignore_obs_next=False, **kwargs)

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        raise NotImplementedError("ListReplayBuffer cannot be sampled!")

    def _add_to_buffer(self, name: str, inst: Any) -> None:
        if self._meta.get(name) is None:
            self._meta.__dict__[name] = []
        self._meta[name].append(inst)

    def reset(self) -> None:
        super().reset()
        for k in self._meta.keys():
            if isinstance(self._meta[k], list):
                self._meta.__dict__[k] = []

    def update(self, buffer: ReplayBuffer) -> None:
        """The ListReplayBuffer cannot be updated by any buffer."""
        raise NotImplementedError


class PrioritizedReplayBuffer(ReplayBuffer):
    """Implementation of Prioritized Experience Replay. arXiv:1511.05952.

    :param float alpha: the prioritization exponent.
    :param float beta: the importance sample soft coefficient.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more detailed
        explanation.
    """

    def __init__(
        self, size: int, alpha: float, beta: float, **kwargs: Any
    ) -> None:
        super().__init__(size, **kwargs)
        assert alpha > 0.0 and beta >= 0.0
        self._alpha, self._beta = alpha, beta
        self._max_prio = self._min_prio = 1.0
        # save weight directly in this class instead of self._meta
        self.weight = SegmentTree(size)
        self.__eps = np.finfo(np.float32).eps.item()

    def add(
        self,
        obs: Any,
        act: Any,
        rew: Union[Number, np.number, np.ndarray],
        done: Union[Number, np.number, np.bool_],
        obs_next: Any = None,
        info: Optional[Union[dict, Batch]] = {},
        policy: Optional[Union[dict, Batch]] = {},
        weight: Optional[Union[Number, np.number]] = None,
        **kwargs: Any,
    ) -> Tuple[int, Union[float, np.ndarray]]:
        if weight is None:
            weight = self._max_prio
        else:
            weight = np.abs(weight)
            self._max_prio = max(self._max_prio, weight)
            self._min_prio = min(self._min_prio, weight)
        self.weight[self._index] = weight ** self._alpha
        return super().add(obs, act, rew, done, obs_next,
                           info, policy, **kwargs)

    def sample_index(self, batch_size: int) -> np.ndarray:
        if batch_size > 0 and self._size > 0:
            scalar = np.random.rand(batch_size) * self.weight.reduce()
            return self.weight.get_prefix_sum_idx(scalar)
        else:
            return super().sample_index(batch_size)

    def get_weight(
        self, index: Union[slice, int, np.integer, np.ndarray]
    ) -> np.ndarray:
        """Get the importance sampling weight.

        The "weight" in the returned Batch is the weight on loss function
        to de-bias the sampling process (some transition tuples are sampled
        more often so their losses are weighted less).
        """
        # important sampling weight calculation
        # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
        # simplified formula: (p_j/p_min)**(-beta)
        return (self.weight[index] / self._min_prio) ** (-self._beta)

    def update_weight(
        self,
        index: np.ndarray,
        new_weight: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Update priority weight by index in this buffer.

        :param np.ndarray index: index you want to update weight.
        :param np.ndarray new_weight: new priority weight you want to update.
        """
        weight = np.abs(to_numpy(new_weight)) + self.__eps
        self.weight[index] = weight ** self._alpha
        self._max_prio = max(self._max_prio, weight.max())
        self._min_prio = min(self._min_prio, weight.min())

    def __getitem__(
        self, index: Union[slice, int, np.integer, np.ndarray]
    ) -> Batch:
        batch = super().__getitem__(index)
        batch.weight = self.get_weight(index)
        return batch


class ReplayBufferManager(ReplayBuffer):
    """ReplayBufferManager contains a list of ReplayBuffer.

    These replay buffers have contiguous memory layout, and the storage space
    each buffer has is a shallow copy of the topmost memory.

    :param int buffer_list: a list of ReplayBuffers needed to be handled.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more detailed
        explanation.
    """

    def __init__(self, buffer_list: List[ReplayBuffer], **kwargs: Any) -> None:
        self.buffer_num = len(buffer_list)
        self.buffers = buffer_list
        self._offset = []
        offset = 0
        for buf in self.buffers:
            # overwrite sub-buffers' _buffer_allocator so that
            # the top buffer can allocate new memory for all sub-buffers
            buf._buffer_allocator = self._buffer_allocator  # type: ignore
            assert buf._meta.is_empty()
            self._offset.append(offset)
            offset += buf.maxsize
        super().__init__(size=offset, **kwargs)

    def __len__(self) -> int:
        return sum([len(buf) for buf in self.buffers])

    def reset(self) -> None:
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
            for offset, buf in zip(self._offset, self.buffers)])

    def prev(self, index: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        index = np.asarray(index) % self.maxsize
        prev_indices = np.zeros_like(index)
        for offset, buf in zip(self._offset, self.buffers):
            mask = (offset <= index) & (index < offset + buf.maxsize)
            if np.any(mask):
                prev_indices[mask] = buf.prev(index[mask] - offset) + offset
        return prev_indices

    def next(self, index: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        index = np.asarray(index) % self.maxsize
        next_indices = np.zeros_like(index)
        for offset, buf in zip(self._offset, self.buffers):
            mask = (offset <= index) & (index < offset + buf.maxsize)
            if np.any(mask):
                next_indices[mask] = buf.next(index[mask] - offset) + offset
        return next_indices

    def update(self, buffer: ReplayBuffer) -> None:
        """The ReplayBufferManager cannot be updated by any buffer."""
        raise NotImplementedError

    def _buffer_allocator(self, key: List[str], value: Any) -> None:
        super()._buffer_allocator(key, value)
        self._set_batch_for_children()

    def add(  # type: ignore
        self,
        obs: Any,
        act: Any,
        rew: np.ndarray,
        done: np.ndarray,
        obs_next: Any = Batch(),
        info: Optional[Batch] = Batch(),
        policy: Optional[Batch] = Batch(),
        buffer_ids: Optional[Union[np.ndarray, List[int]]] = None,
        **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add a batch of data into ReplayBufferManager.

        Each of the data's length (first dimension) must equal to the length of
        buffer_ids. By default buffer_ids is [0, 1, ..., buffer_num - 1].

        Return the array of episode_length and episode_reward with shape
        (len(buffer_ids), ...), where (episode_length[i], episode_reward[i])
        refers to the buffer_ids[i]'s corresponding episode result.
        """
        if buffer_ids is None:
            buffer_ids = np.arange(self.buffer_num)
        # assume each element in buffer_ids is unique
        assert np.bincount(buffer_ids).max() == 1
        batch = Batch(obs=obs, act=act, rew=rew, done=done,
                      obs_next=obs_next, info=info, policy=policy)
        assert len(buffer_ids) == len(batch)
        episode_lengths = []  # (len(buffer_ids),)
        episode_rewards = []  # (len(buffer_ids), ...)
        for batch_idx, buffer_id in enumerate(buffer_ids):
            length, reward = self.buffers[buffer_id].add(**batch[batch_idx])
            episode_lengths.append(length)
            episode_rewards.append(reward)
        return np.stack(episode_lengths), np.stack(episode_rewards)

    def sample_index(self, batch_size: int) -> np.ndarray:
        if batch_size < 0:
            return np.array([], np.int)
        if self._sample_avail and self.stack_num > 1:
            all_indices = np.concatenate([
                buf.sample_index(0) + offset
                for offset, buf in zip(self._offset, self.buffers)])
            if batch_size == 0:
                return all_indices
            else:
                return np.random.choice(all_indices, batch_size)
        if batch_size == 0:  # get all available indices
            sample_num = np.zeros(self.buffer_num, np.int)
        else:
            buffer_lens = np.array([len(buf) for buf in self.buffers])
            buffer_idx = np.random.choice(self.buffer_num, batch_size,
                                          p=buffer_lens / buffer_lens.sum())
            sample_num = np.bincount(buffer_idx, minlength=self.buffer_num)
            # avoid batch_size > 0 and sample_num == 0 -> get child's all data
            sample_num[sample_num == 0] = -1

        return np.concatenate([
            buf.sample_index(bsz) + offset
            for offset, buf, bsz in zip(self._offset, self.buffers, sample_num)
        ])


class CachedReplayBuffer(ReplayBufferManager):
    """CachedReplayBuffer contains a given main buffer and n cached buffers, \
    cached_buffer_num * ReplayBuffer(size=max_episode_length).

    The memory layout is: ``| main_buffer | cached_buffers[0] |
    cached_buffers[1] | ... | cached_buffers[cached_buffer_num - 1]``.

    The data is first stored in cached buffers. When the episode is
    terminated, the data will move to the main buffer and the corresponding
    cached buffer will be reset.

    :param ReplayBuffer main_buffer: the main buffer whose ``.update()``
        function behaves normally.
    :param int cached_buffer_num: number of ReplayBuffer needs to be created
        for cached buffer.
    :param int max_episode_length: the maximum length of one episode, used in
        each cached buffer's maxsize.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` or
        :class:`~tianshou.data.ReplayBufferManager` for more detailed
        explanation.
    """

    def __init__(
        self,
        main_buffer: ReplayBuffer,
        cached_buffer_num: int,
        max_episode_length: int,
    ) -> None:
        assert cached_buffer_num > 0 and max_episode_length > 0
        self._is_prioritized = isinstance(main_buffer, PrioritizedReplayBuffer)
        kwargs = main_buffer.options
        buffers = [main_buffer] + [ReplayBuffer(max_episode_length, **kwargs)
                                   for _ in range(cached_buffer_num)]
        super().__init__(buffer_list=buffers, **kwargs)
        self.main_buffer = self.buffers[0]
        self.cached_buffers = self.buffers[1:]
        self.cached_buffer_num = cached_buffer_num

    def add(  # type: ignore
        self,
        obs: Any,
        act: Any,
        rew: np.ndarray,
        done: np.ndarray,
        obs_next: Any = Batch(),
        info: Optional[Batch] = Batch(),
        policy: Optional[Batch] = Batch(),
        cached_buffer_ids: Optional[Union[np.ndarray, List[int]]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add a batch of data into CachedReplayBuffer.

        Each of the data's length (first dimension) must equal to the length of
        cached_buffer_ids. By default the cached_buffer_ids is [0, 1, ...,
        cached_buffer_num - 1].

        Return the array of episode_length and episode_reward with shape
        (len(cached_buffer_ids), ...), where (episode_length[i],
        episode_reward[i]) refers to the cached_buffer_ids[i]th cached buffer's
        corresponding episode result.
        """
        if cached_buffer_ids is None:
            cached_buffer_ids = np.arange(self.cached_buffer_num)
        else:  # make sure it is np.ndarray
            cached_buffer_ids = np.asarray(cached_buffer_ids)
        # in self.buffers, the first buffer is main_buffer
        buffer_ids = cached_buffer_ids + 1  # type: ignore
        result = super().add(obs, act, rew, done, obs_next, info,
                             policy, buffer_ids=buffer_ids, **kwargs)
        # find the terminated episode, move data from cached buf to main buf
        for buffer_idx in cached_buffer_ids[np.asarray(done, np.bool_)]:
            self.main_buffer.update(self.cached_buffers[buffer_idx])
            self.cached_buffers[buffer_idx].reset()
        return result

    def __getitem__(
        self, index: Union[slice, int, np.integer, np.ndarray]
    ) -> Batch:
        batch = super().__getitem__(index)
        if self._is_prioritized:
            indice = self._indices[index]
            mask = indice < self.main_buffer.maxsize
            batch.weight = np.ones(len(indice))
            batch.weight[mask] = self.main_buffer.get_weight(indice[mask])
        return batch

    def update_weight(
        self,
        index: np.ndarray,
        new_weight: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Update priority weight by index in main buffer.

        :param np.ndarray index: index you want to update weight.
        :param np.ndarray new_weight: new priority weight you want to update.
        """
        if self._is_prioritized:
            mask = index < self.main_buffer.maxsize
            self.main_buffer.update_weight(index[mask], new_weight[mask])
