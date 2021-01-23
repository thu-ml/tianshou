import h5py
import warnings
import numpy as np
from numbers import Number
from typing import Any, Dict, List, Tuple, Union, Callable, Optional

from tianshou.data.batch import _create_value
from tianshou.data import Batch, SegmentTree, to_numpy
from tianshou.data.utils.converter import to_hdf5, from_hdf5


class ReplayBuffer:
    """:class:`~tianshou.data.ReplayBuffer` stores data generated from \
    interaction between the policy and environment. ReplayBuffer can be \
    considered as a specialized form(management) of Batch.

    The current implementation of Tianshou typically use 7 reserved keys in
    :class:`~tianshou.data.Batch`:

    * ``obs`` the observation of step :math:`t` ;
    * ``act`` the action of step :math:`t` ;
    * ``rew`` the reward of step :math:`t` ;
    * ``done`` the done flag of step :math:`t` ;
    * ``obs_next`` the observation of step :math:`t+1` ;
    * ``info`` the info of step :math:`t` (in ``gym.Env``, the ``env.step()`` \
    function returns 4 arguments, and the last one is ``info``);
    * ``policy`` the data computed by policy in step :math:`t`;

    The following code snippet illustrates its usage:
    ::

        >>> import pickle, numpy as np
        >>> from tianshou.data import ReplayBuffer
        >>> buf = ReplayBuffer(size=20)
        >>> for i in range(3):
        ...     buf.add(obs=i, act=i, rew=i, done=i, obs_next=i + 1, info={})
        >>> buf.obs
        # since we set size = 20, len(buf.obs) == 20.
        array([0., 1., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0.])
        >>> # but there are only three valid items, so len(buf) == 3.
        >>> len(buf)
        3
        >>> # save to file "buf.pkl"
        >>> pickle.dump(buf, open('buf.pkl', 'wb'))
        >>> # save to HDF5 file
        >>> buf.save_hdf5('buf.hdf5')
        >>> buf2 = ReplayBuffer(size=10)
        >>> for i in range(15):
        ...     buf2.add(obs=i, act=i, rew=i, done=i, obs_next=i + 1, info={})
        >>> len(buf2)
        10
        >>> buf2.obs
        # since its size = 10, it only stores the last 10 steps' result.
        array([10., 11., 12., 13., 14.,  5.,  6.,  7.,  8.,  9.])

        >>> # move buf2's result into buf (meanwhile keep it chronologically)
        >>> buf.update(buf2)
        array([ 0.,  1.,  2.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.,
                0.,  0.,  0.,  0.,  0.,  0.,  0.])

        >>> # get a random sample from buffer
        >>> # the batch_data is equal to buf[indice].
        >>> batch_data, indice = buf.sample(batch_size=4)
        >>> batch_data.obs == buf[indice].obs
        array([ True,  True,  True,  True])
        >>> len(buf)
        13
        >>> buf = pickle.load(open('buf.pkl', 'rb'))  # load from "buf.pkl"
        >>> len(buf)
        3
        >>> # load complete buffer from HDF5 file
        >>> buf = ReplayBuffer.load_hdf5('buf.hdf5')
        >>> len(buf)
        3
        >>> # load contents of HDF5 file into existing buffer
        >>> # (only possible if size of buffer and data in file match)
        >>> buf.load_contents_hdf5('buf.hdf5')
        >>> len(buf)
        3

    :class:`~tianshou.data.ReplayBuffer` also supports frame_stack sampling
    (typically for RNN usage, see issue#19), ignoring storing the next
    observation (save memory in atari tasks), and multi-modal observation (see
    issue#38):
    ::

        >>> buf = ReplayBuffer(size=9, stack_num=4, ignore_obs_next=True)
        >>> for i in range(16):
        ...     done = i % 5 == 0
        ...     buf.add(obs={'id': i}, act=i, rew=i, done=done,
        ...             obs_next={'id': i + 1})
        >>> print(buf)  # you can see obs_next is not saved in buf
        ReplayBuffer(
            act: array([ 9., 10., 11., 12., 13., 14., 15.,  7.,  8.]),
            done: array([0., 1., 0., 0., 0., 0., 1., 0., 0.]),
            info: Batch(),
            obs: Batch(
                     id: array([ 9., 10., 11., 12., 13., 14., 15.,  7.,  8.]),
                 ),
            policy: Batch(),
            rew: array([ 9., 10., 11., 12., 13., 14., 15.,  7.,  8.]),
        )
        >>> index = np.arange(len(buf))
        >>> print(buf.get(index, 'obs').id)
        [[ 7.  7.  8.  9.]
         [ 7.  8.  9. 10.]
         [11. 11. 11. 11.]
         [11. 11. 11. 12.]
         [11. 11. 12. 13.]
         [11. 12. 13. 14.]
         [12. 13. 14. 15.]
         [ 7.  7.  7.  7.]
         [ 7.  7.  7.  8.]]
        >>> # here is another way to get the stacked data
        >>> # (stack only for obs and obs_next)
        >>> abs(buf.get(index, 'obs')['id'] - buf[index].obs.id).sum().sum()
        0.0
        >>> # we can get obs_next through __getitem__, even if it doesn't exist
        >>> print(buf[:].obs_next.id)
        [[ 7.  8.  9. 10.]
         [ 7.  8.  9. 10.]
         [11. 11. 11. 12.]
         [11. 11. 12. 13.]
         [11. 12. 13. 14.]
         [12. 13. 14. 15.]
         [12. 13. 14. 15.]
         [ 7.  7.  7.  8.]
         [ 7.  7.  8.  9.]]

    :param int size: the size of replay buffer.
    :param int stack_num: the frame-stack sampling argument, should be greater
        than or equal to 1, defaults to 1 (no stacking).
    :param bool ignore_obs_next: whether to store obs_next, defaults to False.
    :param bool save_only_last_obs: only save the last obs/obs_next when it has
        a shape of (timestep, ...)  because of temporal stacking, defaults to
        False.
    """
    _reserved_keys = ("obs", "act", "rew", "done",
                      "obs_next", "info", "policy")

    def __init__(
        self,
        size: int,
        stack_num: int = 1,
        ignore_obs_next: bool = False,
        save_only_last_obs: bool = False,
        alloc_fn: Optional[Callable[["ReplayBuffer", List[str], Any], None]]
        = None,
    ) -> None:
        super().__init__()
        self.maxsize = size
        assert stack_num > 0, "stack_num should greater than 0"
        self.stack_num = stack_num
        self._indices = np.arange(size)
        self._save_obs_next = not ignore_obs_next
        self._save_only_last_obs = save_only_last_obs
        self._index = 0  # current index
        self._size = 0  # current buffer size
        self._meta: Batch = Batch()
        self._alloc = alloc_fn or ReplayBuffer.default_alloc_fn
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

    def __setattr__(self, key: str, value: Any) -> None:
        """Set self.key = value."""
        assert key not in self._reserved_keys, (
            "key '{}' is reserved and cannot be assigned".format(key))
        super().__setattr__(key, value)

    @staticmethod
    def default_alloc_fn(
        buffer: "ReplayBuffer", key: List[str], value: Any
    ) -> None:
        """Allocate memory on buffer._meta for new (key, value) pair."""
        data = buffer._meta
        for k in key[:-1]:
            data = data[k]
        data[key[-1]] = _create_value(value, buffer.maxsize)

    def _add_to_buffer(self, name: str, inst: Any) -> None:
        try:
            value = self._meta.__dict__[name]
        except KeyError:
            self._alloc(self, [name], inst)
            value = self._meta[name]
        try:
            value[self._index] = inst
        except KeyError:  # inst is a dict/Batch
            for key in set(inst.keys()).difference(value.keys()):
                self._alloc(self, [name, key], inst)
            value[self._index] = inst

    def set_batch(self, batch: Batch):
        """Manually choose the batch you want the ReplayBuffer to manage."""
        assert len(batch) == self.maxsize, \
            "Input batch doesn't meet ReplayBuffer's data form requirement."
        self._meta = batch

    def unfinished_index(self) -> np.ndarray:
        """Return the index of unfinished episode."""
        try:
            last = (self._index - 1) % self._size
        except ZeroDivisionError:
            return np.array([])
        return np.array([last]) if not self.done[last] else np.array([])

    def prev(self, index: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        """Return the index of previous transition.

        The index won't be modified if it is the beginning of one episode.
        """
        prev_index = (index - 1) % self._size
        end_flag = self.done[prev_index] | \
            np.isin(prev_index, self.unfinished_index())
        prev_index = (prev_index + end_flag) % self._size
        return prev_index

    def next(self, index: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        """Return the index of next transition.

        The index won't be modified if it is the end of one episode.
        """
        end_flag = self.done[index] | np.isin(index, self.unfinished_index())
        return (index + (1 - end_flag)) % self._size

    def update(self, buffer: "ReplayBuffer") -> None:
        """Move the data from the given buffer to current buffer."""
        if len(buffer) == 0:
            return
        stack_num_orig, buffer.stack_num = buffer.stack_num, 1
        indices = buffer.sample_index(0)  # get all available indices
        for i in indices:
            self.add(**buffer[i])
        buffer.stack_num = stack_num_orig

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
    ) -> None:
        """Add a batch of data into replay buffer."""
        assert isinstance(
            info, (dict, Batch)
        ), "You should return a dict in the last argument of env.step()."
        if self._save_only_last_obs:
            obs = obs[-1]
        self._add_to_buffer("obs", obs)
        self._add_to_buffer("act", act)
        # make sure the data type of reward is float instead of int
        self._add_to_buffer("rew", rew * 1.0)  # type: ignore
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

    def reset(self) -> None:
        """Clear all the data in replay buffer."""
        self._index = self._size = 0

    def sample_index(self, batch_size: int) -> np.ndarray:
        """Get a random sample of index with size = batch_size.

        Return all available indices in the buffer if batch_size is 0.
        """
        if batch_size > 0:
            return np.random.choice(self._size, batch_size)
        else:  # construct current available indices
            return np.concatenate([
                np.arange(self._index, self._size),
                np.arange(0, self._index),
            ])

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with size = batch_size.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.
        """
        indices = self.sample_index(batch_size)
        return self[indices], indices

    def get(
        self,
        indice: Union[int, np.integer, np.ndarray],
        key: str,
        stack_num: Optional[int] = None,
    ) -> Union[Batch, np.ndarray]:
        """Return the stacked result.

        E.g. [s_{t-3}, s_{t-2}, s_{t-1}, s_t], where s is self.key, t is the
        indice.
        """
        if stack_num is None:
            stack_num = self.stack_num
        val = self._meta[key]
        try:
            if stack_num == 1:  # the most often case
                return val[indice]
            stack: List[Any] = []
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
        if self._save_obs_next:
            obs_next = self.get(index, "obs_next")
        else:
            obs_next = self.get(self.next(index), "obs")
        return Batch(
            obs=self.get(index, "obs"),
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            obs_next=obs_next,
            info=self.get(index, "info"),
            policy=self.get(index, "policy"),
        )

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
        warnings.warn("ListReplayBuffer will be replaced soon.")
        super().__init__(size=0, ignore_obs_next=False, **kwargs)

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        raise NotImplementedError("ListReplayBuffer cannot be sampled!")

    def _add_to_buffer(
        self, name: str, inst: Union[dict, Batch, np.ndarray, float, int, bool]
    ) -> None:
        if self._meta.get(name) is None:
            self._meta.__dict__[name] = []
        self._meta[name].append(inst)

    def reset(self) -> None:
        self._index = self._size = 0
        for k in self._meta.keys():
            if isinstance(self._meta[k], list):
                self._meta.__dict__[k] = []


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
    ) -> None:
        """Add a batch of data into replay buffer."""
        if weight is None:
            weight = self._max_prio
        else:
            weight = np.abs(weight)
            self._max_prio = max(self._max_prio, weight)
            self._min_prio = min(self._min_prio, weight)
        self.weight[self._index] = weight ** self._alpha
        super().add(obs, act, rew, done, obs_next, info, policy, **kwargs)

    def sample_index(self, batch_size: int) -> np.ndarray:
        assert self._size > 0, "Cannot sample a buffer with 0 size."
        if batch_size == 0:
            return np.concatenate([
                np.arange(self._index, self._size),
                np.arange(0, self._index),
            ])
        else:
            scalar = np.random.rand(batch_size) * self.weight.reduce()
            return self.weight.get_prefix_sum_idx(scalar)

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with priority probability.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.

        The "weight" in the returned Batch is the weight on loss function
        to de-bias the sampling process (some transition tuples are sampled
        more often so their losses are weighted less).
        """
        indice = self.sample_index(batch_size)
        batch = self[indice]
        # important sampling weight calculation
        # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
        # simplified formula: (p_j/p_min)**(-beta)
        batch.weight = (batch.weight / self._min_prio) ** (-self._beta)
        return batch, indice

    def update_weight(
        self,
        indice: Union[np.ndarray],
        new_weight: np.ndarray
    ) -> None:
        """Update priority weight by indice in this buffer.

        :param np.ndarray indice: indice you want to update weight.
        :param np.ndarray new_weight: new priority weight you want to update.
        """
        weight = np.abs(to_numpy(new_weight)) + self.__eps
        self.weight[indice] = weight ** self._alpha
        self._max_prio = max(self._max_prio, weight.max())
        self._min_prio = min(self._min_prio, weight.min())

    def __getitem__(
        self, index: Union[slice, int, np.integer, np.ndarray]
    ) -> Batch:
        batch = super().__getitem__(index)
        batch.weight = self.weight[index]
        return batch


class VectorReplayBuffer(ReplayBuffer):
    """VectorReplayBuffer contains n ReplayBuffer with the given size, where \
    n equals to buffer_num.

    :param float size: the size of each ReplayBuffer.
    :param float buffer_num: number of ReplayBuffer needs to be handled.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more detailed
        explanation.
    """

    def __init__(
        self, size: int, buffer_num: int, **kwargs: Any
    ) -> None:

        def buffer_alloc_fn(
            buffer: ReplayBuffer, key: List[str], value: Any
        ) -> None:
            data = self._meta
            for k in key[:-1]:
                data = data[k]
            data[key[-1]] = _create_value(value, self.maxsize)
            self._set_batch_for_children()

        assert size > 0 and buffer_num > 0
        self.buffer_num = buffer_num
        kwargs["alloc_fn"] = kwargs.get("alloc_fn", buffer_alloc_fn)
        self.buffers = [ReplayBuffer(size, **kwargs)
                        for _ in range(buffer_num)]
        super().__init__(size=buffer_num * size, **kwargs)

    def __len__(self) -> int:
        return sum([len(buf) for buf in self.buffers])

    def unfinished_index(self) -> np.ndarray:
        return np.concatenate([buf.unfinished_index() for buf in self.buffers])

    def prev(self, index: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        index = np.asarray(index)
        prev_indices = np.zeros_like(index)
        for i, buf in enumerate(self.buffers):
            lower, upper = buf.maxsize * i, buf.maxsize * (i + 1)
            mask = (lower <= index) & (index < upper)
            if np.any(mask):
                prev_indices[mask] = buf.prev(index[mask] - lower) + lower
        return prev_indices

    def next(self, index: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        index = np.asarray(index)
        next_indices = np.zeros_like(index)
        for i, buf in enumerate(self.buffers):
            lower, upper = buf.maxsize * i, buf.maxsize * (i + 1)
            mask = (lower <= index) & (index < upper)
            if np.any(mask):
                next_indices[mask] = buf.next(index[mask] - lower) + lower
        return next_indices

    def update(self, buffer: ReplayBuffer) -> None:
        raise NotImplementedError

    def add(
        self,
        obs: Any,
        act: Any,
        rew: Union[np.ndarray],
        done: Union[np.ndarray],
        obs_next: Any = Batch(),
        info: Optional[Batch] = Batch(),
        policy: Optional[Batch] = Batch(),
        buffer_index: Optional[Union[np.ndarray, List[int]]] = None,
        **kwargs: Any
    ) -> None:
        """Add a batch of data into VectorReplayBuffer.

        Each of the data's length (first dimension) must equal to the length of
        buffer_index.
        """
        assert buffer_index is not None, \
            "buffer_index is required in VectorReplayBuffer.add()"
        batch = Batch(obs=obs, act=act, rew=rew, done=done,
                      obs_next=obs_next, info=info, policy=policy)
        assert len(buffer_index) == len(batch)
        for batch_idx, buffer_idx in enumerate(buffer_index):
            self.buffers[buffer_idx].add(**batch[batch_idx])

    def _set_batch_for_children(self):
        for i, buf in enumerate(self.buffers):
            start, end = buf.maxsize * i, buf.maxsize * (i + 1)
            buf.set_batch(self._meta[start:end])

    def set_batch(self, batch: Batch):
        super().set_batch(batch)
        self._set_batch_for_children()

    def reset(self) -> None:
        for buf in self.buffers:
            buf.reset()

    def sample_index(self, batch_size: int) -> np.ndarray:
        if batch_size == 0:  # get all available indices
            sample_num = np.zeros(self.buffer_num, np.int)
        else:
            buffer_lens = np.array([len(buf) for buf in self.buffers])
            buffer_idx = np.random.choice(self.buffer_num, batch_size,
                                          p=buffer_lens / buffer_lens.sum())
            sample_num = np.bincount(buffer_idx, minlength=self.buffer_num)
            # avoid batch_size > 0 and sample_num == 0 -> get child's all data
            sample_num[sample_num == 0] = -1

        indices = []
        for i, (buf, bsz) in enumerate(zip(self.buffers, sample_num)):
            if bsz >= 0:
                offset = buf.maxsize * i
                indices.append(buf.sample_index(bsz) + offset)
        return np.concatenate(indices)


class CachedReplayBuffer(ReplayBuffer):
    """CachedReplayBuffer can be considered as a combination of one main buffer
    and a list of cached_buffers. It's designed to used by collector to allow
    parallel collecting in collector. In CachedReplayBuffer is not organized
    chronologically, but standard API like start()/starts()/ends/next() are
    provided to help CachedReplayBuffer to be used just like ReplayBuffer.
    #TODO finish doc
    """

    def __init__(
        self,
        size: int,
        cached_buffer_n: int,
        max_episode_length: int,
        **kwargs: Any,
    ) -> None:
        assert cached_buffer_n > 0 and max_episode_length > 0
        if cached_buffer_n == 1:
            warnings.warn(
                "CachedReplayBuffer with cached_buffer_n = 1 will"
                "cause low efficiency. Please consider using ReplayBuffer"
                "which is not in cached form.",
                Warning)
        maxsize = size + cached_buffer_n * max_episode_length
        self.cached_bufs_n = cached_buffer_n
        self.main_buffer = ReplayBuffer(size, **kwargs)
        self.cached_buffer = VectorReplayBuffer(
            max_episode_length, cached_buffer_n, **kwargs)
        super().__init__(size=maxsize, **kwargs)

    def __len__(self) -> int:
        """Return len(self)."""
        return len(self.main_buffer) + len(self.cached_buffer)

    def unfinished_index(self) -> np.ndarray:
        return np.concatenate([self.main_buffer.unfinished_index(),
                               self.cached_buffer.unfinished_index()])

    def prev(self, index: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        """Return one step previous index."""
        assert np.all(index >= 0) and np.all(index < self.maxsize)
        prev_indices = np.full(index.shape, -1)
        mask = index < self.main_buffer.maxsize
        prev_indices[mask] = self.main_buffer.prev(index[mask])
        prev_indices[~mask] = self.cached_buffer.prev(
            index[~mask] - self.main_buffer.maxsize) + \
            self.main_buffer.maxsize
        return prev_indices

    def next(self, index: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        """Return one step next index."""
        assert np.all(index >= 0) and np.all(index < self.maxsize)
        next_indices = np.full(index.shape, -1)
        mask = index < self.main_buffer.maxsize
        next_indices[mask] = self.main_buffer.next(index[mask])
        next_indices[~mask] = self.cached_buffer.next(
            index[~mask] - self.main_buffer.maxsize) + \
            self.main_buffer.maxsize
        return next_indices

    def update(self, buffer: "ReplayBuffer") -> int:
        """CachedReplayBuffer will only update data from buffer which is in
        episode form. Return an integer which indicates the number of steps
        being ignored."""
        # For now update method copy element one by one, which is too slow.
        if isinstance(buffer, CachedReplayBuffer):
            buffer = buffer.main_buffer
        # now treat buffer like a normal ReplayBuffer and
        # remove those incomplete steps
        if len(buffer) == 0:
            return 0
        diposed_count = 0
        # TODO use standard API now
        end = (buffer._index - 1) % len(buffer)
        begin = buffer._index % len(buffer)
        while True:
            if buffer.done[end] > 0:
                break
            else:
                diposed_count = diposed_count + 1
                if end == begin:
                    assert diposed_count == len(self)
                    return diposed_count
                end = (end - 1) % len(buffer)
        while True:
            self.main_buffer.add(**buffer[begin])
            if begin == end:
                return diposed_count
            begin = (begin + 1) % len(buffer)

    def add(
        self,
        obs: Any,
        act: Any,
        rew: Union[Number, np.number, np.ndarray],
        done: Union[Number, np.number, np.bool_],
        obs_next: Any = None,
        info: Optional[Union[dict, Batch]] = {},
        policy: Optional[Union[dict, Batch]] = {},
        index: Optional[Union[int, np.integer, np.ndarray, List[int]]] = None,
        **kwargs: Any
    ) -> None:
        if index is None:
            index = range(self.cached_bufs_n)
        index = np.atleast_1d(index).astype(np.int)
        assert(index.ndim == 1)

        obs = np.atleast_1d(obs)
        act = np.atleast_1d(act)
        rew = np.atleast_1d(rew)
        done = np.atleast_1d(done)
        # TODO ugly code
        if isinstance(obs_next, Batch) and obs_next.is_empty():
            obs_next = None
        if isinstance(info, Batch) and info.is_empty():
            info = {}
        if isinstance(policy, Batch) and policy.is_empty():
            policy = {}
        obs_next = np.atleast_1d([None] * len(index)) \
            if obs_next is None else np.atleast_1d(obs_next)
        info = np.atleast_1d(
            [{}] * len(index)) if info == {} else np.atleast_1d(info)
        policy = np.atleast_1d(
            [{}] * len(index)) if policy == {} else np.atleast_1d(policy)

        # TODO what if data is already in episodes,
        # what if i want to add mutiple data ?
        if self._meta.is_empty():
            self._initialise(obs[0], act[0], rew[0], done[0], obs_next[0],
                             info[0], policy[0])

        self.cached_buffer.add(obs, act, rew, done, obs_next,
                               info, policy, index, False, **kwargs)
        return self._main_buf_update()

    def _main_buf_update(self):
        lens = np.zeros((self.cached_bufs_n, ), dtype=np.int)
        rews = np.zeros((self.cached_bufs_n, ))
        start_indexs = np.zeros((self.cached_bufs_n, ), dtype=np.int)
        for i, buf in enumerate(self.cached_buffer.bufs):
            if buf.done[buf._index - 1] > 0:
                lens[i] = len(buf)
                rews[i] = np.sum(buf.rew[:lens[i]])
                start_indexs[i] = self.main_buffer._index
                if self.main_buffer.maxsize > 0:
                    # _maxsize of main_buffer might be 0 in test collector.
                    self.main_buffer.update(buf)
                buf.reset()
        return lens, rews, start_indexs

    def reset(self) -> None:
        self.cached_buffer.reset()
        self.main_buffer.reset()

    def sample_index(self, batch_size: int,
                     is_from_main_buf=False, **kwargs) -> np.ndarray:
        if is_from_main_buf:
            return self.main_buffer.sample_index(batch_size, **kwargs)

        avail_indexes = np.arange(len(self), dtype=np.int)
        start = len(self.main_buffer)
        add = self.main_buffer.maxsize - len(self.main_buffer)
        for buf in self.cached_buffer.bufs:
            end = start + len(buf)
            avail_indexes[start:end] = avail_indexes[start:end] + add
            start = end
            add = add + buf.maxsize - len(buf)
        assert len(avail_indexes) > 0, "No available indice can be sampled."
        # TODO consider making avail_indexes a seperate method
        if batch_size == 0:
            # data not in chronological order
            return avail_indexes
        else:
            return np.random.choice(avail_indexes, batch_size)

    def _initialise(
        self,
        obs: Any,
        act: Any,
        rew: Union[Number, np.number, np.ndarray],
        done: Union[Number, np.number, np.bool_],
        obs_next: Any = None,
        info: Optional[Union[dict, Batch]] = {},
        policy: Optional[Union[dict, Batch]] = {}
    ) -> None:
        assert(self._meta.is_empty())
        # to initialise self._meta
        super().add(obs, act, rew, done, obs_next, info, policy)
        super().reset()  # TODO delete useless varible?
        del self._index
        del self._size
        self._set_batch_for_children()

    def _set_batch_for_children(self):
        self.main_buffer.set_batch(self._meta[:self.main_buffer.maxsize])
        self.cached_buffer.set_batch(self._meta[self.main_buffer.maxsize:])

    def set_batch(self, batch: "Batch"):
        """Manually choose the batch you want the ReplayBuffer to manage."""
        super().set_batch(batch)
        self._set_batch_for_children()
