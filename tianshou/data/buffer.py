import h5py
import torch
import numpy as np
from numbers import Number
from typing import Any, Dict, List, Tuple, Union, Optional

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
    :param bool sample_avail: the parameter indicating sampling only available
        index when using frame-stack sampling method, defaults to False.
        This feature is not supported in Prioritized Replay Buffer currently.
    """
    _reserved_keys = {'obs', 'act', 'rew', 'done', 'obs_next', 'info', 'policy'}

    def __init__(
        self,
        size: int,
        stack_num: int = 1,
        ignore_obs_next: bool = False,
        save_only_last_obs: bool = False,
        sample_avail: bool = False,
    ) -> None:
        super().__init__()
        #TODO _maxsize == 0 handle
        self._maxsize = size
        self._indices = np.arange(size)
        # consider move stacking option to another
        self.stack_num = stack_num
        self._avail = sample_avail and stack_num > 1
        self._avail_index: List[int] = []
        self._save_s_ = not ignore_obs_next
        self._last_obs = save_only_last_obs
        self._index = 0
        self._size = 0
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
        self._indices = np.arange(state["_maxsize"])
        self.__dict__.update(state)

    def __getstate__(self) -> dict:
        exclude = {"_indices"}
        state = {k: v for k, v in self.__dict__.items() if k not in exclude}
        return state

    def __setattr__(self, key: str, value: Any) -> None:
        """Set self.key = value."""
        assert key not in self._reserved_keys, (
                "key '{}' is reserved and cannot be assigned".format(key))
        super().__setattr__(key, value)

    def _add_to_buffer(self, name: str, inst: Any) -> None:
        try:
            value = self._meta.__dict__[name]
        except KeyError:
            self._meta.__dict__[name] = _create_value(inst, self._maxsize)
            value = self._meta.__dict__[name]
        try:
            value[self._index] = inst
        except KeyError:
            for key in set(inst.keys()).difference(value.__dict__.keys()):
                value.__dict__[key] = _create_value(inst[key], self._maxsize)
            value[self._index] = inst

    @property
    def stack_num(self) -> int:
        return self._stack

    @stack_num.setter
    def stack_num(self, num: int) -> None:
        assert num > 0, "stack_num should greater than 0"
        self._stack = num

    def update(self, buffer: "ReplayBuffer") -> None:
        """Move the data from the given buffer to self."""
         # TODO 'one by one copying' can be greatly improved.
         # Can we move data as a whole batch to save time?
         # what if self._maxsize << buffer._maxize, can we detect that 
         # and just ignore those element to be rewrite?
        if len(buffer) == 0:
            return
        i = begin = buffer._index % len(buffer)
        stack_num_orig = buffer.stack_num
        buffer.stack_num = 1
        while True:
            self.add(**buffer[i])  # type: ignore
            i = (i + 1) % len(buffer)
            if i == begin:
                break
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
        """Add a batch of data into replay buffer.
        Expect all input to be batch, dict, or numpy array"""
        # TODO should we consider to support batch input?
        assert isinstance(
            info, (dict, Batch)
        ), "You should return a dict in the last argument of env.step()."
        if self._last_obs:
            obs = obs[-1]
        self._add_to_buffer("obs", obs)
        self._add_to_buffer("act", act)
        # make sure the reward is a float instead of an int
        self._add_to_buffer("rew", rew * 1.0)  # type: ignore
        self._add_to_buffer("done", done)
        if self._save_s_:
            if obs_next is None:
                obs_next = Batch()
            elif self._last_obs:
                obs_next = obs_next[-1]
            self._add_to_buffer("obs_next", obs_next)
        self._add_to_buffer("info", info)
        self._add_to_buffer("policy", policy)

        # maintain available index for frame-stack sampling
        if self._avail:
            # update current frame
            avail = sum(self.done[i] for i in range(
                self._index - self.stack_num + 1, self._index)) == 0
            if self._size < self.stack_num - 1:
                avail = False
            if avail and self._index not in self._avail_index:
                self._avail_index.append(self._index)
            elif not avail and self._index in self._avail_index:
                self._avail_index.remove(self._index)
            # remove the later available frame because of broken storage
            t = (self._index + self.stack_num - 1) % self._maxsize
            if t in self._avail_index:
                self._avail_index.remove(t)

        if self._maxsize > 0:
            self._size = min(self._size + 1, self._maxsize)
            self._index = (self._index + 1) % self._maxsize
        else:
            self._size = self._index = self._index + 1

    def reset(self) -> None:
        """Clear all the data in replay buffer."""
        self._index = 0
        self._size = 0
        self._avail_index = []

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with size equal to batch_size.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.
        """
        if batch_size > 0:
            _all = self._avail_index if self._avail else self._size
            indice = np.random.choice(_all, batch_size)
        else:
            if self._avail:
                indice = np.array(self._avail_index)
            else:
                indice = np.concatenate([
                    np.arange(self._index, self._size),
                    np.arange(0, self._index),
                ])
        assert len(indice) > 0, "No available indice can be sampled."
        return self[indice], indice

    def get(
        self,
        indice: Union[slice, int, np.integer, np.ndarray],
        key: str,
        stack_num: Optional[int] = None,
    ) -> Union[Batch, np.ndarray]:
        """Return the stacked result.

        E.g. [s_{t-3}, s_{t-2}, s_{t-1}, s_t], where s is self.key, t is the
        indice. The stack_num (here equals to 4) is given from buffer
        initialization procedure.
        """
        if stack_num is None:
            stack_num = self.stack_num
        if stack_num == 1:  # the most often case
            if key != "obs_next" or self._save_s_:
                val = self._meta.__dict__[key]
                try:
                    return val[indice]
                except IndexError as e:
                    if not (isinstance(val, Batch) and val.is_empty()):
                        raise e  # val != Batch()
                    return Batch()
        indice = self._indices[:self._size][indice]
        done = self._meta.__dict__["done"]
        if key == "obs_next" and not self._save_s_:
            indice += 1 - done[indice].astype(np.int)
            indice[indice == self._size] = 0
            key = "obs"
        val = self._meta.__dict__[key]
        try:
            if stack_num == 1:
                return val[indice]
            stack: List[Any] = []
            for _ in range(stack_num):
                stack = [val[indice]] + stack
                pre_indice = np.asarray(indice - 1)
                pre_indice[pre_indice == -1] = self._size - 1
                indice = np.asarray(
                    pre_indice + done[pre_indice].astype(np.int))
                indice[indice == self._size] = 0
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
        return Batch(
            obs=self.get(index, "obs"),
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            obs_next=self.get(index, "obs_next"),
            info=self.get(index, "info"),
            policy=self.get(index, "policy"),
        )

    def set_batch(self, batch: "Batch"):
        """Manually choose the batch you want the ReplayBuffer to manage. This 
        method should be called instantly after the ReplayBuffer is initialised.
        """
        assert self._meta.is_empty(), "This method cannot be called after add() method"
        self._meta = batch
        assert not self._is_meta_corrupted(), (
            "Input batch doesn't meet ReplayBuffer's data form requirement.")

    def _is_meta_corrupted(self)-> bool:
        """Assert if self._meta: Batch is still in legal form.
        """
        #TODO do we need to check the chlid? is cache
        if set(self._meta.keys()) != self._reserved_keys:
            return True
        for v in self._meta.values():
            if isinstance(v, Batch):
                if not v.is_empty() and v.shape[0]!= self._maxsize:
                    return True
            elif isinstance(v, np.ndarray):
                if v.shape[0]!= self._maxsize:
                    return True
            else:
                return True
        return False

    def save_hdf5(self, path: str) -> None:
        """Save replay buffer to HDF5 file."""
        with h5py.File(path, "w") as f:
            to_hdf5(self.__getstate__(), f)

    @classmethod
    def load_hdf5(
        cls, path: str, device: Optional[str] = None
    ) -> "ReplayBuffer":
        """Load replay buffer from HDF5 file."""
        with h5py.File(path, "r") as f:
            buf = cls.__new__(cls)
            buf.__setstate__(from_hdf5(f, device=device))
        return buf

    def start(self, index):
        """return start indices of given indices"""
        assert index < len(self) and index >= 0, "Input index illegal."
        sorted_starts = np.sort(self.starts())
        ret = np.searchsorted(sorted_starts, index, side="right") - 1
        ret[ret < 0] = sorted_starts[-1]
        return ret

    def next(self, index):
        """return next n step indices"""
        assert index < len(self) and index >= 0, "Input index illegal."
        return (index + ~(self.done[index]|index==self._index))%len(self)

    def starts(self):
        """return indices of all episodes"""
        if len(self) > 0:
            return (self.ends()+1)%len(self)
        else:
            return np.array([], dtype = np.int)

    def ends(self):
        """return unfinished indices."""
        if len(self) > 0:
            last_write_in = int((self._index - 1)%len(self))
            if self.done[last_write_in]:
                return np.where(self.done[:len(self)])[0]
            else:
                return np.append(np.where(self.done[:len(self)])[0], last_write_in)
        else:
            return np.array([], dtype = np.int)


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
        super().__init__(size=0, ignore_obs_next=False, **kwargs)

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        raise NotImplementedError("ListReplayBuffer cannot be sampled!")

    def _add_to_buffer(
        self, name: str, inst: Union[dict, Batch, np.ndarray, float, int, bool]
    ) -> None:
        if self._meta.__dict__.get(name) is None:
            self._meta.__dict__[name] = []
        self._meta.__dict__[name].append(inst)

    def reset(self) -> None:
        self._index = self._size = 0
        for k in list(self._meta.__dict__.keys()):
            if isinstance(self._meta.__dict__[k], list):
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

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with priority probability.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.

        The "weight" in the returned Batch is the weight on loss function
        to de-bias the sampling process (some transition tuples are sampled
        more often so their losses are weighted less).
        """
        assert self._size > 0, "Cannot sample a buffer with 0 size!"
        if batch_size == 0:
            indice = np.concatenate([
                np.arange(self._index, self._size),
                np.arange(0, self._index),
            ])
        else:
            scalar = np.random.rand(batch_size) * self.weight.reduce()
            indice = self.weight.get_prefix_sum_idx(scalar)
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
        return Batch(
            obs=self.get(index, "obs"),
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            obs_next=self.get(index, "obs_next"),
            info=self.get(index, "info"),
            policy=self.get(index, "policy"),
            weight=self.weight[index],
        )

class CachedReplayBuffer(ReplayBuffer):
    """CachedReplayBuffer can be considered as a combination of one main buffer
    and a list of cached_buffers. It's designed to used by collector to allow
    parallel collecting in collector. In CachedReplayBuffer is not organized
    chronologically, but standard API like start()/starts()/ends/next() are provided
    to help CachedReplayBuffer to be used just like ReplayBuffer.
    """
    def __init__(
        self,
        size: int,
        cached_buf_n: int,
        max_length: int,
        **kwargs: Any,
    ) -> None:
        """
        TODO support stack in the future
        """
        assert cached_buf_n > 0
        # TODO what if people don't care about how buffer is organized
        assert max_length > 0
        if cached_buf_n == 1:
            import warnings
            warnings.warn(
                "CachedReplayBuffer with cached_buf_n = 1 will cause low efficiency. "
                "Please consider using ReplayBuffer which is not in cached form.",
                Warning)
        
        _maxsize = size+cached_buf_n*max_length
        self.cached_bufs_n = cached_buf_n
        # TODO see if we can generalize to all kinds of buufer
        self.main_buf = ReplayBuffer(size, **kwargs)
        # TODO cached_bufs can be consider to be replced by vector 
        # buffer in the future
        self.cached_bufs = np.array([ReplayBuffer(max_length, **kwargs)
                                        for _ in range(cached_buf_n)])
        super().__init__(size= _maxsize, **kwargs)
        # TODO support, or just delete stack_num option from Replay buffer for now
        assert self.stack_num == 1
        
    def __len__(self) -> int:
        """Return len(self)."""
        return len(self.main_buf) + np.sum([len(b) for b in self.cached_bufs])

    def update(self, buffer: "ReplayBuffer") -> int:
        """CachedReplayBuffer will only update data from buffer which is in
        episode form. Return an integer which indicates the number of steps
        being ignored."""
        # For now update method copy element one by one, which is too slow.
        if isinstance(buffer, CachedReplayBuffer):
            buffer = buffer.main_buf
        # now treat buffer like a normal ReplayBuffer and remove those incomplete steps
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
            self.main_buf.add(**buffer[begin])
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
        """
        
        """
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
        obs_next = np.atleast_1d([None]*len(index)) if obs_next is None else np.atleast_1d(obs_next)
        info = np.atleast_1d([{}]*len(index)) if info == {} else np.atleast_1d(info)
        policy = np.atleast_1d([{}]*len(index)) if policy == {} else np.atleast_1d(policy)

        # TODO what if data is already in episodes, what if i want to add mutiple data ?
        # can accelerate
        if self._meta.is_empty():
            self._cache_initialise(obs[0], act[0], rew[0], done[0], obs_next[0],
                                    info[0], policy[0])
        # now we add data to selected cached_bufs one by one
        cached_bufs_slice = self.cached_bufs[index]
        for i, b in enumerate(cached_bufs_slice):
            b.add(obs[i], act[i], rew[i], done[i],
                    obs_next[i], info[i], policy[i])
        return self._main_buf_update()

    def _main_buf_update(self):
        lens = np.zeros((self.cached_bufs_n, ), dtype = np.int)
        rews = np.zeros((self.cached_bufs_n, ))
        start_indexs = np.zeros((self.cached_bufs_n, ), dtype = np.int)
        for i, buf in enumerate(self.cached_bufs):
            if buf.done[buf._index - 1] > 0:
                lens[i] = len(buf)
                rews[i] = np.sum(buf.rew[:lens[i]])
                start_indexs[i] = self.main_buf._index
                if self.main_buf._maxsize > 0:
                    # _maxsize of main_buf might be 0 in test collector.
                    self.main_buf.update(buf)
                buf.reset()
        return lens, rews, start_indexs

    def reset(self) -> None:
        for buf in self.cached_bufs:
            buf.reset()
        self.main_buf.reset()
        self._avail_index = []
        #TODO finish

    def sample(self, batch_size: int,
               is_from_main_buf = False) -> Tuple[Batch, np.ndarray]:
        if is_from_main_buf:
            return self.main_buf.sample(batch_size)

        _all = np.arange(len(self), dtype=np.int)
        start = len(self.main_buf)
        add = self.main_buf._maxsize - len(self.main_buf)
        for buf in self.cached_bufs:
            end = start + len(buf)
            _all[start:end] =  _all[start:end] + add
            start = end
            add = add + buf._maxsize - len(buf)
        indice = np.random.choice(_all, batch_size)
        assert len(indice) > 0, "No available indice can be sampled."
        return self[indice], indice

    def get(
        self,
        indice: Union[slice, int, np.integer, np.ndarray],
        key: str,
        stack_num: Optional[int] = None,
    ) -> Union[Batch, np.ndarray]:
        if stack_num is None:
            stack_num = self.stack_num
        assert(stack_num == 1)
        #TODO support stack
        return super().get(indice, key, stack_num)

    def _cache_initialise(
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
        super().reset() #TODO delete useless varible?
        del self._index
        del self._size
        self.main_buf.set_batch(self._meta[:self.main_buf._maxsize])
        start = self.main_buf._maxsize
        for buf in self.cached_bufs:
            end = start + buf._maxsize
            buf.set_batch(self._meta[start: end])
            start = end
    
    def ends(self):
        return np.concatenate(
            [self.main_buf.ends(), *[b.ends() for b in self.cached_bufs]])

    def starts(self):
        return np.concatenate(
            [self.main_buf.starts(), *[b.starts() for b in self.cached_bufs]])

    def next(self, index):
        assert index >= 0 and index < self._maxsize, "Input index illegal."
        all_buffer = [self.main_buf, *self.cached_bufs]
        ret = index.copy()
        upper = 0
        lower = 0
        for b in all_buffer:
            lower = upper
            upper += b._maxsize
            mask = ret>=lower and ret<upper
            ret[mask] = b.next(ret[mask]-lower)+lower
        return ret

    def start(self, index):
        """return start indices of given indices"""
        all_buffer = [self.main_buf, *self.cached_bufs]
        ret = np.full(index.shape, -1)
        upper = 0
        lower = 0
        for b in all_buffer:
            lower = upper
            upper += b._maxsize
            mask = ret>=lower and ret<upper
            ret[mask] = b.start(ret[mask]-lower)+lower
        return ret

    def _global2local(self, global_index):
        assert (global_index>=0 and global_index<self._maxsize).all()
        all_buffer = [self.main_buf, *self.cached_bufs]
        local_index = global_index.copy()
        group_index = np.full(global_index.shape, -1)
        upper = 0
        lower = 0
        for i, b in enumerate(all_buffer):
            lower = upper
            upper += b._maxsize
            mask = global_index>=lower and global_index<upper
            group_index[mask] = i
            local_index[mask]-=lower
        return local_index, group_index

    def _local2global(self, local_index, group_index):
        global_index = local_index.copy()
        lowers = [0, self.main_buf._maxsize]
        for i in range(self.cached_bufs_n -1):
            lowers.append(lowers[-1] + self.cached_bufs[0]._max_size)
        for i in range(len(lowers)):
            global_index[group_index == i]+=lowers[i]
        return global_index
