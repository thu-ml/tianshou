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

    def __init__(
        self,
        size: int,
        stack_num: int = 1,
        ignore_obs_next: bool = False,
        save_only_last_obs: bool = False,
        sample_avail: bool = False,
    ) -> None:
        super().__init__()
        self._maxsize = size
        self._indices = np.arange(size)
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

    def _add_to_buffer(self, name: str, inst: Any) -> None:
        try:
            value = self._meta.__dict__[name]
        except KeyError:
            self._meta.__dict__[name] = _create_value(inst, self._maxsize)
            value = self._meta.__dict__[name]
        if isinstance(inst, (torch.Tensor, np.ndarray)):
            if inst.shape != value.shape[1:]:
                raise ValueError(
                    "Cannot add data to a buffer with different shape with key"
                    f" {name}, expect {value.shape[1:]}, given {inst.shape}."
                )
        try:
            value[self._index] = inst
        except ValueError:
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
        """Add a batch of data into replay buffer."""
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
        warnings.warn("ListReplayBuffer will be removed in version 0.4.0.")

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
        new_weight: Union[np.ndarray, torch.Tensor]
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
