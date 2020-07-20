import numpy as np
from typing import Any, Tuple, Union, Optional

from tianshou.data.batch import Batch, _create_value


class ReplayBuffer:
    """:class:`~tianshou.data.ReplayBuffer` stores data generated from
    interaction between the policy and environment. The current implementation
    of Tianshou typically use 7 reserved keys in :class:`~tianshou.data.Batch`:

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

        >>> import numpy as np
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
        >>> # the batch_data is equal to buf[incide].
        >>> batch_data, indice = buf.sample(batch_size=4)
        >>> batch_data.obs == buf[indice].obs
        array([ True,  True,  True,  True])

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
        than 1, defaults to 0 (no stacking).
    :param bool ignore_obs_next: whether to store obs_next, defaults to
        ``False``.
    :param bool sample_avail: the parameter indicating sampling only available
        index when using frame-stack sampling method, defaults to ``False``.
        This feature is not supported in Prioritized Replay Buffer currently.
    """

    def __init__(self, size: int, stack_num: Optional[int] = 0,
                 ignore_obs_next: bool = False,
                 sample_avail: bool = False, **kwargs) -> None:
        super().__init__()
        self._maxsize = size
        self._stack = stack_num
        assert stack_num != 1, 'stack_num should greater than 1'
        self._avail = sample_avail and stack_num > 1
        self._avail_index = []
        self._save_s_ = not ignore_obs_next
        self._index = 0
        self._size = 0
        self._meta = Batch()
        self.reset()

    def __len__(self) -> int:
        """Return len(self)."""
        return self._size

    def __repr__(self) -> str:
        """Return str(self)."""
        return self.__class__.__name__ + self._meta.__repr__()[5:]

    def __getattr__(self, key: str) -> Union['Batch', Any]:
        """Return self.key"""
        return self._meta.__dict__[key]

    def _add_to_buffer(self, name: str, inst: Any) -> None:
        try:
            value = self._meta.__dict__[name]
        except KeyError:
            self._meta.__dict__[name] = _create_value(inst, self._maxsize)
            value = self._meta.__dict__[name]
        if isinstance(inst, np.ndarray) and value.shape[1:] != inst.shape:
            raise ValueError(
                "Cannot add data to a buffer with different shape, key: "
                f"{name}, expect shape: {value.shape[1:]}, "
                f"given shape: {inst.shape}.")
        try:
            value[self._index] = inst
        except KeyError:
            for key in set(inst.keys()).difference(value.__dict__.keys()):
                value.__dict__[key] = _create_value(inst[key], self._maxsize)
            value[self._index] = inst

    def _get_stack_num(self):
        return self._stack

    def _set_stack_num(self, num):
        self._stack = num

    def update(self, buffer: 'ReplayBuffer') -> None:
        """Move the data from the given buffer to self."""
        if len(buffer) == 0:
            return
        i = begin = buffer._index % len(buffer)
        origin = buffer._get_stack_num()
        buffer._set_stack_num(0)
        while True:
            self.add(**buffer[i])
            i = (i + 1) % len(buffer)
            if i == begin:
                break
        buffer._set_stack_num(origin)

    def add(self,
            obs: Union[dict, Batch, np.ndarray],
            act: Union[np.ndarray, float],
            rew: Union[int, float],
            done: bool,
            obs_next: Optional[Union[dict, Batch, np.ndarray]] = None,
            info: dict = {},
            policy: Optional[Union[dict, Batch]] = {},
            **kwargs) -> None:
        """Add a batch of data into replay buffer."""
        assert isinstance(info, (dict, Batch)), \
            'You should return a dict in the last argument of env.step().'
        self._add_to_buffer('obs', obs)
        self._add_to_buffer('act', act)
        self._add_to_buffer('rew', rew)
        self._add_to_buffer('done', done)
        if self._save_s_:
            if obs_next is None:
                obs_next = Batch()
            self._add_to_buffer('obs_next', obs_next)
        self._add_to_buffer('info', info)
        self._add_to_buffer('policy', policy)

        # maintain available index for frame-stack sampling
        if self._avail:
            # update current frame
            avail = sum(self.done[i] for i in range(
                self._index - self._stack + 1, self._index)) == 0
            if self._size < self._stack - 1:
                avail = False
            if avail and self._index not in self._avail_index:
                self._avail_index.append(self._index)
            elif not avail and self._index in self._avail_index:
                self._avail_index.remove(self._index)
            # remove the later available frame because of broken storage
            t = (self._index + self._stack - 1) % self._maxsize
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
        """Get a random sample from buffer with size equal to batch_size. \
        Return all the data in the buffer if batch_size is ``0``.

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
        assert len(indice) > 0, 'No available indice can be sampled.'
        return self[indice], indice

    def get(self, indice: Union[slice, int, np.integer, np.ndarray], key: str,
            stack_num: Optional[int] = None) -> Union[Batch, np.ndarray]:
        """Return the stacked result, e.g. [s_{t-3}, s_{t-2}, s_{t-1}, s_t],
        where s is self.key, t is indice. The stack_num (here equals to 4) is
        given from buffer initialization procedure.
        """
        if stack_num is None:
            stack_num = self._stack
        if isinstance(indice, slice):
            indice = np.arange(
                0 if indice.start is None
                else self._size - indice.start if indice.start < 0
                else indice.start,
                self._size if indice.stop is None
                else self._size - indice.stop if indice.stop < 0
                else indice.stop,
                1 if indice.step is None else indice.step)
        else:
            indice = np.array(indice, copy=True)
        # set last frame done to True
        last_index = (self._index - 1 + self._size) % self._size
        last_done, self.done[last_index] = self.done[last_index], True
        if key == 'obs_next' and (not self._save_s_ or self.obs_next is None):
            indice += 1 - self.done[indice].astype(np.int)
            indice[indice == self._size] = 0
            key = 'obs'
        val = self._meta.__dict__[key]
        try:
            if stack_num > 0:
                stack = []
                for _ in range(stack_num):
                    stack = [val[indice]] + stack
                    pre_indice = np.asarray(indice - 1)
                    pre_indice[pre_indice == -1] = self._size - 1
                    indice = np.asarray(
                        pre_indice + self.done[pre_indice].astype(np.int))
                    indice[indice == self._size] = 0
                if isinstance(val, Batch):
                    stack = Batch.stack(stack, axis=indice.ndim)
                else:
                    stack = np.stack(stack, axis=indice.ndim)
            else:
                stack = val[indice]
        except IndexError as e:
            stack = Batch()
            if not isinstance(val, Batch) or len(val.__dict__) > 0:
                raise e
        self.done[last_index] = last_done
        return stack

    def __getitem__(self, index: Union[
            slice, int, np.integer, np.ndarray]) -> Batch:
        """Return a data batch: self[index]. If stack_num is set to be > 0,
        return the stacked obs and obs_next with shape [batch, len, ...].
        """
        return Batch(
            obs=self.get(index, 'obs'),
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            obs_next=self.get(index, 'obs_next'),
            info=self.get(index, 'info'),
            policy=self.get(index, 'policy')
        )


class ListReplayBuffer(ReplayBuffer):
    """The function of :class:`~tianshou.data.ListReplayBuffer` is almost the
    same as :class:`~tianshou.data.ReplayBuffer`. The only difference is that
    :class:`~tianshou.data.ListReplayBuffer` is based on ``list``. Therefore,
    it does not support advanced indexing, which means you cannot sample a
    batch of data out of it. It is typically used for storing data.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more
        detailed explanation.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(size=0, ignore_obs_next=False, **kwargs)

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        raise NotImplementedError("ListReplayBuffer cannot be sampled!")

    def _add_to_buffer(
            self, name: str,
            inst: Union[dict, Batch, np.ndarray, float, int, bool]) -> None:
        if inst is None:
            return
        if self._meta.__dict__.get(name, None) is None:
            self._meta.__dict__[name] = []
        self._meta.__dict__[name].append(inst)

    def reset(self) -> None:
        self._index = self._size = 0
        for k in list(self._meta.__dict__.keys()):
            if isinstance(self._meta.__dict__[k], list):
                self._meta.__dict__[k] = []


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized replay buffer implementation.

    :param float alpha: the prioritization exponent.
    :param float beta: the importance sample soft coefficient.
    :param str mode: defaults to ``weight``.
    :param bool replace: whether to sample with replacement

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more
        detailed explanation.
    """

    def __init__(self, size: int, alpha: float, beta: float,
                 mode: str = 'weight',
                 replace: bool = False, **kwargs) -> None:
        if mode != 'weight':
            raise NotImplementedError
        super().__init__(size, **kwargs)
        self._alpha = alpha
        self._beta = beta
        self._weight_sum = 0.0
        self._amortization_freq = 50
        self._replace = replace
        self._meta.weight = np.zeros(size, dtype=np.float64)

    def add(self,
            obs: Union[dict, np.ndarray],
            act: Union[np.ndarray, float],
            rew: Union[int, float],
            done: bool,
            obs_next: Optional[Union[dict, np.ndarray]] = None,
            info: dict = {},
            policy: Optional[Union[dict, Batch]] = {},
            weight: float = 1.0,
            **kwargs) -> None:
        """Add a batch of data into replay buffer."""
        # we have to sacrifice some convenience for speed
        self._weight_sum += np.abs(weight) ** self._alpha - \
            self._meta.weight[self._index]
        self._add_to_buffer('weight', np.abs(weight) ** self._alpha)
        super().add(obs, act, rew, done, obs_next, info, policy)

    @property
    def replace(self):
        return self._replace

    @replace.setter
    def replace(self, v: bool):
        self._replace = v

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with priority probability. \
        Return all the data in the buffer if batch_size is ``0``.

        :return: Sample data and its corresponding index inside the buffer.
        """
        assert self._size > 0, 'cannot sample a buffer with size == 0 !'
        p = None
        if batch_size > 0 and (self._replace or batch_size <= self._size):
            # sampling weight
            p = (self.weight / self.weight.sum())[:self._size]
            indice = np.random.choice(
                self._size, batch_size, p=p,
                replace=self._replace)
            p = p[indice]  # weight of each sample
        elif batch_size == 0:
            p = np.full(shape=self._size, fill_value=1.0 / self._size)
            indice = np.concatenate([
                np.arange(self._index, self._size),
                np.arange(0, self._index),
            ])
        else:
            raise ValueError(
                f"batch_size should be less than {len(self)}, \
                    or set replace=True")
        batch = self[indice]
        batch["impt_weight"] = (self._size * p) ** (-self._beta)
        return batch, indice

    def update_weight(self, indice: Union[slice, np.ndarray],
                      new_weight: np.ndarray) -> None:
        """Update priority weight by indice in this buffer.

        :param np.ndarray indice: indice you want to update weight
        :param np.ndarray new_weight: new priority weight you want to update
        """
        if self._replace:
            if isinstance(indice, slice):
                # convert slice to ndarray
                indice = np.arange(indice.stop)[indice]
            # remove the same values in indice
            indice, unique_indice = np.unique(
                indice, return_index=True)
            new_weight = new_weight[unique_indice]
        self.weight[indice] = np.power(np.abs(new_weight), self._alpha)

    def __getitem__(self, index: Union[
            slice, int, np.integer, np.ndarray]) -> Batch:
        return Batch(
            obs=self.get(index, 'obs'),
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            obs_next=self.get(index, 'obs_next'),
            info=self.get(index, 'info'),
            weight=self.weight[index],
            policy=self.get(index, 'policy'),
        )
