import numpy as np
from numbers import Number
from typing import Any, Tuple, Union, Optional

from .batch import Batch


class ReplayBuffer(Batch):
    """:class:`~tianshou.data.ReplayBuffer` stores data generated from
    interaction between the policy and environment. It stores basically 7 types
    of data, as mentioned in :class:`~tianshou.data.Batch`, based on
    ``numpy.ndarray``. Here is the usage:
    ::

        >>> import numpy as np
        >>> from tianshou.data import ReplayBuffer
        >>> buf = ReplayBuffer(size=20)
        >>> for i in range(3):
        ...     buf.add(obs=i, act=i, rew=i, done=i, obs_next=i + 1, info={})
        >>> len(buf)
        3
        >>> buf.obs
        # since we set size = 20, len(buf.obs) == 20.
        array([0., 1., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0.])

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
    """

    def __init__(self, size: int, stack_num: Optional[int] = 0,
                 ignore_obs_next: bool = False, **kwargs) -> None:
        super().__init__()
        self.__dict__['_maxsize'] = size
        self.__dict__['_stack'] = stack_num
        self.__dict__['_save_s_'] = not ignore_obs_next
        self.__dict__['_index'] = 0
        self.__dict__['_size'] = 0
        self.reset()

    def __len__(self) -> int:
        """Return len(self)."""
        return self._size

    def _add_to_buffer(self, name: str, inst: Any) -> None:
        def _create_value(inst: Any) -> Union['Batch', np.ndarray]:
            if isinstance(inst, np.ndarray):
                return np.zeros(
                    (self._maxsize, *inst.shape), dtype=inst.dtype)
            elif isinstance(inst, (dict, Batch)):
                return Batch([Batch(inst) for _ in range(self._maxsize)])
            elif isinstance(inst, (np.generic, Number)):
                return np.zeros(
                    (self._maxsize,), dtype=np.asarray(inst).dtype)
            else:  # fall back to np.object
                return np.array([None for _ in range(self._maxsize)])

        if inst is None:
            inst = Batch()
        if name not in self.keys():
            self._data[name] = _create_value(inst)
        if isinstance(inst, np.ndarray) and \
                self._data[name].shape[1:] != inst.shape:
            raise ValueError(
                "Cannot add data to a buffer with different shape, "
                f"key: {name}, expect shape: {self._data[name].shape[1:]}"
                f", given shape: {inst.shape}.")
        if isinstance(self._data[name], Batch):
            field_keys = self._data[name].keys()
            for key, val in inst.items():
                if key not in field_keys:
                    self._data[name][key] = _create_value(val)
        self._data[name][self._index] = inst

    def update(self, buffer: 'ReplayBuffer') -> None:
        """Move the data from the given buffer to self."""
        i = begin = buffer._index % len(buffer)
        while True:
            self.add(**buffer[i])
            i = (i + 1) % len(buffer)
            if i == begin:
                break

    def add(self,
            obs: Union[dict, np.ndarray],
            act: Union[np.ndarray, float],
            rew: float,
            done: bool,
            obs_next: Optional[Union[dict, np.ndarray]] = None,
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
            self._add_to_buffer('obs_next', obs_next)
        self._add_to_buffer('info', info)
        self._add_to_buffer('policy', policy)
        if self._maxsize > 0:
            self._size = min(self._size + 1, self._maxsize)
            self._index = (self._index + 1) % self._maxsize
        else:
            self._size = self._index = self._index + 1

    def reset(self) -> None:
        """Clear all the data in replay buffer."""
        self._index = 0
        self._size = 0

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with size equal to batch_size. \
        Return all the data in the buffer if batch_size is ``0``.

        :return: Sample data and its corresponding index inside the buffer.
        """
        if batch_size > 0:
            indice = np.random.choice(self._size, batch_size)
        else:
            indice = np.concatenate([
                np.arange(self._index, self._size),
                np.arange(0, self._index),
            ])
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
        indice = np.array(indice, copy=True)
        # set last frame done to True
        last_index = (self._index - 1 + self._size) % self._size
        last_done, self.done[last_index] = self.done[last_index], True
        if key == 'obs_next' and (not self._save_s_ or self.obs_next is None):
            indice += 1 - self.done[indice].astype(np.int)
            indice[indice == self._size] = 0
            key = 'obs'
        if stack_num == 0:
            self.done[last_index] = last_done
            val = self._data[key]
            if isinstance(val, Batch) and val.size == 0:
                return val
            else:
                if isinstance(indice, (int, np.integer)) or \
                        (isinstance(indice, np.ndarray) and
                            indice.ndim == 0) or not isinstance(val, list):
                    return val[indice]
                else:
                    return [val[i] for i in indice]
        else:
            val = self._data[key]
            if not isinstance(val, Batch) or val.size > 0:
                stack = []
                for _ in range(stack_num):
                    stack = [val[indice]] + stack
                    pre_indice = np.asarray(indice - 1)
                    pre_indice[pre_indice == -1] = self._size - 1
                    indice = np.asarray(
                        pre_indice + self.done[pre_indice].astype(np.int))
                    indice[indice == self._size] = 0
                if isinstance(stack[0], Batch):
                    stack = Batch.stack(stack, axis=indice.ndim)
                else:
                    stack = np.stack(stack, axis=indice.ndim)
            else:
                stack = Batch()
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
            # act_=self.get(index, 'act'),  # stacked action, for RNN
            rew=self.rew[index],
            done=self.done[index],
            obs_next=self.get(index, 'obs_next'),
            info=self.get(index, 'info', stack_num=0),
            policy=self.get(index, 'policy'),
        )


class ListReplayBuffer(ReplayBuffer):
    """The function of :class:`~tianshou.data.ListReplayBuffer` is almost the
    same as :class:`~tianshou.data.ReplayBuffer`. The only difference is that
    :class:`~tianshou.data.ListReplayBuffer` is based on ``list``.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more
        detailed explanation.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(size=0, ignore_obs_next=False, **kwargs)

    def _add_to_buffer(
            self, name: str,
            inst: Union[dict, Batch, np.ndarray, float, int, bool]) -> None:
        if inst is None:
            return
        if self._data.get(name, None) is None:
            self._data[name] = []
        self._data[name].append(inst)

    def reset(self) -> None:
        self._index = self._size = 0
        for k in list(self._data):
            if isinstance(self._data[k], list):
                self._data[k] = []


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
        self.weight = np.zeros(size, dtype=np.float64)
        self._amortization_freq = 50
        self._amortization_counter = 0
        self._replace = replace

    def add(self,
            obs: Union[dict, np.ndarray],
            act: Union[np.ndarray, float],
            rew: float,
            done: bool,
            obs_next: Optional[Union[dict, np.ndarray]] = None,
            info: dict = {},
            policy: Optional[Union[dict, Batch]] = {},
            weight: float = 1.0,
            **kwargs) -> None:
        """Add a batch of data into replay buffer."""
        self._weight_sum += np.abs(weight) ** self._alpha - \
            self.weight[self._index]
        # we have to sacrifice some convenience for speed :(
        self._add_to_buffer('weight', np.abs(weight) ** self._alpha)
        super().add(obs, act, rew, done, obs_next, info, policy)
        self._check_weight_sum()

    @property
    def replace(self):
        return self._replace

    @replace.setter
    def replace(self, v: bool):
        self._replace = v

    def sample(self, batch_size: int,
               importance_sample: bool = True
               ) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with priority probability. \
        Return all the data in the buffer if batch_size is ``0``.

        :return: Sample data and its corresponding index inside the buffer.
        """
        if batch_size > 0 and batch_size <= self._size:
            # Multiple sampling of the same sample
            # will cause weight update conflict
            indice = np.random.choice(
                self._size, batch_size,
                p=(self.weight / self.weight.sum())[:self._size],
                replace=self._replace)
            # self._weight_sum is not work for the accuracy issue
            # p=(self.weight/self._weight_sum)[:self._size], replace=False)
        elif batch_size == 0:
            indice = np.concatenate([
                np.arange(self._index, self._size),
                np.arange(0, self._index),
            ])
        else:
            # if batch_size larger than len(self),
            # it will lead to a bug in update weight
            raise ValueError(
                "batch_size should be less than len(self), \
                    or set replace=False")
        batch = self[indice]
        if importance_sample:
            impt_weight = Batch(
                impt_weight=1 / np.power(
                    self._size * (batch.weight / self._weight_sum),
                    self._beta))
            batch.cat_(impt_weight)
        self._check_weight_sum()
        return batch, indice

    def reset(self) -> None:
        self._amortization_counter = 0
        super().reset()

    def update_weight(self, indice: Union[slice, np.ndarray],
                      new_weight: np.ndarray) -> None:
        """Update priority weight by indice in this buffer.

        :param np.ndarray indice: indice you want to update weight
        :param np.ndarray new_weight: new priority weight you wangt to update
        """
        if self._replace:
            if isinstance(indice, slice):
                # convert slice to ndarray
                indice = np.arange(indice.stop)[indice]
            # remove the same values in indice
            indice, unique_indice = np.unique(
                indice, return_index=True)
            new_weight = new_weight[unique_indice]
        self._weight_sum += np.power(np.abs(new_weight), self._alpha).sum() \
            - self.weight[indice].sum()
        self.weight[indice] = np.power(np.abs(new_weight), self._alpha)

    def __getitem__(self, index: Union[slice, np.ndarray]) -> Batch:
        return Batch(
            obs=self.get(index, 'obs'),
            act=self.act[index],
            # act_=self.get(index, 'act'),  # stacked action, for RNN
            rew=self.rew[index],
            done=self.done[index],
            obs_next=self.get(index, 'obs_next'),
            info=self.get(index, 'info'),
            weight=self.weight[index],
            policy=self.get(index, 'policy'),
        )

    def _check_weight_sum(self) -> None:
        # keep an accurate _weight_sum
        self._amortization_counter += 1
        if self._amortization_counter % self._amortization_freq == 0:
            self._weight_sum = np.sum(self.weight)
            self._amortization_counter = 0
