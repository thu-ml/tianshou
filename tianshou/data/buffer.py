import numpy as np
from tianshou.data.batch import Batch


class ReplayBuffer(object):
    """:class:`~tianshou.data.ReplayBuffer` stores data generated from
    interaction between the policy and environment. It stores basically 6 types
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

    From version v0.2.2, :class:`~tianshou.data.ReplayBuffer` supports
    frame_stack sampling, typically for RNN usage:
    ::

        >>> buf = ReplayBuffer(size=9, stack_num=4)
        >>> for i in range(16):
        ...     done = i % 5 == 0
        ...     buf.add(obs=i, act=i, rew=i, done=done, obs_next=0, info={})
        >>> print(buf)
        ReplayBuffer(
            obs: [ 9. 10. 11. 12. 13. 14. 15.  7.  8.],
            act: [ 9. 10. 11. 12. 13. 14. 15.  7.  8.],
            rew: [ 9. 10. 11. 12. 13. 14. 15.  7.  8.],
            done: [0. 1. 0. 0. 0. 0. 1. 0. 0.],
            obs_next: [0. 0. 0. 0. 0. 0. 0. 0. 0.],
            info: [{} {} {} {} {} {} {} {} {}],
        )
        >>> index = np.arange(len(buf))
        >>> print(buf.get_stack(index, 'obs'))
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
        >>> abs(buf.get_stack(index, 'obs') - buf[index].obs).sum().sum()
        0.0
    """

    def __init__(self, size, stack_num=0):
        super().__init__()
        self._maxsize = size
        self._stack = stack_num
        self.reset()

    def __len__(self):
        """Return len(self)."""
        return self._size

    def __repr__(self):
        """Return str(self)."""
        s = self.__class__.__name__ + '(\n'
        flag = False
        for k in self.__dict__.keys():
            if k[0] != '_' and self.__dict__[k] is not None:
                rpl = '\n' + ' ' * (6 + len(k))
                obj = str(self.__dict__[k]).replace('\n', rpl)
                s += f'    {k}: {obj},\n'
                flag = True
        if flag:
            s += ')\n'
        else:
            s = self.__class__.__name__ + '()\n'
        return s

    def _add_to_buffer(self, name, inst):
        if inst is None:
            if getattr(self, name, None) is None:
                self.__dict__[name] = None
            return
        if self.__dict__.get(name, None) is None:
            if isinstance(inst, np.ndarray):
                self.__dict__[name] = np.zeros([self._maxsize, *inst.shape])
            elif isinstance(inst, dict):
                self.__dict__[name] = np.array(
                    [{} for _ in range(self._maxsize)])
            else:  # assume `inst` is a number
                self.__dict__[name] = np.zeros([self._maxsize])
        if isinstance(inst, np.ndarray) and \
                self.__dict__[name].shape[1:] != inst.shape:
            self.__dict__[name] = np.zeros([self._maxsize, *inst.shape])
        self.__dict__[name][self._index] = inst

    def update(self, buffer):
        """Move the data from the given buffer to self."""
        i = begin = buffer._index % len(buffer)
        while True:
            self.add(
                buffer.obs[i], buffer.act[i], buffer.rew[i], buffer.done[i],
                None if buffer.obs_next is None else buffer.obs_next[i],
                buffer.info[i])
            i = (i + 1) % len(buffer)
            if i == begin:
                break

    def add(self, obs, act, rew, done, obs_next=None, info={}, weight=None):
        """Add a batch of data into replay buffer."""
        assert isinstance(info, dict), \
            'You should return a dict in the last argument of env.step().'
        self._add_to_buffer('obs', obs)
        self._add_to_buffer('act', act)
        self._add_to_buffer('rew', rew)
        self._add_to_buffer('done', done)
        self._add_to_buffer('obs_next', obs_next)
        self._add_to_buffer('info', info)
        if self._maxsize > 0:
            self._size = min(self._size + 1, self._maxsize)
            self._index = (self._index + 1) % self._maxsize
        else:
            self._size = self._index = self._index + 1

    def reset(self):
        """Clear all the data in replay buffer."""
        self._index = self._size = 0

    def sample(self, batch_size):
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

    def get_stack(self, indice, key):
        """Return the stacked result, e.g. [s_{t-3}, s_{t-2}, s_{t-1}, s_t],
        where s is self.key, t is indice. The stack_num (here equals to 4) is
        given from buffer initialization procedure.
        """
        if self.__dict__.get(key, None) is None:
            return None
        if self._stack == 0:
            return self.__dict__[key][indice]
        stack = []
        # set last frame done to True
        last_index = (self._index - 1 + self._size) % self._size
        last_done, self.done[last_index] = self.done[last_index], True
        for i in range(self._stack):
            stack = [self.__dict__[key][indice]] + stack
            pre_indice = indice - 1
            pre_indice[pre_indice == -1] = self._size - 1
            indice = pre_indice + self.done[pre_indice].astype(np.int)
            indice[indice == self._size] = 0
        self.done[last_index] = last_done
        return np.stack(stack, axis=1)

    def __getitem__(self, index):
        """Return a data batch: self[index]. If stack_num is set to be > 0,
        return the stacked obs and obs_next with shape [batch, len, ...].
        """
        return Batch(
            obs=self.get_stack(index, 'obs'),
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            obs_next=self.get_stack(index, 'obs_next'),
            info=self.info[index]
        )


class ListReplayBuffer(ReplayBuffer):
    """The function of :class:`~tianshou.data.ListReplayBuffer` is almost the
    same as :class:`~tianshou.data.ReplayBuffer`. The only difference is that
    :class:`~tianshou.data.ListReplayBuffer` is based on ``list``.

    .. seealso::

        Please refer to :class:`~tianshou.data.ListReplayBuffer` for more
        detailed explanation.
    """

    def __init__(self):
        super().__init__(size=0)

    def _add_to_buffer(self, name, inst):
        if inst is None:
            return
        if self.__dict__.get(name, None) is None:
            self.__dict__[name] = []
        self.__dict__[name].append(inst)

    def reset(self):
        self._index = self._size = 0
        for k in list(self.__dict__.keys()):
            if not k.startswith('_'):
                self.__dict__[k] = []


class PrioritizedReplayBuffer(ReplayBuffer):
    """docstring for PrioritizedReplayBuffer"""

    def __init__(self, size):
        super().__init__(size)

    def add(self, obs, act, rew, done, obs_next=0, info={}, weight=None):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
