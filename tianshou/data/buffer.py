import numpy as np
from tianshou.data.batch import Batch


class ReplayBuffer(object):
    """:class:`~tianshou.data.ReplayBuffer` stores data generated from
    interaction between the policy and environment. It stores basically 6 types
    of data, as mentioned in :class:`~tianshou.data.Batch`, based on
    ``numpy.ndarray``. Here is the usage:
    ::

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
    """

    def __init__(self, size):
        super().__init__()
        self._maxsize = size
        self.reset()

    def __len__(self):
        """Return len(self)."""
        return self._size

    def _add_to_buffer(self, name, inst):
        if inst is None:
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
                buffer.obs[i], buffer.act[i], buffer.rew[i],
                buffer.done[i], buffer.obs_next[i], buffer.info[i])
            i = (i + 1) % len(buffer)
            if i == begin:
                break

    def add(self, obs, act, rew, done, obs_next=0, info={}, weight=None):
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
        self.indice = []

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

    def __getitem__(self, index):
        """Return a data batch: self[index]."""
        return Batch(
            obs=self.obs[index],
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            obs_next=self.obs_next[index],
            info=self.info[index]
        )


class ListReplayBuffer(ReplayBuffer):
    """The function of :class:`~tianshou.data.ListReplayBuffer` is almost the
    same as :class:`~tianshou.data.ReplayBuffer`. The only difference is that
    :class:`~tianshou.data.ListReplayBuffer` is based on ``list``.
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
