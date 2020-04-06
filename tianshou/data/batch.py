import torch
import numpy as np


class Batch(object):
    """Tianshou provides :class:`~tianshou.data.Batch` as the internal data
    structure to pass any kind of data to other methods, for example, a
    collector gives a :class:`~tianshou.data.Batch` to policy for learning.
    Here is the usage:
    ::

        >>> import numpy as np
        >>> from tianshou.data import Batch
        >>> data = Batch(a=4, b=[5, 5], c='2312312')
        >>> data.b
        [5, 5]
        >>> data.b = np.array([3, 4, 5])
        >>> len(data.b)
        3
        >>> data.b[-1]
        5

    In short, you can define a :class:`Batch` with any key-value pair. The
    current implementation of Tianshou typically use 6 keys in
    :class:`~tianshou.data.Batch`:

    * ``obs`` the observation of step :math:`t` ;
    * ``act`` the action of step :math:`t` ;
    * ``rew`` the reward of step :math:`t` ;
    * ``done`` the done flag of step :math:`t` ;
    * ``obs_next`` the observation of step :math:`t+1` ;
    * ``info`` the info of step :math:`t` (in ``gym.Env``, the ``env.step()``\
        function return 4 arguments, and the last one is ``info``);

    :class:`~tianshou.data.Batch` has other methods, including
    :meth:`~tianshou.data.Batch.__getitem__`,
    :meth:`~tianshou.data.Batch.__len__`,
    :meth:`~tianshou.data.Batch.append`,
    and :meth:`~tianshou.data.Batch.split`:
    ::

        >>> data = Batch(obs=np.array([0, 11, 22]), rew=np.array([6, 6, 6]))
        >>> # here we test __getitem__
        >>> index = [2, 1]
        >>> data[index].obs
        array([22, 11])

        >>> # here we test __len__
        >>> len(data)
        3

        >>> data.append(data)  # similar to list.append
        >>> data.obs
        array([0, 11, 22, 0, 11, 22])

        >>> # split whole data into multiple small batch
        >>> for d in data.split(size=2, permute=False):
        ...     print(d.obs, d.rew)
        [ 0 11] [6 6]
        [22  0] [6 6]
        [11 22] [6 6]
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def __getitem__(self, index):
        """Return self[index]."""
        b = Batch()
        for k in self.__dict__.keys():
            if self.__dict__[k] is not None:
                b.__dict__.update(**{k: self.__dict__[k][index]})
        return b

    def append(self, batch):
        """Append a :class:`~tianshou.data.Batch` object to current batch."""
        assert isinstance(batch, Batch), 'Only append Batch is allowed!'
        for k in batch.__dict__.keys():
            if batch.__dict__[k] is None:
                continue
            if not hasattr(self, k) or self.__dict__[k] is None:
                self.__dict__[k] = batch.__dict__[k]
            elif isinstance(batch.__dict__[k], np.ndarray):
                self.__dict__[k] = np.concatenate([
                    self.__dict__[k], batch.__dict__[k]])
            elif isinstance(batch.__dict__[k], torch.Tensor):
                self.__dict__[k] = torch.cat([
                    self.__dict__[k], batch.__dict__[k]])
            elif isinstance(batch.__dict__[k], list):
                self.__dict__[k] += batch.__dict__[k]
            else:
                s = 'No support for append with type'\
                    + str(type(batch.__dict__[k]))\
                    + 'in class Batch.'
                raise TypeError(s)

    def __len__(self):
        """Return len(self)."""
        return min([
            len(self.__dict__[k]) for k in self.__dict__.keys()
            if self.__dict__[k] is not None])

    def split(self, size=None, permute=True):
        """Split whole data into multiple small batch.

        :param int size: if it is ``None``, it does not split the data batch;
            otherwise it will divide the data batch with the given size.
            Default to ``None``.
        :param bool permute: randomly shuffle the entire data batch if it is
            ``True``, otherwise remain in the same. Default to ``True``.
        """
        length = len(self)
        if size is None:
            size = length
        temp = 0
        if permute:
            index = np.random.permutation(length)
        else:
            index = np.arange(length)
        while temp < length:
            yield self[index[temp:temp + size]]
            temp += size
