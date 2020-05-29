import torch
import pprint
import numpy as np
from typing import Any, List, Union, Iterator, Optional


class Batch:
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
        >>> print(data)
        Batch(
            a: 4,
            b: array([3, 4, 5]),
            c: '2312312',
        )

    In short, you can define a :class:`Batch` with any key-value pair. The
    current implementation of Tianshou typically use 7 reserved keys in
    :class:`~tianshou.data.Batch`:

    * ``obs`` the observation of step :math:`t` ;
    * ``act`` the action of step :math:`t` ;
    * ``rew`` the reward of step :math:`t` ;
    * ``done`` the done flag of step :math:`t` ;
    * ``obs_next`` the observation of step :math:`t+1` ;
    * ``info`` the info of step :math:`t` (in ``gym.Env``, the ``env.step()``\
        function return 4 arguments, and the last one is ``info``);
    * ``policy`` the data computed by policy in step :math:`t`;

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
        >>> for d in data.split(size=2, shuffle=False):
        ...     print(d.obs, d.rew)
        [ 0 11] [6 6]
        [22  0] [6 6]
        [11 22] [6 6]
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._meta = {}
        for k, v in kwargs.items():
            if isinstance(v, (list, np.ndarray)) \
                    and len(v) > 0 and isinstance(v[0], dict) and k != 'info':
                self._meta[k] = list(v[0].keys())
                for k_ in v[0].keys():
                    k__ = '_' + k + '@' + k_
                    self.__dict__[k__] = np.array([
                        v[i][k_] for i in range(len(v))
                    ])
            elif isinstance(v, dict):
                self._meta[k] = list(v.keys())
                for k_, v_ in v.items():
                    k__ = '_' + k + '@' + k_
                    self.__dict__[k__] = v_
            else:
                self.__dict__[k] = kwargs[k]

    def __getitem__(self, index: Union[str, slice]) -> Union['Batch', dict]:
        """Return self[index]."""
        if isinstance(index, str):
            return self.__getattr__(index)
        b = Batch()
        for k, v in self.__dict__.items():
            if k != '_meta' and v is not None:
                b.__dict__.update(**{k: v[index]})
        b._meta = self._meta
        return b

    def __getattr__(self, key: str) -> Union['Batch', Any]:
        """Return self.key"""
        if key not in self._meta.keys():
            if key not in self.__dict__:
                raise AttributeError(key)
            return self.__dict__[key]
        d = {}
        for k_ in self._meta[key]:
            k__ = '_' + key + '@' + k_
            d[k_] = self.__dict__[k__]
        return Batch(**d)

    def __repr__(self) -> str:
        """Return str(self)."""
        s = self.__class__.__name__ + '(\n'
        flag = False
        for k in sorted(list(self.__dict__) + list(self._meta)):
            if k[0] != '_' and (self.__dict__.get(k, None) is not None or
                                k in self._meta):
                rpl = '\n' + ' ' * (6 + len(k))
                obj = pprint.pformat(self.__getattr__(k)).replace('\n', rpl)
                s += f'    {k}: {obj},\n'
                flag = True
        if flag:
            s += ')'
        else:
            s = self.__class__.__name__ + '()'
        return s

    def keys(self) -> List[str]:
        """Return self.keys()."""
        return sorted(list(self._meta.keys()) +
                      [k for k in self.__dict__.keys() if k[0] != '_'])

    def values(self) -> List[Any]:
        """Return self.values()."""
        return [self[k] for k in self.keys()]

    def get(self, k: str, d: Optional[Any] = None) -> Union['Batch', Any]:
        """Return self[k] if k in self else d. d defaults to None."""
        if k in self.__dict__ or k in self._meta:
            return self.__getattr__(k)
        return d

    def to_numpy(self) -> np.ndarray:
        """Change all torch.Tensor to numpy.ndarray. This is an inplace
        operation.
        """
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.cpu().numpy()

    def append(self, batch: 'Batch') -> None:
        """Append a :class:`~tianshou.data.Batch` object to current batch."""
        assert isinstance(batch, Batch), 'Only append Batch is allowed!'
        for k, v in batch.__dict__.items():
            if k == '_meta':
                self._meta.update(batch._meta)
                continue
            if v is None:
                continue
            if not hasattr(self, k) or self.__dict__[k] is None:
                self.__dict__[k] = v
            elif isinstance(v, np.ndarray):
                self.__dict__[k] = np.concatenate([self.__dict__[k], v])
            elif isinstance(v, torch.Tensor):
                self.__dict__[k] = torch.cat([self.__dict__[k], v])
            elif isinstance(v, list):
                self.__dict__[k] += v
            else:
                s = f'No support for append with type \
                      {type(v)} in class Batch.'
                raise TypeError(s)

    def __len__(self) -> int:
        """Return len(self)."""
        return min([len(v) for k, v in self.__dict__.items()
                    if k != '_meta' and v is not None])

    def split(self, size: Optional[int] = None,
              shuffle: bool = True) -> Iterator['Batch']:
        """Split whole data into multiple small batch.

        :param int size: if it is ``None``, it does not split the data batch;
            otherwise it will divide the data batch with the given size.
            Default to ``None``.
        :param bool shuffle: randomly shuffle the entire data batch if it is
            ``True``, otherwise remain in the same. Default to ``True``.
        """
        length = len(self)
        if size is None:
            size = length
        if shuffle:
            indices = np.random.permutation(length)
        else:
            indices = np.arange(length)
        for idx in np.arange(0, length, size):
            yield self[indices[idx:(idx + size)]]
