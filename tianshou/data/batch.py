import torch
import copy
import pprint
import warnings
import numpy as np
from numbers import Number
from typing import Any, List, Tuple, Union, Iterator, Optional

# Disable pickle warning related to torch, since it has been removed
# on torch master branch. See Pull Request #39003 for details:
# https://github.com/pytorch/pytorch/pull/39003
warnings.filterwarnings(
    "ignore", message="pickle support for Storage will be removed in 1.5.")


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

    def __new__(cls, *args, **kwargs) -> 'Batch':
        self = super().__new__(cls)
        self.__dict__['_data'] = {}
        return self

    def __init__(self,
                 batch_dict: Optional[Union[
                     dict, 'Batch', Tuple[Union[dict, 'Batch']],
                     List[Union[dict, 'Batch']], np.ndarray]] = None,
                 **kwargs) -> None:
        def _is_batch_set(data: Any) -> bool:
            if isinstance(data, (list, tuple)):
                if len(data) > 0 and isinstance(data[0], (dict, Batch)):
                    return True
            elif isinstance(data, np.ndarray):
                if isinstance(data.item(0), (dict, Batch)):
                    return True
            return False

        if isinstance(batch_dict, np.ndarray) and batch_dict.ndim == 0:
            batch_dict = batch_dict[()]
        if _is_batch_set(batch_dict):
            for k, v in zip(batch_dict[0].keys(),
                            zip(*[e.values() for e in batch_dict])):
                if isinstance(v[0], dict) or _is_batch_set(v[0]):
                    self[k] = Batch(v)
                elif isinstance(v[0], (np.generic, np.ndarray)):
                    self[k] = np.stack(v, axis=0)
                elif isinstance(v[0], torch.Tensor):
                    self[k] = torch.stack(v, dim=0)
                elif isinstance(v[0], Batch):
                    self[k] = Batch.stack(v)
                else:
                    self[k] = np.array(v)  # fall back to np.object
        elif isinstance(batch_dict, (dict, Batch)):
            for k, v in batch_dict.items():
                if isinstance(v, dict) or _is_batch_set(v):
                    self[k] = Batch(v)
                else:
                    self[k] = v
        if len(kwargs) > 0:
            self.__init__(kwargs)

    def __getstate__(self):
        """Pickling interface. Only the actual data are serialized
        for both efficiency and simplicity.
        """
        state = {}
        for k in self.keys():
            v = self[k]
            if isinstance(v, Batch):
                v = v.__getstate__()
            state[k] = v
        return state

    def __setstate__(self, state):
        """Unpickling interface. At this point, self is an empty Batch
        instance that has not been initialized, so it can safely be
        initialized by the pickle state.
        """
        self.__init__(**state)

    def __getitem__(self, index: Union[
            str, slice, int, np.integer, np.ndarray, List[int]]) -> 'Batch':
        """Return self[index]."""
        def _valid_bounds(length: int, index: Union[
                slice, int, np.integer, np.ndarray, List[int]]) -> bool:
            if isinstance(index, (int, np.integer)):
                return -length <= index and index < length
            elif isinstance(index, (list, np.ndarray)):
                return _valid_bounds(length, np.min(index)) and \
                    _valid_bounds(length, np.max(index))
            elif isinstance(index, slice):
                if index.start is not None:
                    start_valid = _valid_bounds(length, index.start)
                else:
                    start_valid = True
                if index.stop is not None:
                    stop_valid = _valid_bounds(length, index.stop - 1)
                else:
                    stop_valid = True
                return start_valid and stop_valid

        if isinstance(index, str):
            return getattr(self, index)

        if not _valid_bounds(len(self), index):
            raise IndexError(
                f"Index {index} out of bounds for Batch of len {len(self)}.")
        else:
            b = Batch()
            for k, v in self.items():
                if isinstance(v, Batch) and v.size == 0:
                    b[k] = Batch()
                elif hasattr(v, '__len__') and (not isinstance(
                        v, (np.ndarray, torch.Tensor)) or v.ndim > 0):
                    if _valid_bounds(len(v), index):
                        if isinstance(index, (int, np.integer)) or \
                                (isinstance(index, np.ndarray) and
                                    index.ndim == 0) or \
                                not isinstance(v, list):
                            b[k] = v[index]
                        else:
                            b[k] = [v[i] for i in index]
                    else:
                        raise IndexError(
                            f"Index {index} out of bounds for {type(v)} of "
                            f"len {len(self)}.")
            return b

    def __setitem__(self, index: Union[
                        str, slice, int, np.integer, np.ndarray, List[int]],
                    batch: Union[dict, 'Batch']) -> None:
        for key in self.keys():
            self[key][index] = batch[key]

    def __iadd__(self, val: Union['Batch', Number]):
        if isinstance(val, Batch):
            for k, r, v in zip(self.keys(),
                               self.values(),
                               val.values()):
                if r is None:
                    self[k] = r
                elif isinstance(r, list):
                    self[k] = [r_ + v_ for r_, v_ in zip(r, v)]
                else:
                    self[k] = r + v
            return self
        elif isinstance(val, Number):
            for k, r in zip(self.keys(), self.values()):
                if r is None:
                    self[k] = r
                elif isinstance(r, list):
                    self[k] = [r_ + val for r_ in r]
                else:
                    self[k] = r + val
            return self
        else:
            raise TypeError("Only addition of Batch or number is supported.")

    def __add__(self, val: Union['Batch', Number]):
        return copy.deepcopy(self).__iadd__(val)

    def __mul__(self, val: Number):
        assert isinstance(val, Number), \
            "Only multiplication by a number is supported."
        result = self.__class__()
        for k, r in zip(self.keys(), self.values()):
            result[k] = r * val
        return result

    def __truediv__(self, val: Number):
        assert isinstance(val, Number), \
            "Only division by a number is supported."
        result = self.__class__()
        for k, r in zip(self.keys(), self.values()):
            result[k] = r / val
        return result

    def __getattr__(self, key: str) -> Union['Batch', Any]:
        """Return self.key"""
        if key in self.__dict__.keys():
            return self.__dict__[key]
        elif key in self._data.keys():
            return self._data[key]
        raise AttributeError(key)

    def __setattr__(self, key, value):
        if key in self._data.keys():
            self._data[key] = value
        elif key in self.__dict__.keys():
            self.__dict__[key] = value
        else:
            self._data[key] = value

    def __repr__(self) -> str:
        """Return str(self)."""
        s = self.__class__.__name__ + '(\n'
        flag = False
        for k, v in self.items():
            rpl = '\n' + ' ' * (6 + len(k))
            obj = pprint.pformat(v).replace('\n', rpl)
            s += f'    {k}: {obj},\n'
            flag = True
        if flag:
            s += ')'
        else:
            s = self.__class__.__name__ + '()'
        return s

    def keys(self) -> List[str]:
        """Return self.keys()."""
        return self._data.keys()

    def values(self) -> List[Any]:
        """Return self.values()."""
        return self._data.values()

    def items(self) -> Any:
        """Return self.items()."""
        return self._data.items()

    def get(self, k: str, d: Optional[Any] = None) -> Union['Batch', Any]:
        """Return self[k] if k in self else d. d defaults to None."""
        if k in self.keys():
            return getattr(self, k)
        return d

    def to_numpy(self) -> None:
        """Change all torch.Tensor to numpy.ndarray. This is an in-place
        operation.
        """
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.detach().cpu().numpy()
            elif isinstance(v, Batch):
                v.to_numpy()

    def to_torch(self,
                 dtype: Optional[torch.dtype] = None,
                 device: Union[str, int, torch.device] = 'cpu'
                 ) -> None:
        """Change all numpy.ndarray to torch.Tensor. This is an inplace
        operation.
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)

        for k, v in self.items():
            if isinstance(v, (np.generic, np.ndarray)):
                v = torch.from_numpy(v).to(device)
                if dtype is not None:
                    v = v.type(dtype)
                self[k] = v
            if isinstance(v, torch.Tensor):
                if dtype is not None and v.dtype != dtype:
                    must_update_tensor = True
                elif v.device.type != device.type:
                    must_update_tensor = True
                elif device.index is not None and \
                        device.index != v.device.index:
                    must_update_tensor = True
                else:
                    must_update_tensor = False
                if must_update_tensor:
                    if dtype is not None:
                        v = v.type(dtype)
                    self[k] = v.to(device)
            elif isinstance(v, Batch):
                v.to_torch(dtype, device)

    def append(self, batch: 'Batch') -> None:
        warnings.warn('Method append will be removed soon, please use '
                      ':meth:`~tianshou.data.Batch.cat`')
        return self.cat_(batch)

    def cat_(self, batch: 'Batch') -> None:
        """Concatenate a :class:`~tianshou.data.Batch` object to current
        batch.
        """
        assert isinstance(batch, Batch), \
            'Only Batch is allowed to be concatenated in-place!'
        for k, v in batch.items():
            if v is None:
                continue
            if not hasattr(self, k) or self[k] is None:
                self[k] = copy.deepcopy(v)
            elif isinstance(v, np.ndarray) and v.ndim > 0:
                self[k] = np.concatenate([self[k], v])
            elif isinstance(v, torch.Tensor):
                self[k] = torch.cat([self[k], v])
            elif isinstance(v, list):
                self[k] = self[k] + copy.deepcopy(v)
            elif isinstance(v, Batch):
                self[k].cat_(v)
            else:
                s = 'No support for method "cat" with type '\
                    f'{type(v)} in class Batch.'
                raise TypeError(s)

    @classmethod
    def cat(cls, batches: List['Batch']) -> 'Batch':
        """Concatenate a :class:`~tianshou.data.Batch` object into a
        single new batch.
        """
        assert isinstance(batches, (tuple, list)), \
            'Only list of Batch instances is allowed to be '\
            'concatenated out-of-place!'
        batch = cls()
        for batch_ in batches:
            batch.cat_(batch_)
        return batch

    @classmethod
    def stack(cls, batches: List['Batch'], axis: int = 0) -> 'Batch':
        """Stack a :class:`~tianshou.data.Batch` object into a
        single new batch.
        """
        assert isinstance(batches, (tuple, list)), \
            'Only list of Batch instances is allowed to be '\
            'stacked out-of-place!'
        if axis == 0:
            return cls(batches)
        else:
            batch = Batch()
            for k, v in zip(batches[0].keys(),
                            zip(*[e.values() for e in batches])):
                if isinstance(v[0], (np.generic, np.ndarray, list)):
                    batch[k] = np.stack(v, axis)
                elif isinstance(v[0], torch.Tensor):
                    batch[k] = torch.stack(v, axis)
                elif isinstance(v[0], Batch):
                    batch[k] = Batch.stack(v, axis)
                else:
                    s = 'No support for method "stack" with type '\
                        f'{type(v[0])} in class Batch and axis != 0.'
                    raise TypeError(s)
            return batch

    def __len__(self) -> int:
        """Return len(self)."""
        r = []
        for v in self.values():
            if isinstance(v, Batch) and v.size == 0:
                continue
            elif isinstance(v, list) and len(v) == 0:
                continue
            elif hasattr(v, '__len__') and (not isinstance(
                    v, (np.ndarray, torch.Tensor)) or v.ndim > 0):
                r.append(len(v))
            else:
                raise TypeError("Object of type 'Batch' has no len()")
        if len(r) == 0:
            raise TypeError("Object of type 'Batch' has no len()")
        return min(r)

    @property
    def size(self) -> int:
        """Return self.size."""
        if len(self.keys()) == 0:
            return 0
        else:
            r = []
            for v in self.values():
                if isinstance(v, Batch):
                    r.append(v.size)
                elif hasattr(v, '__len__') and (not isinstance(
                        v, (np.ndarray, torch.Tensor)) or v.ndim > 0):
                    r.append(len(v))
            return max(1, min(r) if len(r) > 0 else 0)

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
