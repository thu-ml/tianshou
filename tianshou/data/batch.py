import torch
import copy
import pprint
import warnings
import numpy as np
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

    def __init__(self,
                 batch_dict: Optional[
                     Union[dict, Tuple[dict], List[dict], np.ndarray]] = None,
                 **kwargs) -> None:
        def _is_batch_set(data: Any) -> bool:
            if isinstance(data, (list, tuple)):
                if len(data) > 0 and isinstance(data[0], dict):
                    return True
            elif isinstance(data, np.ndarray):
                if isinstance(data.item(0), dict):
                    return True
            return False

        if isinstance(batch_dict, np.ndarray) and batch_dict.ndim == 0:
            batch_dict = batch_dict[()]
        if _is_batch_set(batch_dict):
            for k, v in zip(batch_dict[0].keys(),
                            zip(*[e.values() for e in batch_dict])):
                if isinstance(v[0], dict) or _is_batch_set(v[0]):
                    self.__dict__[k] = Batch(v)
                elif isinstance(v[0], (np.generic, np.ndarray)):
                    self.__dict__[k] = np.stack(v, axis=0)
                elif isinstance(v[0], torch.Tensor):
                    self.__dict__[k] = torch.stack(v, dim=0)
                elif isinstance(v[0], Batch):
                    self.__dict__[k] = Batch.stack(v)
                else:
                    self.__dict__[k] = list(v)
        elif isinstance(batch_dict, dict):
            for k, v in batch_dict.items():
                if isinstance(v, dict) or _is_batch_set(v):
                    self.__dict__[k] = Batch(v)
                else:
                    self.__dict__[k] = v
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
                return _valid_bounds(length, min(index)) and \
                    _valid_bounds(length, max(index))
            elif isinstance(index, slice):
                return _valid_bounds(length, index.start) and \
                    _valid_bounds(length, index.stop - 1)

        if isinstance(index, str):
            return self.__getattr__(index)
        else:
            b = Batch()
            for k, v in self.__dict__.items():
                if isinstance(v, Batch):
                    b.__dict__[k] = v[index]
                elif hasattr(v, '__len__') and (not isinstance(
                        v, (np.ndarray, torch.Tensor)) or v.ndim > 0):
                    if _valid_bounds(len(v), index):
                        b.__dict__[k] = v[index]
            return b

    def __getattr__(self, key: str) -> Union['Batch', Any]:
        """Return self.key"""
        if key not in self.__dict__:
            raise AttributeError(key)
        return self.__dict__[key]

    def __repr__(self) -> str:
        """Return str(self)."""
        s = self.__class__.__name__ + '(\n'
        flag = False
        for k in sorted(self.__dict__.keys()):
            if self.__dict__.get(k, None) is not None:
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
        return self.__dict__.keys()

    def values(self) -> List[Any]:
        """Return self.values()."""
        return self.__dict__.values()

    def items(self) -> Any:
        """Return self.items()."""
        return self.__dict__.items()

    def get(self, k: str, d: Optional[Any] = None) -> Union['Batch', Any]:
        """Return self[k] if k in self else d. d defaults to None."""
        if k in self.__dict__:
            return self.__getattr__(k)
        return d

    def to_numpy(self) -> None:
        """Change all torch.Tensor to numpy.ndarray. This is an inplace
        operation.
        """
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.detach().cpu().numpy()
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

        for k, v in self.__dict__.items():
            if isinstance(v, (np.generic, np.ndarray)):
                v = torch.from_numpy(v).to(device)
                if dtype is not None:
                    v = v.type(dtype)
                self.__dict__[k] = v
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
                    self.__dict__[k] = v.to(device)
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
        for k, v in batch.__dict__.items():
            if v is None:
                continue
            if not hasattr(self, k) or self.__dict__[k] is None:
                self.__dict__[k] = copy.deepcopy(v)
            elif isinstance(v, np.ndarray) and v.ndim > 0:
                self.__dict__[k] = np.concatenate([self.__dict__[k], v])
            elif isinstance(v, torch.Tensor):
                self.__dict__[k] = torch.cat([self.__dict__[k], v])
            elif isinstance(v, list):
                self.__dict__[k] += copy.deepcopy(v)
            elif isinstance(v, Batch):
                self.__dict__[k].cat_(v)
            else:
                s = 'No support for method "cat" with type '\
                    f'{type(v)} in class Batch.'
                raise TypeError(s)

    @staticmethod
    def cat(batches: List['Batch']) -> None:
        """Concatenate a :class:`~tianshou.data.Batch` object into a
        single new batch.
        """
        assert isinstance(batches, (tuple, list)), \
            'Only list of Batch instances is allowed to be '\
            'concatenated out-of-place!'
        batch = Batch()
        for batch_ in batches:
            batch.cat_(batch_)
        return batch

    @staticmethod
    def stack(batches: List['Batch']):
        """Stack a :class:`~tianshou.data.Batch` object into a
        single new batch.
        """
        assert isinstance(batches, (tuple, list)), \
            'Only list of Batch instances is allowed to be '\
            'stacked out-of-place!'
        return Batch(np.array([batch.__dict__ for batch in batches]))

    def __len__(self) -> int:
        """Return len(self)."""
        r = []
        for v in self.__dict__.values():
            if hasattr(v, '__len__') and (not isinstance(
                    v, (np.ndarray, torch.Tensor)) or v.ndim > 0):
                r.append(len(v))
        return max(r) if len(r) > 0 else 0

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
