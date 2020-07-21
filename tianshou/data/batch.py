import torch
import pprint
import warnings
import numpy as np
from copy import deepcopy
from numbers import Number
from collections.abc import Collection
from typing import Any, List, Tuple, Union, Iterator, Optional

# Disable pickle warning related to torch, since it has been removed
# on torch master branch. See Pull Request #39003 for details:
# https://github.com/pytorch/pytorch/pull/39003
warnings.filterwarnings(
    "ignore", message="pickle support for Storage will be removed in 1.5.")


def _is_batch_set(data: Any) -> bool:
    # Batch set is a list/tuple of dict/Batch objects,
    # or 1-D np.ndarray with np.object type,
    # where each element is a dict/Batch object
    if isinstance(data, (list, tuple)):
        if len(data) > 0 and all(isinstance(e, (dict, Batch)) for e in data):
            return True
    elif isinstance(data, np.ndarray) and data.dtype == np.object:
        # ``for e in data`` will just unpack the first dimension,
        # but data.tolist() will flatten ndarray of objects
        # so do not use data.tolist()
        if all(isinstance(e, (dict, Batch)) for e in data):
            return True
    return False


def _is_scalar(value: Any) -> bool:
    # check if the value is a scalar
    # 1. python bool object, number object: isinstance(value, Number)
    # 2. numpy scalar: isinstance(value, np.generic)
    # 3. python object rather than dict / Batch / tensor
    # the check of dict / Batch is omitted because this only checks a value.
    # a dict / Batch will eventually check their values
    if isinstance(value, torch.Tensor):
        return value.numel() == 1 and not value.shape
    else:
        value = np.asanyarray(value)
        return value.size == 1 and not value.shape


def _is_number(value: Any) -> bool:
    # isinstance(value, Number) checks 1, 1.0, np.int(1), np.float(1.0), etc.
    # isinstance(value, np.nummber) checks np.int32(1), np.float64(1.0), etc.
    # isinstance(value, np.bool_) checks np.bool_(True), etc.
    is_number = isinstance(value, Number)
    is_number = is_number or isinstance(value, np.number)
    is_number = is_number or isinstance(value, np.bool_)
    return is_number


def _to_array_with_correct_type(v: Any) -> np.ndarray:
    # convert the value to np.ndarray
    # convert to np.object data type if neither bool nor number
    # raises an exception if array's elements are tensors themself
    v = np.asanyarray(v)
    if not issubclass(v.dtype.type, (np.bool_, np.number)):
        v = v.astype(np.object)
    if v.dtype == np.object:
        # scalar ndarray with np.object data type is very annoying
        # a=np.array([np.array({}, dtype=object), np.array({}, dtype=object)])
        # a is not array([{}, {}], dtype=object), and a[0]={} results in
        # something very strange:
        # array([{}, array({}, dtype=object)], dtype=object)
        if not v.shape:
            v = v.item(0)
        elif any(isinstance(e, (np.ndarray, torch.Tensor))
                 for e in v.reshape(-1)):
            raise ValueError("Numpy arrays of tensors are not supported yet.")
    return v


def _create_value(inst: Any, size: int, stack=True) -> Union[
        'Batch', np.ndarray, torch.Tensor]:
    """
    :param bool stack: whether to stack or to concatenate. E.g. if inst has
        shape of (3, 5), size = 10, stack=True returns an np.ndarry with shape
        of (10, 3, 5), otherwise (10, 5)
    """
    has_shape = isinstance(inst, (np.ndarray, torch.Tensor))
    is_scalar = _is_scalar(inst)
    if not stack and is_scalar:
        # here we do not consider scalar types, following the behavior of numpy
        # which does not support concatenation of zero-dimensional arrays
        # (scalars)
        raise TypeError(f"cannot concatenate with {inst} which is scalar")
    if has_shape:
        shape = (size, *inst.shape) if stack else (size, *inst.shape[1:])
    if isinstance(inst, np.ndarray):
        if issubclass(inst.dtype.type, (np.bool_, np.number)):
            target_type = inst.dtype.type
        else:
            target_type = np.object
        return np.full(shape,
                       fill_value=None if target_type == np.object else 0,
                       dtype=target_type)
    elif isinstance(inst, torch.Tensor):
        return torch.full(shape,
                          fill_value=0,
                          device=inst.device,
                          dtype=inst.dtype)
    elif isinstance(inst, (dict, Batch)):
        zero_batch = Batch()
        for key, val in inst.items():
            zero_batch.__dict__[key] = _create_value(val, size, stack=stack)
        return zero_batch
    elif is_scalar:
        return _create_value(np.asarray(inst), size, stack=stack)
    else:  # fall back to np.object
        return np.array([None for _ in range(size)])


def _assert_type_keys(keys):
    keys = list(keys)
    assert all(isinstance(e, str) for e in keys), \
        f"keys should all be string, but got {keys}"


def _parse_value(v: Any):
    if isinstance(v, dict):
        v = Batch(v)
    elif isinstance(v, (Batch, torch.Tensor)):
        pass
    else:
        if not isinstance(v, np.ndarray) and isinstance(v, Collection) and \
                len(v) > 0 and all(isinstance(e, torch.Tensor) for e in v):
            try:
                return torch.stack(v)
            except RuntimeError as e:
                raise TypeError("Batch does not support non-stackable iterable"
                                " of torch.Tensor as unique value yet.") from e
        try:
            v_ = _to_array_with_correct_type(v)
        except ValueError as e:
            raise TypeError("Batch does not support heterogeneous list/tuple"
                            " of tensors as unique value yet.") from e
        if _is_batch_set(v):
            v = Batch(v)  # list of dict / Batch
        else:
            # None, scalar, normal data list (main case)
            # or an actual list of objects
            v = v_
    return v


class Batch:
    """Tianshou provides :class:`~tianshou.data.Batch` as the internal data
    structure to pass any kind of data to other methods, for example, a
    collector gives a :class:`~tianshou.data.Batch` to policy for learning.

    For a detailed description, please refer to :ref:`batch_concept`.
    """
    def __init__(self,
                 batch_dict: Optional[Union[
                     dict, 'Batch', Tuple[Union[dict, 'Batch']],
                     List[Union[dict, 'Batch']], np.ndarray]] = None,
                 copy: bool = False,
                 **kwargs) -> None:
        if copy:
            batch_dict = deepcopy(batch_dict)
        if batch_dict is not None:
            if isinstance(batch_dict, (dict, Batch)):
                _assert_type_keys(batch_dict.keys())
                for k, v in batch_dict.items():
                    self.__dict__[k] = _parse_value(v)
            elif _is_batch_set(batch_dict):
                self.stack_(batch_dict)
        if len(kwargs) > 0:
            self.__init__(kwargs, copy=copy)

    def __setattr__(self, key: str, value: Any):
        """self.key = value"""
        self.__dict__[key] = _parse_value(value)

    def __getstate__(self):
        """Pickling interface. Only the actual data are serialized for both
        efficiency and simplicity.
        """
        state = {}
        for k, v in self.items():
            if isinstance(v, Batch):
                v = v.__getstate__()
            state[k] = v
        return state

    def __setstate__(self, state):
        """Unpickling interface. At this point, self is an empty Batch instance
        that has not been initialized, so it can safely be initialized by the
        pickle state.
        """
        self.__init__(**state)

    def __getitem__(self, index: Union[
            str, slice, int, np.integer, np.ndarray, List[int]]) -> 'Batch':
        """Return self[index]."""
        if isinstance(index, str):
            return self.__dict__[index]
        batch_items = self.items()
        if len(batch_items) > 0:
            b = Batch()
            for k, v in batch_items:
                if isinstance(v, Batch) and v.is_empty():
                    b.__dict__[k] = Batch()
                else:
                    b.__dict__[k] = v[index]
            return b
        else:
            raise IndexError("Cannot access item from empty Batch object.")

    def __setitem__(self, index: Union[
            str, slice, int, np.integer, np.ndarray, List[int]],
            value: Any) -> None:
        """Assign value to self[index]."""
        if isinstance(index, str):
            self.__dict__[index] = _parse_value(value)
            return
        value = _parse_value(value)
        if isinstance(value, (np.ndarray, torch.Tensor)):
            raise ValueError("Batch does not supported tensor assignment."
                             " Use a compatible Batch or dict instead.")
        if not set(value.keys()).issubset(self.__dict__.keys()):
            raise KeyError(
                "Creating keys is not supported by item assignment.")
        for key, val in self.items():
            try:
                self.__dict__[key][index] = value[key]
            except KeyError:
                if isinstance(val, Batch):
                    self.__dict__[key][index] = Batch()
                elif isinstance(val, torch.Tensor) or \
                        (isinstance(val, np.ndarray) and
                         issubclass(val.dtype.type, (np.bool_, np.number))):
                    self.__dict__[key][index] = 0
                else:
                    self.__dict__[key][index] = None

    def __iadd__(self, other: Union['Batch', Number, np.number]):
        """Algebraic addition with another :class:`~tianshou.data.Batch`
        instance in-place."""
        if isinstance(other, Batch):
            for (k, r), v in zip(self.__dict__.items(),
                                 other.__dict__.values()):
                # TODO are keys consistent?
                if isinstance(r, Batch) and r.is_empty():
                    continue
                else:
                    self.__dict__[k] += v
            return self
        elif _is_number(other):
            for k, r in self.items():
                if isinstance(r, Batch) and r.is_empty():
                    continue
                else:
                    self.__dict__[k] += other
            return self
        else:
            raise TypeError("Only addition of Batch or number is supported.")

    def __add__(self, other: Union['Batch', Number, np.number]):
        """Algebraic addition with another :class:`~tianshou.data.Batch`
        instance out-of-place."""
        return deepcopy(self).__iadd__(other)

    def __imul__(self, val: Union[Number, np.number]):
        """Algebraic multiplication with a scalar value in-place."""
        assert _is_number(val), \
            "Only multiplication by a number is supported."
        for k, r in self.__dict__.items():
            if isinstance(r, Batch) and r.is_empty():
                continue
            self.__dict__[k] *= val
        return self

    def __mul__(self, val: Union[Number, np.number]):
        """Algebraic multiplication with a scalar value out-of-place."""
        return deepcopy(self).__imul__(val)

    def __itruediv__(self, val: Union[Number, np.number]):
        """Algebraic division with a scalar value in-place."""
        assert _is_number(val), \
            "Only division by a number is supported."
        for k, r in self.__dict__.items():
            if isinstance(r, Batch) and r.is_empty():
                continue
            self.__dict__[k] /= val
        return self

    def __truediv__(self, val: Union[Number, np.number]):
        """Algebraic division with a scalar value out-of-place."""
        return deepcopy(self).__itruediv__(val)

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
        return self.__dict__.keys()

    def values(self) -> List[Any]:
        """Return self.values()."""
        return self.__dict__.values()

    def items(self) -> List[Tuple[str, Any]]:
        """Return self.items()."""
        return self.__dict__.items()

    def get(self, k: str, d: Optional[Any] = None) -> Union['Batch', Any]:
        """Return self[k] if k in self else d. d defaults to None."""
        return self.__dict__.get(k, d)

    def to_numpy(self) -> None:
        """Change all torch.Tensor to numpy.ndarray. This is an in-place
        operation.
        """
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.detach().cpu().numpy()
            elif isinstance(v, Batch):
                v.to_numpy()

    def to_torch(self, dtype: Optional[torch.dtype] = None,
                 device: Union[str, int, torch.device] = 'cpu') -> None:
        """Change all numpy.ndarray to torch.Tensor. This is an in-place
        operation.
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)

        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                if dtype is not None and v.dtype != dtype or \
                        v.device.type != device.type or \
                        device.index is not None and \
                        device.index != v.device.index:
                    if dtype is not None:
                        v = v.type(dtype)
                    self.__dict__[k] = v.to(device)
            elif isinstance(v, Batch):
                v.to_torch(dtype, device)
            else:
                # ndarray or scalar
                if not isinstance(v, np.ndarray):
                    v = np.asanyarray(v)
                v = torch.from_numpy(v).to(device)
                if dtype is not None:
                    v = v.type(dtype)
                self.__dict__[k] = v

    def __cat(self,
              batches: Union['Batch', List[Union[dict, 'Batch']]],
              lens: List[int]) -> None:
        """::

            >>> a = Batch(a=np.random.randn(3, 4))
            >>> x = Batch(a=a, b=np.random.randn(4, 4))
            >>> y = Batch(a=Batch(a=Batch()), b=np.random.randn(4, 4))

        If we want to concatenate x and y, we want to pad y.a.a with zeros.
        Without ``lens`` as a hint, when we concatenate x.a and y.a, we would
        not be able to know how to pad y.a. So ``Batch.cat_`` should compute
        the ``lens`` to give ``Batch.__cat`` a hint.
        ::

            >>> ans = Batch.cat([x, y])
            >>> # this is equivalent to the following line
            >>> ans = Batch(); ans.__cat([x, y], lens=[3, 4])
            >>> # this lens is equal to [len(a), len(b)]
        """
        # partial keys will be padded by zeros
        # with the shape of [len, rest_shape]
        sum_lens = [0]
        for x in lens:
            sum_lens.append(sum_lens[-1] + x)
        # collect non-empty keys
        keys_map = [
            set(k for k, v in batch.items()
                if not (isinstance(v, Batch) and v.is_empty()))
            for batch in batches]
        keys_shared = set.intersection(*keys_map)
        values_shared = [[e[k] for e in batches] for k in keys_shared]
        _assert_type_keys(keys_shared)
        for k, v in zip(keys_shared, values_shared):
            if all(isinstance(e, (dict, Batch)) for e in v):
                batch_holder = Batch()
                batch_holder.__cat(v, lens=lens)
                self.__dict__[k] = batch_holder
            elif all(isinstance(e, torch.Tensor) for e in v):
                self.__dict__[k] = torch.cat(v)
            else:
                # cat Batch(a=np.zeros((3, 4))) and Batch(a=Batch(b=Batch()))
                # will fail here
                v = np.concatenate(v)
                v = _to_array_with_correct_type(v)
                self.__dict__[k] = v
        keys_total = set.union(*[set(b.keys()) for b in batches])
        keys_reserve_or_partial = set.difference(keys_total, keys_shared)
        _assert_type_keys(keys_reserve_or_partial)
        # keys that are reserved in all batches
        keys_reserve = set.difference(keys_total, set.union(*keys_map))
        # keys that occur only in some batches, but not all
        keys_partial = keys_reserve_or_partial.difference(keys_reserve)
        for k in keys_reserve:
            # reserved keys
            self.__dict__[k] = Batch()
        for k in keys_partial:
            for i, e in enumerate(batches):
                if k not in e.__dict__:
                    continue
                val = e.get(k)
                if isinstance(val, Batch) and val.is_empty():
                    continue
                try:
                    self.__dict__[k][sum_lens[i]:sum_lens[i + 1]] = val
                except KeyError:
                    self.__dict__[k] = \
                        _create_value(val, sum_lens[-1], stack=False)
                    self.__dict__[k][sum_lens[i]:sum_lens[i + 1]] = val

    def cat_(self,
             batches: Union['Batch', List[Union[dict, 'Batch']]]) -> None:
        """Concatenate a list of (or one) :class:`~tianshou.data.Batch` objects
        into current batch.
        """
        if isinstance(batches, Batch):
            batches = [batches]
        if len(batches) == 0:
            return
        batches = [x if isinstance(x, Batch) else Batch(x) for x in batches]

        # x.is_empty() means that x is Batch() and should be ignored
        batches = [x for x in batches if not x.is_empty()]
        try:
            # x.is_empty(recurse=True) here means x is a nested empty batch
            # like Batch(a=Batch), and we have to treat it as length zero and
            # keep it.
            lens = [0 if x.is_empty(recurse=True) else len(x)
                    for x in batches]
        except TypeError as e:
            e2 = ValueError(
                f'Batch.cat_ meets an exception. Maybe because there is '
                f'any scalar in {batches} but Batch.cat_ does not support'
                f'the concatenation of scalar.')
            raise Exception([e, e2])
        if not self.is_empty():
            batches = [self] + list(batches)
            lens = [0 if self.is_empty(recurse=True) else len(self)] + lens
        return self.__cat(batches, lens)

    @staticmethod
    def cat(batches: List[Union[dict, 'Batch']]) -> 'Batch':
        """Concatenate a list of :class:`~tianshou.data.Batch` object into a
        single new batch. For keys that are not shared across all batches,
        batches that do not have these keys will be padded by zeros with
        appropriate shapes. E.g.
        ::

            >>> a = Batch(a=np.zeros([3, 4]), common=Batch(c=np.zeros([3, 5])))
            >>> b = Batch(b=np.zeros([4, 3]), common=Batch(c=np.zeros([4, 5])))
            >>> c = Batch.cat([a, b])
            >>> c.a.shape
            (7, 4)
            >>> c.b.shape
            (7, 3)
            >>> c.common.c.shape
            (7, 5)
        """
        batch = Batch()
        batch.cat_(batches)
        return batch

    def stack_(self,
               batches: List[Union[dict, 'Batch']],
               axis: int = 0) -> None:
        """Stack a list of :class:`~tianshou.data.Batch` object into current
        batch.
        """
        if len(batches) == 0:
            return
        batches = [x if isinstance(x, Batch) else Batch(x) for x in batches]
        if not self.is_empty():
            batches = [self] + list(batches)
        # collect non-empty keys
        keys_map = [
            set(k for k, v in batch.items()
                if not (isinstance(v, Batch) and v.is_empty()))
            for batch in batches]
        keys_shared = set.intersection(*keys_map)
        values_shared = [[e[k] for e in batches] for k in keys_shared]
        _assert_type_keys(keys_shared)
        for k, v in zip(keys_shared, values_shared):
            if all(isinstance(e, (dict, Batch)) for e in v):
                self.__dict__[k] = Batch.stack(v, axis)
            elif all(isinstance(e, torch.Tensor) for e in v):
                self.__dict__[k] = torch.stack(v, axis)
            else:
                v = np.stack(v, axis)
                v = _to_array_with_correct_type(v)
                self.__dict__[k] = v
        # all the keys
        keys_total = set.union(*[set(b.keys()) for b in batches])
        # keys that are reserved in all batches
        keys_reserve = set.difference(keys_total, set.union(*keys_map))
        # keys that are either partial or reserved
        keys_reserve_or_partial = set.difference(keys_total, keys_shared)
        # keys that occur only in some batches, but not all
        keys_partial = keys_reserve_or_partial.difference(keys_reserve)
        if keys_partial and axis != 0:
            raise ValueError(
                f"Stack of Batch with non-shared keys {keys_partial} "
                f"is only supported with axis=0, but got axis={axis}!")
        _assert_type_keys(keys_reserve_or_partial)
        for k in keys_reserve:
            # reserved keys
            self.__dict__[k] = Batch()
        for k in keys_partial:
            for i, e in enumerate(batches):
                if k not in e.__dict__:
                    continue
                val = e.get(k)
                if isinstance(val, Batch) and val.is_empty():
                    continue
                try:
                    self.__dict__[k][i] = val
                except KeyError:
                    self.__dict__[k] = \
                        _create_value(val, len(batches))
                    self.__dict__[k][i] = val

    @staticmethod
    def stack(batches: List[Union[dict, 'Batch']], axis: int = 0) -> 'Batch':
        """Stack a list of :class:`~tianshou.data.Batch` object into a single
        new batch. For keys that are not shared across all batches,
        batches that do not have these keys will be padded by zeros. E.g.
        ::

            >>> a = Batch(a=np.zeros([4, 4]), common=Batch(c=np.zeros([4, 5])))
            >>> b = Batch(b=np.zeros([4, 6]), common=Batch(c=np.zeros([4, 5])))
            >>> c = Batch.stack([a, b])
            >>> c.a.shape
            (2, 4, 4)
            >>> c.b.shape
            (2, 4, 6)
            >>> c.common.c.shape
            (2, 4, 5)

        .. note::

            If there are keys that are not shared across all batches, ``stack``
            with ``axis != 0`` is undefined, and will cause an exception.
        """
        batch = Batch()
        batch.stack_(batches, axis)
        return batch

    def empty_(self, index: Union[
        str, slice, int, np.integer, np.ndarray, List[int]] = None
    ) -> 'Batch':
        """Return an empty a :class:`~tianshou.data.Batch` object with 0 or
        ``None`` filled. If ``index`` is specified, it will only reset the
        specific indexed-data.
        ::

            >>> data.empty_()
            >>> print(data)
            Batch(
                a: array([[0., 0.],
                          [0., 0.]]),
                b: array([None, None], dtype=object),
            )
            >>> b={'c': [2., 'st'], 'd': [1., 0.]}
            >>> data = Batch(a=[False,  True], b=b)
            >>> data[0] = Batch.empty(data[1])
            >>> data
            Batch(
                a: array([False,  True]),
                b: Batch(
                       c: array([None, 'st']),
                       d: array([0., 0.]),
                   ),
            )
        """
        for k, v in self.items():
            if v is None:
                continue
            if isinstance(v, Batch):
                self.__dict__[k].empty_(index=index)
            elif isinstance(v, torch.Tensor):
                self.__dict__[k][index] = 0
            elif isinstance(v, np.ndarray):
                if v.dtype == np.object:
                    self.__dict__[k][index] = None
                else:
                    self.__dict__[k][index] = 0
            else:  # scalar value
                warnings.warn('You are calling Batch.empty on a NumPy scalar, '
                              'which may cause undefined behaviors.')
                if _is_number(v):
                    self.__dict__[k] = v.__class__(0)
                else:
                    self.__dict__[k] = None
        return self

    @staticmethod
    def empty(batch: 'Batch', index: Union[
        str, slice, int, np.integer, np.ndarray, List[int]] = None
    ) -> 'Batch':
        """Return an empty :class:`~tianshou.data.Batch` object with 0 or
        ``None`` filled, the shape is the same as the given
        :class:`~tianshou.data.Batch`.
        """
        return deepcopy(batch).empty_(index)

    def update(self, batch: Optional[Union[dict, 'Batch']] = None,
               **kwargs) -> None:
        """Update this batch from another dict/Batch."""
        if batch is None:
            self.update(kwargs)
            return
        if isinstance(batch, dict):
            batch = Batch(batch)
        for k, v in batch.items():
            self.__dict__[k] = v
        if kwargs:
            self.update(kwargs)

    def __len__(self) -> int:
        """Return len(self)."""
        r = []
        for v in self.__dict__.values():
            if isinstance(v, Batch) and v.is_empty(recurse=True):
                continue
            elif hasattr(v, '__len__') and (not isinstance(
                    v, (np.ndarray, torch.Tensor)) or v.ndim > 0):
                r.append(len(v))
            else:
                raise TypeError(f"Object {v} in {self} has no len()")
        if len(r) == 0:
            # empty batch has the shape of any, like the tensorflow '?' shape.
            # So it has no length.
            raise TypeError(f"Object {self} has no len()")
        return min(r)

    def is_empty(self, recurse: bool = False):
        """
        Test if a Batch is empty. If ``recurse=True``, it further tests the
        values of the object; else it only tests the existence of any key.

        ``b.is_empty(recurse=True)`` is mainly used to distinguish
        ``Batch(a=Batch(a=Batch()))`` and ``Batch(a=1)``. They both raise
        exceptions when applied to ``len()``, but the former can be used in
        ``cat``, while the latter is a scalar and cannot be used in ``cat``.

        Another usage is in ``__len__``, where we have to skip checking the
        length of recursively empty Batch.
        ::

            >>> Batch().is_empty()
            True
            >>> Batch(a=Batch(), b=Batch(c=Batch())).is_empty()
            False
            >>> Batch(a=Batch(), b=Batch(c=Batch())).is_empty(recurse=True)
            True
            >>> Batch(d=1).is_empty()
            False
            >>> Batch(a=np.float64(1.0)).is_empty()
            False
        """
        if len(self.__dict__) == 0:
            return True
        if not recurse:
            return False
        return all(False if not isinstance(x, Batch)
                   else x.is_empty(recurse=True) for x in self.values())

    @property
    def shape(self) -> List[int]:
        """Return self.shape."""
        if self.is_empty():
            return []
        else:
            data_shape = []
            for v in self.__dict__.values():
                try:
                    data_shape.append(list(v.shape))
                except AttributeError:
                    data_shape.append([])
            return list(map(min, zip(*data_shape))) if len(data_shape) > 1 \
                else data_shape[0]

    def split(self, size: Optional[int] = None,
              shuffle: bool = True) -> Iterator['Batch']:
        """Split whole data into multiple small batches.

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
