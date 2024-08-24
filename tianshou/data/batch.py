"""This module implements :class:`Batch`, a flexible data structure for
handling heterogeneous data in reinforcement learning algorithms. Such a data structure
is needed since RL algorithms differ widely in the conceptual fields that they need.
`Batch` is the main data carrier in Tianshou. It bears some similarities to
`TensorDict <https://github.com/pytorch/tensordict>`_
that is used for a similar purpose in `pytorch-rl <https://github.com/pytorch/rl>`_.
The main differences between the two are that `Batch` can hold arbitrary objects (and not just torch tensors),
and that Tianshou implements `BatchProtocol` for enabling type checking and autocompletion (more on that below).

The `Batch` class is designed to store and manipulate collections of data with
varying types and structures. It strikes a balance between flexibility and type safety, the latter mainly
achieved through the use of protocols. One can thing of it as a mixture of a dictionary and an array,
as it has both key-value pairs and nesting, while also having a shape, being indexable and sliceable.

Key features of the `Batch` class include:

1. Flexible data storage: Can hold numpy arrays, torch tensors, scalars, and nested Batch objects.
2. Dynamic attribute access: Allows setting and accessing data using attribute notation (e.g., `batch.observation`).
   This allows for type-safe and readable code and enables IDE autocompletion. See comments on `BatchProtocol` below.
3. Indexing and slicing: Supports numpy-like indexing and slicing operations. The slicing is extended to nested
   Batch objects and torch Distributions.
4. Batch operations: Provides methods for splitting, shuffling, concatenating and stacking multiple Batch objects.
5. Data type conversion: Offers methods to convert data between numpy arrays and torch tensors.
6. Value transformations: Allows applying functions to all values in the Batch recursively.
7. Analysis utilities: Provides methods for checking for missing values, dropping entries with missing values,
   and others.

Since we want to keep `Batch` flexible and not fix a specific set of fields or their types,
we don't have fixed interfaces for actual `Batch` objects that are used throughout
tianshou (such interfaces could be dataclasses, for example). However, we still want to enable
IDE autocompletion and type checking for `Batch` objects. To achieve this, we rely on dynamic duck typing
by using `Protocol`. The :class:`BatchProtocol` defines the interface that all `Batch` objects should adhere to,
and its various implementations (like :class:`~.types.ActBatchProtocol` or :class:`~.types.RolloutBatchProtocol`) define the specific
fields that are expected in the respective `Batch` objects. The protocols are then used as type hints
throughout the codebase. Protocols can't be instantiated, but we can cast to them.
For example, we "instantiate" an `ActBatchProtocol` with something like:

>>> act_batch = cast(ActBatchProtocol, Batch(act=my_action))

The users can decide for themselves how to structure their `Batch` objects, and can opt in to the
`BatchProtocol` style to enable type checking and autocompletion. Opting out will have no effect on
the functionality.
"""

import pprint
import warnings
from collections.abc import Callable, Collection, Iterable, Iterator, KeysView, Sequence
from copy import deepcopy
from numbers import Number
from types import EllipsisType
from typing import (
    Any,
    Literal,
    Protocol,
    Self,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

import numpy as np
import pandas as pd
import torch
from deepdiff import DeepDiff
from sensai.util import logging
from torch.distributions import Categorical, Distribution, Independent, Normal

_SingleIndexType = slice | int | EllipsisType
IndexType = np.ndarray | _SingleIndexType | Sequence[_SingleIndexType]
TBatch = TypeVar("TBatch", bound="BatchProtocol")
TDistribution = TypeVar("TDistribution", bound=Distribution)
T = TypeVar("T")
TArr = torch.Tensor | np.ndarray

log = logging.getLogger(__name__)


def _is_batch_set(obj: Any) -> bool:
    # Batch set is a list/tuple of dict/Batch objects,
    # or 1-D np.ndarray with object type,
    # where each element is a dict/Batch object
    if isinstance(obj, np.ndarray):  # most often case
        # "for element in obj" will just unpack the first dimension,
        # but obj.tolist() will flatten ndarray of objects
        # so do not use obj.tolist()
        if obj.shape == ():
            return False
        return obj.dtype == object and all(isinstance(element, dict | Batch) for element in obj)
    if (
        isinstance(obj, list | tuple)
        and len(obj) > 0
        and all(isinstance(element, dict | Batch) for element in obj)
    ):
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
    # np.asanyarray will cause dead loop in some cases
    return np.isscalar(value)


def _is_number(value: Any) -> bool:
    # isinstance(value, Number) checks 1, 1.0, np.int(1), np.float(1.0), etc.
    # isinstance(value, np.nummber) checks np.int32(1), np.float64(1.0), etc.
    # isinstance(value, np.bool_) checks np.bool_(True), etc.
    # similar to np.isscalar but np.isscalar('st') returns True
    return isinstance(value, Number | np.number | np.bool_)


def _to_array_with_correct_type(obj: Any) -> np.ndarray:
    if isinstance(obj, np.ndarray) and issubclass(obj.dtype.type, np.bool_ | np.number):
        return obj  # most often case
    # convert the value to np.ndarray
    # convert to object obj type if neither bool nor number
    # raises an exception if array's elements are tensors themselves
    try:
        obj_array = np.asanyarray(obj)
    except ValueError:
        obj_array = np.asanyarray(obj, dtype=object)
    if not issubclass(obj_array.dtype.type, np.bool_ | np.number):
        obj_array = obj_array.astype(object)
    if obj_array.dtype == object:
        # scalar ndarray with object obj type is very annoying
        # a=np.array([np.array({}, dtype=object), np.array({}, dtype=object)])
        # a is not array([{}, {}], dtype=object), and a[0]={} results in
        # something very strange:
        # array([{}, array({}, dtype=object)], dtype=object)
        if not obj_array.shape:
            obj_array = obj_array.item(0)
        elif all(isinstance(arr, np.ndarray) for arr in obj_array.reshape(-1)):
            return obj_array  # various length, np.array([[1], [2, 3], [4, 5, 6]])
        elif any(isinstance(arr, torch.Tensor) for arr in obj_array.reshape(-1)):
            raise ValueError("Numpy arrays of tensors are not supported yet.")
    return obj_array


def create_value(
    inst: Any,
    size: int,
    stack: bool = True,
) -> Union["Batch", np.ndarray, torch.Tensor]:
    """Create empty place-holders according to inst's shape.

    :param stack: whether to stack or to concatenate. E.g. if inst has shape of
        (3, 5), size = 10, stack=True returns an np.array with shape of (10, 3, 5),
        otherwise (10, 5)
    """
    has_shape = isinstance(inst, np.ndarray | torch.Tensor)
    is_scalar = _is_scalar(inst)
    if not stack and is_scalar:
        # should never hit since it has already checked in Batch.cat_ , here we do not
        # consider scalar types, following the behavior of numpy which does not support
        # concatenation of zero-dimensional arrays (scalars)
        raise TypeError(f"cannot concatenate with {inst} which is scalar")
    if has_shape:
        shape = (size, *inst.shape) if stack else (size, *inst.shape[1:])
    if isinstance(inst, np.ndarray):
        target_type = (
            inst.dtype.type if issubclass(inst.dtype.type, np.bool_ | np.number) else object
        )
        return np.full(shape, fill_value=None if target_type == object else 0, dtype=target_type)
    if isinstance(inst, torch.Tensor):
        return torch.full(shape, fill_value=0, device=inst.device, dtype=inst.dtype)
    if isinstance(inst, dict | Batch):
        zero_batch = Batch()
        for key, val in inst.items():
            zero_batch.__dict__[key] = create_value(val, size, stack=stack)
        return zero_batch
    if is_scalar:
        return create_value(np.asarray(inst), size, stack=stack)
    # fall back to object
    return np.array([None for _ in range(size)], object)


def _assert_type_keys(keys: Iterable[str]) -> None:
    assert all(isinstance(key, str) for key in keys), f"keys should all be string, but got {keys}"


def _parse_value(obj: Any) -> Union["Batch", np.ndarray, torch.Tensor] | None:
    if isinstance(obj, Batch):  # most often case
        return obj
    if (
        (isinstance(obj, np.ndarray) and issubclass(obj.dtype.type, np.bool_ | np.number))
        or isinstance(obj, torch.Tensor)
        or obj is None
    ):  # third often case
        return obj
    if _is_number(obj):  # second often case, but it is more time-consuming
        return np.asanyarray(obj)
    if isinstance(obj, dict):
        return Batch(obj)
    if (
        not isinstance(obj, np.ndarray)
        and isinstance(obj, Collection)
        and len(obj) > 0
        and all(isinstance(element, torch.Tensor) for element in obj)
    ):
        try:
            obj = cast(list[torch.Tensor], obj)
            return torch.stack(obj)
        except RuntimeError as exception:
            raise TypeError(
                "Batch does not support non-stackable iterable"
                " of torch.Tensor as unique value yet.",
            ) from exception
    if _is_batch_set(obj):
        obj = Batch(obj)  # list of dict / Batch
    else:
        # None, scalar, normal obj list (main case)
        # or an actual list of objects
        try:
            obj = _to_array_with_correct_type(obj)
        except ValueError as exception:
            raise TypeError(
                "Batch does not support heterogeneous list/tuple of tensors as unique value yet.",
            ) from exception
    return obj


def alloc_by_keys_diff(
    meta: "BatchProtocol",
    batch: "BatchProtocol",
    size: int,
    stack: bool = True,
) -> None:
    """Creates place-holders inside meta for keys that are in batch but not in meta.

    This mainly is an internal method, use it only if you know what you are doing.
    """
    for key in batch.get_keys():
        if key in meta.get_keys():
            if isinstance(meta[key], Batch) and isinstance(batch[key], Batch):
                alloc_by_keys_diff(meta[key], batch[key], size, stack)
            elif isinstance(meta[key], Batch) and len(meta[key].get_keys()) == 0:
                meta[key] = create_value(batch[key], size, stack)
        else:
            meta[key] = create_value(batch[key], size, stack)


class ProtocolCalledException(Exception):
    """The methods of a Protocol should never be called.

    Currently, no static type checker actually verifies that a class that inherits
    from a Protocol does in fact provide the correct interface. Thus, it may happen
    that a method of the protocol is called accidentally (this is an
    implementation error). The normal error for that is a somewhat cryptic
    AttributeError, wherefore we instead raise this custom exception in the
    BatchProtocol.

    Finally and importantly: using this in BatchProtocol makes mypy verify the fields
    in the various sub-protocols and thus renders is MUCH more useful!
    """


def get_sliced_dist(dist: TDistribution, index: IndexType) -> TDistribution:
    """Slice a distribution object by the given index."""
    if isinstance(dist, Categorical):
        return Categorical(probs=dist.probs[index])  # type: ignore[return-value]
    if isinstance(dist, Normal):
        return Normal(loc=dist.loc[index], scale=dist.scale[index])  # type: ignore[return-value]
    if isinstance(dist, Independent):
        return Independent(
            get_sliced_dist(dist.base_dist, index),
            dist.reinterpreted_batch_ndims,
        )  # type: ignore[return-value]
    else:
        raise NotImplementedError(f"Unsupported distribution for slicing: {dist}")


def get_len_of_dist(dist: Distribution) -> int:
    """Return the length (typically batch size) of a distribution object."""
    if len(dist.batch_shape) == 0:
        raise TypeError(f"scalar Distribution has no length: {dist=}")
    return dist.batch_shape[0]


def dist_to_atleast_2d(dist: TDistribution) -> TDistribution:
    """Convert a distribution to at least 2D, such that the `batch_shape` attribute has a len of at least 1."""
    if len(dist.batch_shape) > 0:
        return dist
    if isinstance(dist, Categorical):
        return Categorical(probs=dist.probs.unsqueeze(0))  # type: ignore[return-value]
    elif isinstance(dist, Normal):
        return Normal(loc=dist.loc.unsqueeze(0), scale=dist.scale.unsqueeze(0))  # type: ignore[return-value]
    elif isinstance(dist, Independent):
        return Independent(
            dist_to_atleast_2d(dist.base_dist),
            dist.reinterpreted_batch_ndims,
        )  # type: ignore[return-value]
    else:
        raise NotImplementedError(f"Unsupported distribution for conversion to 2D: {type(dist)}")


# Note: This is implemented as a protocol because the interface
# of Batch is always extended by adding new fields. Having a hierarchy of
# protocols building off this one allows for type safety and IDE support despite
# the dynamic nature of Batch
@runtime_checkable
class BatchProtocol(Protocol):
    """The internal data structure in Tianshou.

    Batch is a kind of supercharged array (of temporal data) stored individually in a
    (recursive) dictionary of objects that can be either numpy arrays, torch tensors, or
    batches themselves. It is designed to make it extremely easily to access, manipulate
    and set partial view of the heterogeneous data conveniently.

    For a detailed description, please refer to :ref:`batch_concept`.
    """

    @property
    def shape(self) -> list[int]:
        raise ProtocolCalledException

    # NOTE: even though setattr and getattr are defined for any object, we need
    # to explicitly define them for the BatchProtocol, since otherwise mypy will
    # complain about new fields being added dynamically. For example, things like
    # `batch.new_field = ...` followed by using `batch.new_field` become type errors
    # if getattr and setattr are missing in the BatchProtocol.
    #
    # For the moment, tianshou relies on this kind of dynamic-field-addition
    # in many, many places. In principle, it would be better to construct new
    # objects with new combinations of fields instead of mutating existing ones - the
    # latter is error-prone and can't properly be expressed with types. May be in a
    # future, rather different version of tianshou it would be feasible to have stricter
    # typing. Then the need for Protocols would in fact disappear
    def __setattr__(self, key: str, value: Any) -> None:
        raise ProtocolCalledException

    def __getattr__(self, key: str) -> Any:
        raise ProtocolCalledException

    def __iter__(self) -> Iterator[Self]:
        raise ProtocolCalledException

    @overload
    def __getitem__(self, index: str) -> Any:
        raise ProtocolCalledException

    @overload
    def __getitem__(self, index: IndexType) -> Self:
        raise ProtocolCalledException

    def __getitem__(self, index: str | IndexType) -> Any:
        raise ProtocolCalledException

    def __setitem__(self, index: str | IndexType, value: Any) -> None:
        raise ProtocolCalledException

    def __iadd__(self, other: Self | Number | np.number) -> Self:
        raise ProtocolCalledException

    def __add__(self, other: Self | Number | np.number) -> Self:
        raise ProtocolCalledException

    def __imul__(self, value: Number | np.number) -> Self:
        raise ProtocolCalledException

    def __mul__(self, value: Number | np.number) -> Self:
        raise ProtocolCalledException

    def __itruediv__(self, value: Number | np.number) -> Self:
        raise ProtocolCalledException

    def __truediv__(self, value: Number | np.number) -> Self:
        raise ProtocolCalledException

    def __repr__(self) -> str:
        raise ProtocolCalledException

    def __eq__(self, other: Any) -> bool:
        raise ProtocolCalledException

    def to_numpy(self: Self) -> Self:
        """Change all torch.Tensor to numpy.ndarray and return a new Batch."""
        raise ProtocolCalledException

    def to_numpy_(self) -> None:
        """Change all torch.Tensor to numpy.ndarray in-place."""
        raise ProtocolCalledException

    def to_torch(
        self: Self,
        dtype: torch.dtype | None = None,
        device: str | int | torch.device = "cpu",
    ) -> Self:
        """Change all numpy.ndarray to torch.Tensor and return a new Batch."""
        raise ProtocolCalledException

    def to_torch_(
        self,
        dtype: torch.dtype | None = None,
        device: str | int | torch.device = "cpu",
    ) -> None:
        """Change all numpy.ndarray to torch.Tensor in-place."""
        raise ProtocolCalledException

    def cat_(self, batches: Self | Sequence[dict | Self]) -> None:
        """Concatenate a list of (or one) Batch objects into current batch."""
        raise ProtocolCalledException

    @staticmethod
    def cat(batches: Sequence[dict | TBatch]) -> TBatch:
        """Concatenate a list of Batch object into a single new batch.

        For keys that are not shared across all batches, batches that do not
        have these keys will be padded by zeros with appropriate shapes. E.g.
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
        raise ProtocolCalledException

    def stack_(self, batches: Sequence[dict | Self], axis: int = 0) -> None:
        """Stack a list of Batch object into current batch."""
        raise ProtocolCalledException

    @staticmethod
    def stack(batches: Sequence[dict | TBatch], axis: int = 0) -> TBatch:
        """Stack a list of Batch object into a single new batch.

        For keys that are not shared across all batches, batches that do not
        have these keys will be padded by zeros. E.g.
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
        raise ProtocolCalledException

    def empty_(self, index: slice | IndexType | None = None) -> Self:
        """Return an empty Batch object with 0 or None filled.

        If "index" is specified, it will only reset the specific indexed-data.
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
        raise ProtocolCalledException

    @staticmethod
    def empty(batch: TBatch, index: IndexType | None = None) -> TBatch:
        """Return an empty Batch object with 0 or None filled.

        The shape is the same as the given Batch.
        """
        raise ProtocolCalledException

    def update(self, batch: dict | Self | None = None, **kwargs: Any) -> None:
        """Update this batch from another dict/Batch."""
        raise ProtocolCalledException

    def __len__(self) -> int:
        raise ProtocolCalledException

    def split(
        self,
        size: int,
        shuffle: bool = True,
        merge_last: bool = False,
    ) -> Iterator[Self]:
        """Split whole data into multiple small batches.

        :param size: divide the data batch with the given size, but one
            batch if the length of the batch is smaller than "size". Size of -1 means
            the whole batch.
        :param shuffle: randomly shuffle the entire data batch if it is
            True, otherwise remain in the same. Default to True.
        :param merge_last: merge the last batch into the previous one.
            Default to False.
        """
        raise ProtocolCalledException

    def to_dict(self, recurse: bool = True) -> dict[str, Any]:
        raise ProtocolCalledException

    def to_list_of_dicts(self) -> list[dict[str, Any]]:
        raise ProtocolCalledException

    def get_keys(self) -> KeysView:
        raise ProtocolCalledException

    def set_array_at_key(
        self,
        seq: np.ndarray,
        key: str,
        index: IndexType | None = None,
        default_value: float | None = None,
    ) -> None:
        """Set a sequence of values at a given key.

        If `index` is not passed, the sequence must have the same length as the batch.

        :param seq: the array of values to set.
        :param key: the key to set the sequence at.
        :param index: the indices to set the sequence at. If None, the sequence must have
            the same length as the batch and will be set at all indices.
        :param default_value: this only applies if `index` is passed and the key does not exist yet
            in the batch. In that case, entries outside the passed index will be filled
            with this default value.
            Note that the array at the key will be of the same dtype as the passed sequence,
            so `default_value` should be such that numpy can cast it to this dtype.
        """
        raise ProtocolCalledException

    def isnull(self) -> Self:
        """Return a boolean mask of the same shape, indicating missing values."""
        raise ProtocolCalledException

    def hasnull(self) -> bool:
        """Return whether the batch has missing values."""
        raise ProtocolCalledException

    def dropnull(self) -> Self:
        """Return a batch where all items in which any value is null are dropped.

        Note that it is not the same as just dropping the entries of the sequence.
        For example, with

        >>> b = Batch(a=[None, 2, 3, 4], b=[4, 5, None, 7])
        >>> b.dropnull()

        will result in

        >>> Batch(a=[2, 4], b=[5, 7])

        This logic is applied recursively to all nested batches. The result is
        the same as if the batch was flattened, entries were dropped,
        and then the batch was reshaped back to the original nested structure.
        """
        ...

    @overload
    def apply_values_transform(
        self,
        values_transform: Callable[[np.ndarray | torch.Tensor], Any],
    ) -> Self:
        ...

    @overload
    def apply_values_transform(
        self,
        values_transform: Callable,
        inplace: Literal[True],
    ) -> None:
        ...

    @overload
    def apply_values_transform(
        self,
        values_transform: Callable[[np.ndarray | torch.Tensor], Any],
        inplace: Literal[False],
    ) -> Self:
        ...

    def apply_values_transform(
        self,
        values_transform: Callable[[np.ndarray | torch.Tensor], Any],
        inplace: bool = False,
    ) -> None | Self:
        """Apply a function to all arrays in the batch, including nested ones.

        :param values_transform: the function to apply to the arrays.
        :param inplace: whether to apply the function in-place. If False, a new batch is returned,
            otherwise the batch is modified in-place and None is returned.
        """
        raise ProtocolCalledException

    def get(self, key: str, default: Any | None = None) -> Any:
        raise ProtocolCalledException

    def pop(self, key: str, default: Any | None = None) -> Any:
        raise ProtocolCalledException

    def to_at_least_2d(self) -> Self:
        """Ensures that all arrays and dists in the batch have at least 2 dimensions.

        This is useful for ensuring that all arrays in the batch can be concatenated
        along a new axis.
        """
        raise ProtocolCalledException


class Batch(BatchProtocol):
    """See :class:`~tianshou.data.batch.BatchProtocol`."""

    __doc__ = BatchProtocol.__doc__

    def __init__(
        self,
        batch_dict: dict
        | BatchProtocol
        | Sequence[dict | BatchProtocol]
        | np.ndarray
        | None = None,
        copy: bool = False,
        **kwargs: Any,
    ) -> None:
        if copy:
            batch_dict = deepcopy(batch_dict)
        if batch_dict is not None:
            if isinstance(batch_dict, dict | BatchProtocol):
                _assert_type_keys(batch_dict.keys())
                for batch_key, obj in batch_dict.items():
                    self.__dict__[batch_key] = _parse_value(obj)
            elif _is_batch_set(batch_dict):
                batch_dict = cast(Sequence[dict | BatchProtocol], batch_dict)
                self.stack_(batch_dict)
        if len(kwargs) > 0:
            # TODO: that's a rather weird pattern, is it really needed?
            # Feels like kwargs could be just merged into batch_dict in the beginning
            self.__init__(kwargs, copy=copy)  # type: ignore

    def to_dict(self, recursive: bool = True) -> dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if recursive and isinstance(v, Batch):
                v = v.to_dict(recursive=recursive)
            result[k] = v
        return result

    def get_keys(self) -> KeysView:
        return self.__dict__.keys()

    def get(self, key: str, default: Any | None = None) -> Any:
        return self.__dict__.get(key, default)

    def pop(self, key: str, default: Any | None = None) -> Any:
        return self.__dict__.pop(key, default)

    def to_list_of_dicts(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self]

    def __setattr__(self, key: str, value: Any) -> None:
        """Set self.key = value."""
        self.__dict__[key] = _parse_value(value)

    def __getattr__(self, key: str) -> Any:
        """Return self.key. The "Any" return type is needed for mypy."""
        return getattr(self.__dict__, key)

    def __contains__(self, key: str) -> bool:
        """Return key in self."""
        return key in self.__dict__

    def __getstate__(self) -> dict[str, Any]:
        """Pickling interface.

        Only the actual data are serialized for both efficiency and simplicity.
        """
        state = {}
        for batch_key, obj in self.items():
            if isinstance(obj, Batch):
                state[batch_key] = obj.__getstate__()
            else:
                state[batch_key] = obj
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Unpickling interface.

        At this point, self is an empty Batch instance that has not been
        initialized, so it can safely be initialized by the pickle state.
        """
        self.__init__(**state)  # type: ignore

    @overload
    def __getitem__(self, index: str) -> Any:
        ...

    @overload
    def __getitem__(self, index: IndexType) -> Self:
        ...

    def __getitem__(self, index: str | IndexType) -> Any:
        """Returns either the value of a key or a sliced Batch object."""
        if isinstance(index, str):
            return self.__dict__[index]
        batch_items = self.items()
        if len(batch_items) > 0:
            new_batch = Batch()

            sliced_obj: Any
            for batch_key, obj in batch_items:
                # None and empty Batches as values are added to any slice
                if obj is None:
                    sliced_obj = None
                elif isinstance(obj, Batch) and len(obj.get_keys()) == 0:
                    sliced_obj = Batch()
                # We attempt slicing of a distribution. This is hacky, but presents an important special case
                elif isinstance(obj, Distribution):
                    sliced_obj = get_sliced_dist(obj, index)
                # All other objects are either array-like or Batch-like, so hopefully sliceable
                # A batch should have no scalars
                else:
                    sliced_obj = obj[index]
                new_batch.__dict__[batch_key] = sliced_obj
            return new_batch
        raise IndexError("Cannot access item from empty Batch object.")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False

        this_batch_no_torch_tensor = self.to_numpy()
        other_batch_no_torch_tensor = other.to_numpy()
        # DeepDiff 7.0.1 cannot compare 0-dimensional arrays
        # so, we ensure with this transform that all array values have at least 1 dim
        this_batch_no_torch_tensor.apply_values_transform(
            values_transform=np.atleast_1d,
            inplace=True,
        )
        other_batch_no_torch_tensor.apply_values_transform(
            values_transform=np.atleast_1d,
            inplace=True,
        )
        this_dict = this_batch_no_torch_tensor.to_dict(recursive=True)
        other_dict = other_batch_no_torch_tensor.to_dict(recursive=True)

        return not DeepDiff(this_dict, other_dict)

    def __iter__(self) -> Iterator[Self]:
        # TODO: empty batch raises an error on len and needs separate treatment, that's probably not a good idea
        if len(self.__dict__) == 0:
            yield from []
        else:
            for i in range(len(self)):
                yield self[i]

    def __setitem__(self, index: str | IndexType, value: Any) -> None:
        """Assign value to self[index]."""
        value = _parse_value(value)
        if isinstance(index, str):
            self.__dict__[index] = value
            return
        if not isinstance(value, Batch):
            raise ValueError(
                "Batch does not supported tensor assignment. "
                "Use a compatible Batch or dict instead.",
            )
        if not set(value.keys()).issubset(self.__dict__.keys()):
            raise ValueError("Creating keys is not supported by item assignment.")
        for key, val in self.items():
            try:
                self.__dict__[key][index] = value[key]
            except KeyError:
                if isinstance(val, Batch):
                    self.__dict__[key][index] = Batch()
                elif isinstance(val, torch.Tensor) or (
                    isinstance(val, np.ndarray) and issubclass(val.dtype.type, np.bool_ | np.number)
                ):
                    self.__dict__[key][index] = 0
                else:
                    self.__dict__[key][index] = None

    def __iadd__(self, other: Self | Number | np.number) -> Self:
        """Algebraic addition with another Batch instance in-place."""
        if isinstance(other, Batch):
            for (batch_key, obj), value in zip(
                self.__dict__.items(),
                other.__dict__.values(),
                strict=True,
            ):  # TODO are keys consistent?
                if isinstance(obj, Batch) and len(obj.get_keys()) == 0:
                    continue
                self.__dict__[batch_key] += value
            return self
        if _is_number(other):
            for batch_key, obj in self.items():
                if isinstance(obj, Batch) and len(obj.get_keys()) == 0:
                    continue
                self.__dict__[batch_key] += other
            return self
        raise TypeError("Only addition of Batch or number is supported.")

    def __add__(self, other: Self | Number | np.number) -> Self:
        """Algebraic addition with another Batch instance out-of-place."""
        return deepcopy(self).__iadd__(other)

    def __imul__(self, value: Number | np.number) -> Self:
        """Algebraic multiplication with a scalar value in-place."""
        assert _is_number(value), "Only multiplication by a number is supported."
        for batch_key, obj in self.__dict__.items():
            if isinstance(obj, Batch) and len(obj.get_keys()) == 0:
                continue
            self.__dict__[batch_key] *= value
        return self

    def __mul__(self, value: Number | np.number) -> Self:
        """Algebraic multiplication with a scalar value out-of-place."""
        return deepcopy(self).__imul__(value)

    def __itruediv__(self, value: Number | np.number) -> Self:
        """Algebraic division with a scalar value in-place."""
        assert _is_number(value), "Only division by a number is supported."
        for batch_key, obj in self.__dict__.items():
            if isinstance(obj, Batch) and len(obj.get_keys()) == 0:
                continue
            self.__dict__[batch_key] /= value
        return self

    def __truediv__(self, value: Number | np.number) -> Self:
        """Algebraic division with a scalar value out-of-place."""
        return deepcopy(self).__itruediv__(value)

    def __repr__(self) -> str:
        """Return str(self)."""
        self_str = self.__class__.__name__ + "(\n"
        flag = False
        for batch_key, obj in self.__dict__.items():
            rpl = "\n" + " " * (6 + len(batch_key))
            obj_name = pprint.pformat(obj).replace("\n", rpl)
            self_str += f"    {batch_key}: {obj_name},\n"
            flag = True
        if flag:
            self_str += ")"
        else:
            self_str = self.__class__.__name__ + "()"
        return self_str

    def to_numpy(self: Self) -> Self:
        result = deepcopy(self)
        result.to_numpy_()
        return result

    def to_numpy_(self) -> None:
        def arr_to_numpy(arr: TArr) -> TArr:
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
            return arr

        self.apply_values_transform(arr_to_numpy, inplace=True)

    def to_torch(
        self: Self,
        dtype: torch.dtype | None = None,
        device: str | int | torch.device = "cpu",
    ) -> Self:
        result = deepcopy(self)
        result.to_torch_(dtype=dtype, device=device)
        return result

    def to_torch_(
        self,
        dtype: torch.dtype | None = None,
        device: str | int | torch.device = "cpu",
    ) -> None:
        if not isinstance(device, torch.device):
            device = torch.device(device)

        def arr_to_torch(arr: TArr) -> TArr:
            if isinstance(arr, np.ndarray):
                return torch.from_numpy(arr).to(device)

            # TODO: simplify
            if (
                dtype is not None
                and arr.dtype != dtype
                or arr.device.type != device.type
                or device.index != arr.device.index
            ):
                if dtype is not None:
                    arr = arr.type(dtype)
                return arr.to(device)
            return arr

        self.apply_values_transform(arr_to_torch, inplace=True)

    def __cat(self, batches: Sequence[dict | Self], lens: list[int]) -> None:
        """Private method for Batch.cat_.

        ::

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
        for len_ in lens:
            sum_lens.append(sum_lens[-1] + len_)
        # collect non-empty keys
        keys_map = [
            {
                batch_key
                for batch_key, obj in batch.items()
                if not (isinstance(obj, Batch) and len(obj.get_keys()) == 0)
            }
            for batch in batches
        ]
        keys_shared = set.intersection(*keys_map)
        values_shared = [[batch[key] for batch in batches] for key in keys_shared]
        for key, shared_value in zip(keys_shared, values_shared, strict=True):
            if all(isinstance(element, dict | Batch) for element in shared_value):
                batch_holder = Batch()
                batch_holder.__cat(shared_value, lens=lens)
                self.__dict__[key] = batch_holder
            elif all(isinstance(element, torch.Tensor) for element in shared_value):
                self.__dict__[key] = torch.cat(shared_value)
            else:
                # cat Batch(a=np.zeros((3, 4))) and Batch(a=Batch(b=Batch()))
                # will fail here
                self.__dict__[key] = _to_array_with_correct_type(np.concatenate(shared_value))
        keys_total = set.union(*[set(batch.keys()) for batch in batches])
        keys_reserve_or_partial = set.difference(keys_total, keys_shared)
        # keys that are reserved in all batches
        keys_reserve = set.difference(keys_total, set.union(*keys_map))
        # keys that occur only in some batches, but not all
        keys_partial = keys_reserve_or_partial.difference(keys_reserve)
        for key in keys_reserve:
            # reserved keys
            self.__dict__[key] = Batch()
        for key in keys_partial:
            for i, batch in enumerate(batches):
                if key not in batch.__dict__:
                    continue
                value = batch.get(key)
                if isinstance(value, Batch) and len(value.get_keys()) == 0:
                    continue
                try:
                    self.__dict__[key][sum_lens[i] : sum_lens[i + 1]] = value
                except KeyError:
                    self.__dict__[key] = create_value(value, sum_lens[-1], stack=False)
                    self.__dict__[key][sum_lens[i] : sum_lens[i + 1]] = value

    def cat_(self, batches: BatchProtocol | Sequence[dict | BatchProtocol]) -> None:
        if isinstance(batches, BatchProtocol | dict):
            batches = [batches]
        # check input format
        batch_list = []

        original_keys_only_batch = None
        """A batch with all values removed, just keys left. Can be considered a sort of schema.
        Will be either the schema of self, or of the first non-empty batch in the sequence.
        """
        if len(self) > 0:
            original_keys_only_batch = self.apply_values_transform(lambda x: None)
            original_keys_only_batch.replace_empty_batches_by_none()

        for batch in batches:
            if isinstance(batch, dict):
                batch = Batch(batch)
            if not isinstance(batch, Batch):
                raise ValueError(f"Cannot concatenate {type(batch)} in Batch.cat_")
            if len(batch.get_keys()) == 0:
                continue
            if original_keys_only_batch is None:
                original_keys_only_batch = batch.apply_values_transform(lambda x: None)
                original_keys_only_batch.replace_empty_batches_by_none()
                batch_list.append(batch)
                continue

            cur_keys_only_batch = batch.apply_values_transform(lambda x: None)
            cur_keys_only_batch.replace_empty_batches_by_none()
            if original_keys_only_batch != cur_keys_only_batch:
                raise ValueError(
                    f"Batch.cat_ only supports concatenation of batches with the same structure but got "
                    f"structures: \n{original_keys_only_batch}\n   and\n{cur_keys_only_batch}.",
                )
            batch_list.append(batch)
        if len(batch_list) == 0:
            return

        batches = batch_list

        # TODO: lot's of the remaining logic is devoted to filling up remaining keys with zeros
        #   this should be removed, and also the check above should be extended to nested keys
        try:
            # len(batch) here means batch is a nested empty batch
            # like Batch(a=Batch), and we have to treat it as length zero and
            # keep it.
            lens = [0 if len(batch) == 0 else len(batch) for batch in batches]
        except TypeError as exception:
            raise ValueError(
                "Batch.cat_ meets an exception. Maybe because there is any "
                f"scalar in {batches} but Batch.cat_ does not support the "
                "concatenation of scalar.",
            ) from exception
        if len(self.get_keys()) != 0:
            batches = [self, *list(batches)]
            # len of zero means that that item is Batch() and should be ignored
            lens = [0 if len(self) == 0 else len(self), *lens]
        self.__cat(batches, lens)

    @staticmethod
    def cat(batches: Sequence[dict | TBatch]) -> TBatch:
        batch = Batch()
        batch.cat_(batches)
        return batch  # type: ignore

    def stack_(self, batches: Sequence[dict | BatchProtocol], axis: int = 0) -> None:
        # check input format
        batch_list = []
        for batch in batches:
            if isinstance(batch, dict):
                if len(batch) > 0:
                    batch_list.append(Batch(batch))
            elif isinstance(batch, Batch):
                if len(batch.get_keys()) != 0:
                    batch_list.append(batch)
            else:
                raise ValueError(f"Cannot concatenate {type(batch)} in Batch.stack_")
        if len(batch_list) == 0:
            return
        batches = batch_list
        if len(self.get_keys()) != 0:
            batches = [self, *batches]
        # collect non-empty keys
        keys_map = [
            {
                batch_key
                for batch_key, obj in batch.items()
                if not (isinstance(obj, BatchProtocol) and len(obj.get_keys()) == 0)
            }
            for batch in batches
        ]
        keys_shared = set.intersection(*keys_map)
        values_shared = [[batch[key] for batch in batches] for key in keys_shared]
        for shared_key, value in zip(keys_shared, values_shared, strict=True):
            # second often
            if all(isinstance(element, torch.Tensor) for element in value):
                self.__dict__[shared_key] = torch.stack(value, axis)
            # third often
            elif all(isinstance(element, BatchProtocol | dict) for element in value):
                self.__dict__[shared_key] = Batch.stack(value, axis)
            else:  # most often case is np.ndarray
                try:
                    self.__dict__[shared_key] = _to_array_with_correct_type(np.stack(value, axis))
                except ValueError:
                    warnings.warn(
                        "You are using tensors with different shape,"
                        " fallback to dtype=object by default.",
                    )
                    self.__dict__[shared_key] = np.array(value, dtype=object)
        # all the keys
        keys_total = set.union(*[set(batch.keys()) for batch in batches])
        # keys that are reserved in all batches
        keys_reserve = set.difference(keys_total, set.union(*keys_map))
        # keys that are either partial or reserved
        keys_reserve_or_partial = set.difference(keys_total, keys_shared)
        # keys that occur only in some batches, but not all
        keys_partial = keys_reserve_or_partial.difference(keys_reserve)
        if keys_partial and axis != 0:
            raise ValueError(
                f"Stack of Batch with non-shared keys {keys_partial} is only "
                f"supported with axis=0, but got axis={axis}!",
            )
        for key in keys_reserve:
            # reserved keys
            self.__dict__[key] = Batch()
        for key in keys_partial:
            for i, batch in enumerate(batches):
                if key not in batch.__dict__:
                    continue
                value = batch.get(key)
                # TODO: fix code/annotations s.t. the ignores can be removed
                if (
                    isinstance(value, BatchProtocol)  # type: ignore
                    and len(value.get_keys()) == 0  # type: ignore
                ):
                    continue  # type: ignore
                try:
                    self.__dict__[key][i] = value
                except KeyError:
                    self.__dict__[key] = create_value(value, len(batches))
                    self.__dict__[key][i] = value

    @staticmethod
    def stack(batches: Sequence[dict | TBatch], axis: int = 0) -> TBatch:
        batch = Batch()
        batch.stack_(batches, axis)
        # can't cast to a generic type, so we have to ignore the type here
        return batch  # type: ignore

    def empty_(self, index: slice | IndexType | None = None) -> Self:
        for batch_key, obj in self.items():
            if isinstance(obj, torch.Tensor):  # most often case
                self.__dict__[batch_key][index] = 0
            elif obj is None:
                continue
            elif isinstance(obj, np.ndarray):
                if obj.dtype == object:
                    self.__dict__[batch_key][index] = None
                else:
                    self.__dict__[batch_key][index] = 0
            elif isinstance(obj, Batch):
                self.__dict__[batch_key].empty_(index=index)
            else:  # scalar value
                warnings.warn(
                    "You are calling Batch.empty on a NumPy scalar, "
                    "which may cause undefined behaviors.",
                )
                if _is_number(obj):
                    self.__dict__[batch_key] = obj.__class__(0)
                else:
                    self.__dict__[batch_key] = None
        return self

    @staticmethod
    def empty(batch: TBatch, index: IndexType | None = None) -> TBatch:
        return deepcopy(batch).empty_(index)

    def update(self, batch: dict | Self | None = None, **kwargs: Any) -> None:
        if batch is None:
            self.update(kwargs)
            return
        for batch_key, obj in batch.items():
            self.__dict__[batch_key] = _parse_value(obj)
        if kwargs:
            self.update(kwargs)

    def __len__(self) -> int:
        """Raises `TypeError` if any value in the batch has no len(), typically meaning it's a batch of scalars."""
        lens = []
        for key, obj in self.__dict__.items():
            # TODO: causes inconsistent behavior to batch with empty batches
            #  and batch with empty sequences of other type. Remove, but only after
            #  Buffer and Collectors have been improved to no longer rely on this
            if isinstance(obj, Batch) and len(obj) == 0:
                continue
            if obj is None:
                continue
            if hasattr(obj, "__len__") and (isinstance(obj, Batch) or obj.ndim > 0):
                lens.append(len(obj))
                continue
            if isinstance(obj, Distribution):
                lens.append(get_len_of_dist(obj))
                continue
            raise TypeError(f"Entry for {key} in {self} is {obj} has no len()")
        if not lens:
            return 0
        return min(lens)

    @property
    def shape(self) -> list[int]:
        """Return self.shape."""
        if len(self.get_keys()) == 0:
            return []
        data_shape = []
        for obj in self.__dict__.values():
            try:
                data_shape.append(list(obj.shape))
            except AttributeError:
                data_shape.append([])
        return (
            list(map(min, zip(*data_shape, strict=False))) if len(data_shape) > 1 else data_shape[0]
        )

    def split(
        self,
        size: int,
        shuffle: bool = True,
        merge_last: bool = False,
    ) -> Iterator[Self]:
        length = len(self)
        if size == -1:
            size = length
        assert size >= 1  # size can be greater than length, return whole batch
        indices = np.random.permutation(length) if shuffle else np.arange(length)
        merge_last = merge_last and length % size > 0
        for idx in range(0, length, size):
            if merge_last and idx + size + size >= length:
                yield self[indices[idx:]]
                break
            yield self[indices[idx : idx + size]]

    @overload
    def apply_values_transform(
        self,
        values_transform: Callable,
    ) -> Self:
        ...

    @overload
    def apply_values_transform(
        self,
        values_transform: Callable,
        inplace: Literal[True],
    ) -> None:
        ...

    @overload
    def apply_values_transform(
        self,
        values_transform: Callable,
        inplace: Literal[False],
    ) -> Self:
        ...

    def apply_values_transform(
        self,
        values_transform: Callable,
        inplace: bool = False,
    ) -> None | Self:
        """Applies a function to all non-batch-values in the batch, including
        values in nested batches.

        A batch with keys pointing to either batches or to non-batch values can
        be thought of as a tree of Batch nodes. This function traverses the tree
        and applies the function to all leaf nodes (i.e. values that are not
        batches themselves).

        The values are usually arrays, but can also be scalar values of an
        arbitrary type since retrieving a single entry from a Batch a la
        `batch[0]` will return a batch with scalar values.
        """
        return _apply_batch_values_func_recursively(self, values_transform, inplace=inplace)

    def set_array_at_key(
        self,
        arr: np.ndarray,
        key: str,
        index: IndexType | None = None,
        default_value: float | None = None,
    ) -> None:
        if index is not None:
            if key not in self.get_keys():
                log.info(
                    f"Key {key} not found in batch, "
                    f"creating a sequence of len {len(self)} with {default_value=} for it.",
                )
                try:
                    self[key] = np.array([default_value] * len(self), dtype=arr.dtype)
                except TypeError as exception:
                    raise TypeError(
                        f"Cannot create a sequence of dtype {arr.dtype} with default value {default_value}. "
                        f"You can fix this either by passing an array with the correct dtype or by passing "
                        f"a different default value that can be cast to the array's dtype (or both).",
                    ) from exception
            else:
                existing_entry = self[key]
                if isinstance(existing_entry, BatchProtocol):
                    raise ValueError(
                        f"Cannot set sequence at key {key} because it is a nested batch, "
                        f"can only set a subsequence of an array.",
                    )
            self[key][index] = arr
        else:
            if len(arr) != len(self):
                raise ValueError(
                    f"Sequence length {len(arr)} does not match "
                    f"batch length {len(self)}. For setting a subsequence with missing "
                    f"entries filled up by default values, consider passing an index.",
                )
            self[key] = arr

    def isnull(self) -> Self:
        return self.apply_values_transform(pd.isnull, inplace=False)

    def hasnull(self) -> bool:
        isnan_batch = self.isnull()
        is_any_null_batch = isnan_batch.apply_values_transform(np.any, inplace=False)

        def is_any_true(boolean_batch: BatchProtocol) -> bool:
            for val in boolean_batch.values():
                if isinstance(val, BatchProtocol):
                    if is_any_true(val):
                        return True
                else:
                    assert val.size == 1, "This shouldn't have happened, it's a bug!"
                    # an unsized array with a boolean, e.g. np.array(False). behaves like the boolean itself
                    if val:
                        return True
            return False

        return is_any_true(is_any_null_batch)

    def dropnull(self) -> Self:
        # we need to use dicts since a batch retrieved for a single index has no length and cat fails
        # TODO: make cat work with batches containing scalars?
        sub_batches = []
        for b in self:
            if b.hasnull():
                continue
            # needed for cat to work
            b = b.apply_values_transform(np.atleast_1d)
            sub_batches.append(b)
        return Batch.cat(sub_batches)

    def replace_empty_batches_by_none(self) -> None:
        """Goes through the batch-tree" recursively and replaces empty batches by None.

        This is useful for extracting the structure of a batch without the actual data,
        especially in combination with `apply_values_transform` with a
        transform function a la `lambda x: None`.
        """
        empty_batch = Batch()
        for key, val in self.items():
            if isinstance(val, Batch):
                if val == empty_batch:
                    self[key] = None
                else:
                    val.replace_empty_batches_by_none()

    def to_at_least_2d(self) -> Self:
        """Ensures that all arrays and dists in the batch have at least 2 dimensions.

        This is useful for ensuring that all arrays in the batch can be concatenated
        along a new axis.
        """
        result = self.apply_values_transform(np.atleast_2d, inplace=False)
        for key, val in self.items():
            if isinstance(val, Distribution):
                result[key] = dist_to_atleast_2d(val)
        return result


def _apply_batch_values_func_recursively(
    batch: TBatch,
    values_transform: Callable,
    inplace: bool = False,
) -> TBatch | None:
    """Applies the desired function on all values of the batch recursively.

    See docstring of the corresponding method in the Batch class for more details.
    """
    result = batch if inplace else deepcopy(batch)
    for key, val in batch.__dict__.items():
        if isinstance(val, BatchProtocol):
            result[key] = _apply_batch_values_func_recursively(val, values_transform, inplace=False)
        else:
            result[key] = values_transform(val)
    if not inplace:
        return result
    return None
