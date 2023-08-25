import pickle
from copy import deepcopy
from numbers import Number
from typing import Any, Optional, Union, no_type_check

import h5py
import numpy as np
import torch

from tianshou.data.batch import Batch, _parse_value


@no_type_check
def to_numpy(x: Any) -> Union[Batch, np.ndarray]:
    """Return an object without torch.Tensor."""
    if isinstance(x, torch.Tensor):  # most often case
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):  # second often case
        return x
    if isinstance(x, (np.number, np.bool_, Number)):
        return np.asanyarray(x)
    if x is None:
        return np.array(None, dtype=object)
    if isinstance(x, (dict, Batch)):
        x = Batch(x) if isinstance(x, dict) else deepcopy(x)
        x.to_numpy()
        return x
    if isinstance(x, (list, tuple)):
        return to_numpy(_parse_value(x))
    # fallback
    return np.asanyarray(x)


@no_type_check
def to_torch(
    x: Any,
    dtype: Optional[torch.dtype] = None,
    device: Union[str, int, torch.device] = "cpu",
) -> Union[Batch, torch.Tensor]:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray) and issubclass(
        x.dtype.type,
        (np.bool_, np.number),
    ):  # most often case
        x = torch.from_numpy(x).to(device)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)
    if isinstance(x, (np.number, np.bool_, Number)):
        return to_torch(np.asanyarray(x), dtype, device)
    if isinstance(x, (dict, Batch)):
        x = Batch(x, copy=True) if isinstance(x, dict) else deepcopy(x)
        x.to_torch(dtype, device)
        return x
    if isinstance(x, (list, tuple)):
        return to_torch(_parse_value(x), dtype, device)
    # fallback
    raise TypeError(f"object {x} cannot be converted to torch.")


@no_type_check
def to_torch_as(x: Any, y: torch.Tensor) -> Union[Batch, torch.Tensor]:
    """Return an object without np.ndarray.

    Same as ``to_torch(x, dtype=y.dtype, device=y.device)``.
    """
    assert isinstance(y, torch.Tensor)
    return to_torch(x, dtype=y.dtype, device=y.device)


# Note: object is used as a proxy for objects that can be pickled
# Note: mypy does not support cyclic definition currently
Hdf5ConvertibleValues = Union[
    int,
    float,
    Batch,
    np.ndarray,
    torch.Tensor,
    object,
    "Hdf5ConvertibleType",
]

Hdf5ConvertibleType = dict[str, Hdf5ConvertibleValues]


def to_hdf5(x: Hdf5ConvertibleType, y: h5py.Group, compression: Optional[str] = None) -> None:
    """Copy object into HDF5 group."""

    def to_hdf5_via_pickle(
        x: object,
        y: h5py.Group,
        key: str,
        compression: Optional[str] = None,
    ) -> None:
        """Pickle, convert to numpy array and write to HDF5 dataset."""
        data = np.frombuffer(pickle.dumps(x), dtype=np.byte)
        y.create_dataset(key, data=data, compression=compression)

    for k, v in x.items():
        if isinstance(v, (Batch, dict)):
            # dicts and batches are both represented by groups
            subgrp = y.create_group(k)
            if isinstance(v, Batch):
                subgrp_data = v.__getstate__()
                subgrp.attrs["__data_type__"] = "Batch"
            else:
                subgrp_data = v
            to_hdf5(subgrp_data, subgrp, compression=compression)
        elif isinstance(v, torch.Tensor):
            # PyTorch tensors are written to datasets
            y.create_dataset(k, data=to_numpy(v), compression=compression)
            y[k].attrs["__data_type__"] = "Tensor"
        elif isinstance(v, np.ndarray):
            try:
                # NumPy arrays are written to datasets
                y.create_dataset(k, data=v, compression=compression)
                y[k].attrs["__data_type__"] = "ndarray"
            except TypeError:
                # If data type is not supported by HDF5 fall back to pickle.
                # This happens if dtype=object (e.g. due to entries being None)
                # and possibly in other cases like structured arrays.
                try:
                    to_hdf5_via_pickle(v, y, k, compression=compression)
                except Exception as exception:
                    raise RuntimeError(
                        f"Attempted to pickle {v.__class__.__name__} due to "
                        "data type not supported by HDF5 and failed.",
                    ) from exception
                y[k].attrs["__data_type__"] = "pickled_ndarray"
        elif isinstance(v, (int, float)):
            # ints and floats are stored as attributes of groups
            y.attrs[k] = v
        else:  # resort to pickle for any other type of object
            try:
                to_hdf5_via_pickle(v, y, k, compression=compression)
            except Exception as exception:
                raise NotImplementedError(
                    f"No conversion to HDF5 for object of type '{type(v)}' "
                    "implemented and fallback to pickle failed.",
                ) from exception
            y[k].attrs["__data_type__"] = v.__class__.__name__


def from_hdf5(x: h5py.Group, device: Optional[str] = None) -> Hdf5ConvertibleValues:
    """Restore object from HDF5 group."""
    if isinstance(x, h5py.Dataset):
        # handle datasets
        if x.attrs["__data_type__"] == "ndarray":
            return np.array(x)
        if x.attrs["__data_type__"] == "Tensor":
            return torch.tensor(x, device=device)
        return pickle.loads(x[()])
    # handle groups representing a dict or a Batch
    y = dict(x.attrs.items())
    data_type = y.pop("__data_type__", None)
    for k, v in x.items():
        y[k] = from_hdf5(v, device)
    return Batch(y) if data_type == "Batch" else y
