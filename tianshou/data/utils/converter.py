import h5py
import torch
import pickle
import numpy as np
from copy import deepcopy
from numbers import Number
from typing import Dict, Union, Optional

from tianshou.data.batch import _parse_value, Batch


def to_numpy(
    x: Optional[Union[Batch, dict, list, tuple, np.number, np.bool_, Number,
                      np.ndarray, torch.Tensor]]
) -> Union[Batch, dict, list, tuple, np.ndarray]:
    """Return an object without torch.Tensor."""
    if isinstance(x, torch.Tensor):  # most often case
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):  # second often case
        return x
    elif isinstance(x, (np.number, np.bool_, Number)):
        return np.asanyarray(x)
    elif x is None:
        return np.array(None, dtype=np.object)
    elif isinstance(x, Batch):
        x = deepcopy(x)
        x.to_numpy()
        return x
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        try:
            return to_numpy(_parse_value(x))
        except TypeError:
            return [to_numpy(e) for e in x]
    else:  # fallback
        return np.asanyarray(x)


def to_torch(
    x: Union[Batch, dict, list, tuple, np.number, np.bool_, Number, np.ndarray,
             torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Union[str, int, torch.device] = "cpu",
) -> Union[Batch, dict, list, tuple, torch.Tensor]:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray) and issubclass(
        x.dtype.type, (np.bool_, np.number)
    ):  # most often case
        x = torch.from_numpy(x).to(device)  # type: ignore
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)  # type: ignore
    elif isinstance(x, (np.number, np.bool_, Number)):
        return to_torch(np.asanyarray(x), dtype, device)
    elif isinstance(x, dict):
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif isinstance(x, Batch):
        x = deepcopy(x)
        x.to_torch(dtype, device)
        return x
    elif isinstance(x, (list, tuple)):
        try:
            return to_torch(_parse_value(x), dtype, device)
        except TypeError:
            return [to_torch(e, dtype, device) for e in x]
    else:  # fallback
        raise TypeError(f"object {x} cannot be converted to torch.")


def to_torch_as(
    x: Union[Batch, dict, list, tuple, np.ndarray, torch.Tensor],
    y: torch.Tensor,
) -> Union[Batch, dict, list, tuple, torch.Tensor]:
    """Return an object without np.ndarray.

    Same as ``to_torch(x, dtype=y.dtype, device=y.device)``.
    """
    assert isinstance(y, torch.Tensor)
    return to_torch(x, dtype=y.dtype, device=y.device)


# Note: object is used as a proxy for objects that can be pickled
# Note: mypy does not support cyclic definition currently
Hdf5ConvertibleValues = Union[  # type: ignore
    int, float, Batch, np.ndarray, torch.Tensor, object,
    'Hdf5ConvertibleType',  # type: ignore
]

Hdf5ConvertibleType = Dict[str, Hdf5ConvertibleValues]  # type: ignore


def to_hdf5(x: Hdf5ConvertibleType, y: h5py.Group) -> None:
    """Copy object into HDF5 group."""

    def to_hdf5_via_pickle(x: object, y: h5py.Group, key: str) -> None:
        """Pickle, convert to numpy array and write to HDF5 dataset."""
        data = np.frombuffer(pickle.dumps(x), dtype=np.byte)
        y.create_dataset(key, data=data)

    for k, v in x.items():
        if isinstance(v, (Batch, dict)):
            # dicts and batches are both represented by groups
            subgrp = y.create_group(k)
            if isinstance(v, Batch):
                subgrp_data = v.__getstate__()
                subgrp.attrs["__data_type__"] = "Batch"
            else:
                subgrp_data = v
            to_hdf5(subgrp_data, subgrp)
        elif isinstance(v, torch.Tensor):
            # PyTorch tensors are written to datasets
            y.create_dataset(k, data=to_numpy(v))
            y[k].attrs["__data_type__"] = "Tensor"
        elif isinstance(v, np.ndarray):
            try:
                # NumPy arrays are written to datasets
                y.create_dataset(k, data=v)
                y[k].attrs["__data_type__"] = "ndarray"
            except TypeError:
                # If data type is not supported by HDF5 fall back to pickle.
                # This happens if dtype=object (e.g. due to entries being None)
                # and possibly in other cases like structured arrays.
                try:
                    to_hdf5_via_pickle(v, y, k)
                except Exception as e:
                    raise RuntimeError(
                        f"Attempted to pickle {v.__class__.__name__} due to "
                        "data type not supported by HDF5 and failed."
                    ) from e
                y[k].attrs["__data_type__"] = "pickled_ndarray"
        elif isinstance(v, (int, float)):
            # ints and floats are stored as attributes of groups
            y.attrs[k] = v
        else:  # resort to pickle for any other type of object
            try:
                to_hdf5_via_pickle(v, y, k)
            except Exception as e:
                raise NotImplementedError(
                    f"No conversion to HDF5 for object of type '{type(v)}' "
                    "implemented and fallback to pickle failed."
                ) from e
            y[k].attrs["__data_type__"] = v.__class__.__name__


def from_hdf5(
    x: h5py.Group, device: Optional[str] = None
) -> Hdf5ConvertibleType:
    """Restore object from HDF5 group."""
    if isinstance(x, h5py.Dataset):
        # handle datasets
        if x.attrs["__data_type__"] == "ndarray":
            y = np.array(x)
        elif x.attrs["__data_type__"] == "Tensor":
            y = torch.tensor(x, device=device)
        else:
            y = pickle.loads(x[()])
    else:
        # handle groups representing a dict or a Batch
        y = {k: v for k, v in x.attrs.items() if k != "__data_type__"}
        for k, v in x.items():
            y[k] = from_hdf5(v, device)
        if "__data_type__" in x.attrs:
            # if dictionary represents Batch, convert to Batch
            if x.attrs["__data_type__"] == "Batch":
                y = Batch(y)
    return y
