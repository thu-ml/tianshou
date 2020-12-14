import h5py
import pickle
import torch
import numpy as np
from copy import deepcopy
from numbers import Number
from typing import Union, Optional, Dict

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
Hdf5ConvertibleValues = Union[
    int, float, Batch, np.ndarray, torch.Tensor, object, 'Hdf5ConvertibleType'
]
Hdf5ConvertibleType = Dict[str, Hdf5ConvertibleValues]


def to_hdf5(x: Hdf5ConvertibleType, y: h5py.Group) -> None:
    """Copy object into HDF5 group."""
    for k, v in x.items():
        # dicts and batches are both represented by groups
        if isinstance(v, (Batch, dict)):
            subgrp = y.create_group(k)
            if isinstance(v, Batch):
                subgrp_data = v.__getstate__()
            else:
                subgrp_data = v
            subgrp.attrs["__data_type__"] = v.__class__.__name__
            to_hdf5(subgrp_data, subgrp)
        # numpy arrays and pytorch tensors are written to datasets
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            y.create_dataset(k, data=to_numpy(v))
            y[k].attrs["__data_type__"] = v.__class__.__name__
        # ints and floats are stored as attributes of groups
        elif isinstance(v, (int, float)):
            y.attrs[k] = v
        # resort to pickle for any other type of object
        else:
            try:
                int_data = np.frombuffer(pickle.dumps(v), dtype="uint8")
            except Exception as e:
                raise NotImplementedError(
                        "No coonversion to HDF5 for object of type "
                        f"'{type(v)}' implemented and fallback to pickle "
                        "failed.\n" + str(e)
                )
            y.create_dataset(k, data=int_data, dtype="uint8")
            y[k].attrs["__data_type__"] = v.__class__.__name__


def from_hdf5(
    x: h5py.Group, device: Optional[str] = None
) -> Hdf5ConvertibleType:
    """Restore object from HDF5 group."""
    # handle datasets
    if isinstance(x, h5py.Dataset):
        if x.attrs["__data_type__"] == "ndarray":
            y = np.array(x)
        elif x.attrs["__data_type__"] == "Tensor":
            y = torch.tensor(x, device=device)
        else:
            y = pickle.loads(x[()])
    # handle groups representing a dict or a batch
    else:
        y = {k: v for k, v in x.attrs.items() if k != "__data_type__"}
        for k, v in x.items():
            y[k] = from_hdf5(v, device)
        if "__data_type__" in x.attrs:
            # if dictionary represents Batch have to convert to Batch
            if x.attrs["__data_type__"] == "Batch":
                y = Batch(y)
    return y
