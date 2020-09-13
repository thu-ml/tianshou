import torch
import numpy as np
from copy import deepcopy
from numbers import Number
from typing import Union, Optional

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
