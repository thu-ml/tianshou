import torch
import numpy as np
from numbers import Number
from typing import Union, Optional

from tianshou.data import Batch


def to_numpy(x: Union[
    torch.Tensor, dict, Batch, np.ndarray]) -> Union[
        dict, Batch, np.ndarray]:
    """Return an object without torch.Tensor."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = to_numpy(v)
    elif isinstance(x, Batch):
        x.to_numpy()
    return x


def to_torch(x: Union[torch.Tensor, dict, Batch, np.ndarray],
             dtype: Optional[torch.dtype] = None,
             device: Union[str, int, torch.device] = 'cpu'
             ) -> Union[dict, Batch, torch.Tensor]:
    """Return an object without np.ndarray."""
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        x = x.to(device)
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = to_torch(v, dtype, device)
    elif isinstance(x, Batch):
        x.to_torch(dtype, device)
    elif isinstance(x, (np.number, np.bool_, Number)):
        x = to_torch(np.asanyarray(x), dtype, device)
    elif isinstance(x, list) and len(x) > 0 and \
            all(isinstance(e, (np.number, np.bool_, Number)) for e in x):
        x = to_torch(np.asanyarray(x), dtype, device)
    elif isinstance(x, np.ndarray) and \
            isinstance(x.item(0), (np.number, np.bool_, Number)):
        x = torch.from_numpy(x).to(device)
        if dtype is not None:
            x = x.type(dtype)
    return x


def to_torch_as(x: Union[torch.Tensor, dict, Batch, np.ndarray],
                y: torch.Tensor
                ) -> Union[dict, Batch, torch.Tensor]:
    """Return an object without np.ndarray. Same as
    ``to_torch(x, dtype=y.dtype, device=y.device)``.
    """
    assert isinstance(y, torch.Tensor)
    return to_torch(x, dtype=y.dtype, device=y.device)
