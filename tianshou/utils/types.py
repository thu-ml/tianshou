import torch
from typing import Any, Dict, Iterable, Optional, Protocol, Sequence, Union

TDevice = Optional[Union[str, int, torch.device]]
TShape = Union[int, Sequence[int]]
# copy of the private _t_params in torch.optim.optimizer
TParams = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


# can't express kwargs with Callable, so we need a callable protocol
class TOptimFactory(Protocol):
    def __call__(self, params: TParams, lr: Optional[float] = None, **kwargs) -> torch.optim.Optimizer:
        pass
