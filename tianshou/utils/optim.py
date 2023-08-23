from typing import Optional

import torch
from torch import nn


def optim_step(
    loss: torch.Tensor,
    optim: torch.optim.Optimizer,
    module: nn.Module,
    max_grad_norm: Optional[float] = None,
) -> None:
    """Perform a single optimization step.

    :param loss:
    :param optim:
    :param module:
    :param max_grad_norm: if passed, will clip gradients using this
    """
    optim.zero_grad()
    loss.backward()
    if max_grad_norm:
        nn.utils.clip_grad_norm_(module.parameters(), max_norm=max_grad_norm)
    optim.step()
