from collections.abc import Iterator
from typing import TypeVar

import torch
from torch import nn


def optim_step(
    loss: torch.Tensor,
    optim: torch.optim.Optimizer,
    module: nn.Module | None = None,
    max_grad_norm: float | None = None,
) -> None:
    """Perform a single optimization step: zero_grad -> backward (-> clip_grad_norm) -> step.

    :param loss:
    :param optim:
    :param module: the module to optimize, required if max_grad_norm is passed
    :param max_grad_norm: if passed, will clip gradients using this
    """
    optim.zero_grad()
    loss.backward()
    if max_grad_norm:
        if not module:
            raise ValueError(
                "module must be passed if max_grad_norm is passed. "
                "Note: often the module will be the policy, i.e.`self`",
            )
        nn.utils.clip_grad_norm_(module.parameters(), max_norm=max_grad_norm)
    optim.step()


_STANDARD_TORCH_OPTIMIZERS = [
    torch.optim.Adam,
    torch.optim.SGD,
    torch.optim.RMSprop,
    torch.optim.Adadelta,
    torch.optim.AdamW,
    torch.optim.Adamax,
    torch.optim.NAdam,
    torch.optim.SparseAdam,
    torch.optim.LBFGS,
]

TOptim = TypeVar("TOptim", bound=torch.optim.Optimizer)


def clone_optimizer(
    optim: TOptim,
    new_params: nn.Parameter | Iterator[nn.Parameter],
) -> TOptim:
    """Clone an optimizer to get a new optim instance with new parameters.

    **WARNING**: This is a temporary measure, and should not be used in downstream code!
    Once tianshou interfaces have moved to optimizer factories instead of optimizers,
    this will be removed.

    :param optim: the optimizer to clone
    :param new_params: the new parameters to use
    :return: a new optimizer with the same configuration as the old one
    """
    optim_class = type(optim)
    # custom optimizers may not behave as expected
    if optim_class not in _STANDARD_TORCH_OPTIMIZERS:
        raise ValueError(
            f"Cannot clone optimizer {optim} of type {optim_class}"
            f"Currently, only standard torch optimizers are supported.",
        )
    return optim_class(new_params, **optim.defaults)
