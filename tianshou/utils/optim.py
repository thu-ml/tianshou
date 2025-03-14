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
