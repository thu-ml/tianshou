from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, overload

import torch
import torch.distributions as dist
from gymnasium import spaces
from torch import nn

if TYPE_CHECKING:
    from tianshou.policy import BasePolicy


@contextmanager
def torch_train_mode(module: nn.Module, enabled: bool = True) -> Iterator[None]:
    """Temporarily switch to `module.training=enabled`, affecting things like `BatchNormalization`."""
    original_mode = module.training
    try:
        module.train(enabled)
        yield
    finally:
        module.train(original_mode)


@contextmanager
def policy_within_training_step(policy: "BasePolicy", enabled: bool = True) -> Iterator[None]:
    """Temporarily switch to `policy.is_within_training_step=enabled`.

    Enabling this ensures that the policy is able to adapt its behavior,
    allowing it to differentiate between training and inference/evaluation,
    e.g., to sample actions instead of using the most probable action (where applicable)
    Note that for rollout, which also happens within a training step, one would usually want
    the wrapped torch module to be in evaluation mode, which can be achieved using
    `with torch_train_mode(policy, False)`. For subsequent gradient updates, the policy should be both
    within training step and in torch train mode.
    """
    original_mode = policy.is_within_training_step
    try:
        policy.is_within_training_step = enabled
        yield
    finally:
        policy.is_within_training_step = original_mode


@overload
def create_uniform_action_dist(action_space: spaces.Box, batch_size: int = 1) -> dist.Uniform:
    ...


@overload
def create_uniform_action_dist(
    action_space: spaces.Discrete,
    batch_size: int = 1,
) -> dist.Categorical:
    ...


def create_uniform_action_dist(
    action_space: spaces.Box | spaces.Discrete,
    batch_size: int = 1,
) -> dist.Uniform | dist.Categorical:
    """Create a Distribution such that sampling from it is equivalent to sampling a batch with `action_space.sample()`.

    :param action_space: The action space of the environment.
    :param batch_size: The number of environments or batch size for sampling.
    :return: A PyTorch distribution for sampling actions.
    """
    if isinstance(action_space, spaces.Box):
        low = torch.FloatTensor(action_space.low).unsqueeze(0).repeat(batch_size, 1)
        high = torch.FloatTensor(action_space.high).unsqueeze(0).repeat(batch_size, 1)
        return dist.Uniform(low, high)

    elif isinstance(action_space, spaces.Discrete):
        return dist.Categorical(torch.ones(batch_size, int(action_space.n)))

    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")
