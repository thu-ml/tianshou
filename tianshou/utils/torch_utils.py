from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

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
