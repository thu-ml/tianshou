from typing import Any, Literal, Self

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import (
    TLearningRateScheduler,
    TrainingStats,
    TrainingStatsWrapper,
    TTrainingStats,
)
from tianshou.utils.net.discrete import IntrinsicCuriosityModule


class ICMTrainingStats(TrainingStatsWrapper):
    def __init__(
        self,
        wrapped_stats: TrainingStats,
        *,
        icm_loss: float,
        icm_forward_loss: float,
        icm_inverse_loss: float,
    ) -> None:
        self.icm_loss = icm_loss
        self.icm_forward_loss = icm_forward_loss
        self.icm_inverse_loss = icm_inverse_loss
        super().__init__(wrapped_stats)


class ICMPolicy(BasePolicy[ICMTrainingStats]):
    """Implementation of Intrinsic Curiosity Module. arXiv:1705.05363.

    :param policy: a base policy to add ICM to.
    :param model: the ICM model.
    :param optim: a torch.optim for optimizing the model.
    :param lr_scale: the scaling factor for ICM learning.
    :param forward_loss_weight: the weight for forward model loss.
    :param observation_space: Env's observation space.
    :param action_scaling: if True, scale the action from [-1, 1] to the range
        of action_space. Only used if the action_space is continuous.
    :param action_bound_method: method to bound action to range [-1, 1].
        Only used if the action_space is continuous.
    :param lr_scheduler: if not None, will be called in `policy.update()`.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        *,
        policy: BasePolicy[TTrainingStats],
        model: IntrinsicCuriosityModule,
        optim: torch.optim.Optimizer,
        lr_scale: float,
        reward_scale: float,
        forward_loss_weight: float,
        action_space: gym.Space,
        observation_space: gym.Space | None = None,
        action_scaling: bool = False,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        self.policy = policy
        self.model = model
        self.optim = optim
        self.lr_scale = lr_scale
        self.reward_scale = reward_scale
        self.forward_loss_weight = forward_loss_weight

    def train(self, mode: bool = True) -> Self:
        """Set the module in training mode."""
        self.policy.train(mode)
        self.training = mode
        self.model.train(mode)
        return self

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:
        """Compute action over the given batch data by inner policy.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        return self.policy.forward(batch, state, **kwargs)

    def exploration_noise(
        self,
        act: np.ndarray | BatchProtocol,
        batch: RolloutBatchProtocol,
    ) -> np.ndarray | BatchProtocol:
        return self.policy.exploration_noise(act, batch)

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        if hasattr(self.policy, "set_eps"):
            self.policy.set_eps(eps)
        else:
            raise NotImplementedError

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        """Pre-process the data from the provided replay buffer.

        Used in :meth:`update`. Check out :ref:`process_fn` for more information.
        """
        mse_loss, act_hat = self.model(batch.obs, batch.act, batch.obs_next)
        batch.policy = Batch(orig_rew=batch.rew, act_hat=act_hat, mse_loss=mse_loss)
        batch.rew += to_numpy(mse_loss * self.reward_scale)
        return self.policy.process_fn(batch, buffer, indices)

    def post_process_fn(
        self,
        batch: BatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> None:
        """Post-process the data from the provided replay buffer.

        Typical usage is to update the sampling weight in prioritized
        experience replay. Used in :meth:`update`.
        """
        self.policy.post_process_fn(batch, buffer, indices)
        batch.rew = batch.policy.orig_rew  # restore original reward

    def learn(
        self,
        batch: RolloutBatchProtocol,
        *args: Any,
        **kwargs: Any,
    ) -> ICMTrainingStats:
        training_stat = self.policy.learn(batch, **kwargs)
        self.optim.zero_grad()
        act_hat = batch.policy.act_hat
        act = to_torch(batch.act, dtype=torch.long, device=act_hat.device)
        inverse_loss = F.cross_entropy(act_hat, act).mean()
        forward_loss = batch.policy.mse_loss.mean()
        loss = (
            (1 - self.forward_loss_weight) * inverse_loss + self.forward_loss_weight * forward_loss
        ) * self.lr_scale
        loss.backward()
        self.optim.step()

        return ICMTrainingStats(
            training_stat,
            icm_loss=loss.item(),
            icm_forward_loss=forward_loss.item(),
            icm_inverse_loss=inverse_loss.item(),
        )
