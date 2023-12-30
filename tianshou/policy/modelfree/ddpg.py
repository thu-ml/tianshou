import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    ActStateBatchProtocol,
    BatchWithReturnsProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.policy import BasePolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats


@dataclass(kw_only=True)
class DDPGTrainingStats(TrainingStats):
    actor_loss: float
    critic_loss: float


TDDPGTrainingStats = TypeVar("TDDPGTrainingStats", bound=DDPGTrainingStats)


class DDPGPolicy(BasePolicy[TDDPGTrainingStats], Generic[TDDPGTrainingStats]):
    """Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971.

    :param actor: The actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> model_output)
    :param actor_optim: The optimizer for actor network.
    :param critic: The critic network. (s, a -> Q(s, a))
    :param critic_optim: The optimizer for critic network.
    :param action_space: Env's action space.
    :param tau: Param for soft update of the target network.
    :param gamma: Discount factor, in [0, 1].
    :param exploration_noise: The exploration noise, added to the action. Defaults
        to ``GaussianNoise(sigma=0.1)``.
    :param estimation_step: The number of steps to look ahead.
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
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic: torch.nn.Module,
        critic_optim: torch.optim.Optimizer,
        action_space: gym.Space,
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: BaseNoise | Literal["default"] | None = "default",
        estimation_step: int = 1,
        observation_space: gym.Space | None = None,
        action_scaling: bool = True,
        # tanh not supported, see assert below
        action_bound_method: Literal["clip"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        assert 0.0 <= tau <= 1.0, f"tau should be in [0, 1] but got: {tau}"
        assert 0.0 <= gamma <= 1.0, f"gamma should be in [0, 1] but got: {gamma}"
        assert action_bound_method != "tanh", (  # type: ignore[comparison-overlap]
            "tanh mapping is not supported"
            "in policies where action is used as input of critic , because"
            "raw action in range (-inf, inf) will cause instability in training"
        )
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        if action_scaling and not np.isclose(actor.max_action, 1.0):
            warnings.warn(
                "action_scaling and action_bound_method are only intended to deal"
                "with unbounded model action space, but find actor model bound"
                f"action space with max_action={actor.max_action}."
                "Consider using unbounded=True option of the actor model,"
                "or set action_scaling to False and action_bound_method to None.",
            )
        self.actor = actor
        self.actor_old = deepcopy(actor)
        self.actor_old.eval()
        self.actor_optim = actor_optim
        self.critic = critic
        self.critic_old = deepcopy(critic)
        self.critic_old.eval()
        self.critic_optim = critic_optim
        self.tau = tau
        self.gamma = gamma
        if exploration_noise == "default":
            exploration_noise = GaussianNoise(sigma=0.1)
        # TODO: IMPORTANT - can't call this "exploration_noise" because confusingly,
        #  there is already a method called exploration_noise() in the base class
        #  Now this method doesn't apply any noise and is also not overridden. See TODO there
        self._exploration_noise = exploration_noise
        # it is only a little difference to use GaussianNoise
        # self.noise = OUNoise()
        self.estimation_step = estimation_step

    def set_exp_noise(self, noise: BaseNoise | None) -> None:
        """Set the exploration noise."""
        self._exploration_noise = noise

    def train(self, mode: bool = True) -> Self:
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        self.soft_update(self.actor_old, self.actor, self.tau)
        self.soft_update(self.critic_old, self.critic, self.tau)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # obs_next: s_{t+n}
        return self.critic_old(obs_next_batch.obs, self(obs_next_batch, model="actor_old").act)

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol | BatchWithReturnsProtocol:
        return self.compute_nstep_return(
            batch=batch,
            buffer=buffer,
            indices=indices,
            target_q_fn=self._target_q,
            gamma=self.gamma,
            n_step=self.estimation_step,
        )

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        model: Literal["actor", "actor_old"] = "actor",
        **kwargs: Any,
    ) -> ActStateBatchProtocol:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        actions, hidden = model(batch.obs, state=state, info=batch.info)
        return cast(ActStateBatchProtocol, Batch(act=actions, state=hidden))

    @staticmethod
    def _mse_optimizer(
        batch: RolloutBatchProtocol,
        critic: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        # critic_loss = F.mse_loss(current_q1, target_q)
        critic_loss = (td.pow(2) * weight).mean()
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return td, critic_loss

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TDDPGTrainingStats:  # type: ignore
        # critic
        td, critic_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        batch.weight = td  # prio-buffer
        # actor
        actor_loss = -self.critic(batch.obs, self(batch).act).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()

        return DDPGTrainingStats(actor_loss=actor_loss.item(), critic_loss=critic_loss.item())  # type: ignore[return-value]

    def exploration_noise(
        self,
        act: np.ndarray | BatchProtocol,
        batch: RolloutBatchProtocol,
    ) -> np.ndarray | BatchProtocol:
        if self._exploration_noise is None:
            return act
        if isinstance(act, np.ndarray):
            return act + self._exploration_noise(act.shape)
        warnings.warn("Cannot add exploration noise to non-numpy_array action.")
        return act
