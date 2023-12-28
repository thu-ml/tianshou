from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, TypeVar

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol
from tianshou.exploration import BaseNoise
from tianshou.policy import DDPGPolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.utils.optim import clone_optimizer


@dataclass(kw_only=True)
class TD3TrainingStats(TrainingStats):
    actor_loss: float
    critic1_loss: float
    critic2_loss: float


TTD3TrainingStats = TypeVar("TTD3TrainingStats", bound=TD3TrainingStats)


# TODO: the type ignore here is needed b/c the hierarchy is actually broken! Should reconsider the inheritance structure.
class TD3Policy(DDPGPolicy[TTD3TrainingStats], Generic[TTD3TrainingStats]):  # type: ignore[type-var]
    """Implementation of TD3, arXiv:1802.09477.

    :param actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param actor_optim: the optimizer for actor network.
    :param critic: the first critic network. (s, a -> Q(s, a))
    :param critic_optim: the optimizer for the first critic network.
    :param action_space: Env's action space. Should be gym.spaces.Box.
    :param critic2: the second critic network. (s, a -> Q(s, a)).
        If None, use the same network as critic (via deepcopy).
    :param critic2_optim: the optimizer for the second critic network.
        If None, clone critic_optim to use for critic2.parameters().
    :param tau: param for soft update of the target network.
    :param gamma: discount factor, in [0, 1].
    :param exploration_noise: add noise to action for exploration.
        This is useful when solving "hard exploration" problems.
        "default" is equivalent to GaussianNoise(sigma=0.1).
    :param policy_noise: the noise used in updating policy network.
    :param update_actor_freq: the update frequency of actor network.
    :param noise_clip: the clipping range used in updating policy network.
    :param observation_space: Env's observation space.
    :param action_scaling: if True, scale the action from [-1, 1] to the range
        of action_space. Only used if the action_space is continuous.
    :param action_bound_method: method to bound action to range [-1, 1].
        Only used if the action_space is continuous.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate
        in optimizer in each policy.update()

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
        critic2: torch.nn.Module | None = None,
        critic2_optim: torch.optim.Optimizer | None = None,
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: BaseNoise | Literal["default"] | None = "default",
        policy_noise: float = 0.2,
        update_actor_freq: int = 2,
        noise_clip: float = 0.5,
        estimation_step: int = 1,
        observation_space: gym.Space | None = None,
        action_scaling: bool = True,
        action_bound_method: Literal["clip"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        # TODO: reduce duplication with SAC.
        #  Some intermediate class, like TwoCriticPolicy?
        super().__init__(
            actor=actor,
            actor_optim=actor_optim,
            critic=critic,
            critic_optim=critic_optim,
            action_space=action_space,
            tau=tau,
            gamma=gamma,
            exploration_noise=exploration_noise,
            estimation_step=estimation_step,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=observation_space,
            lr_scheduler=lr_scheduler,
        )
        if critic2 and not critic2_optim:
            raise ValueError("critic2_optim must be provided if critic2 is provided")
        critic2 = critic2 or deepcopy(critic)
        critic2_optim = critic2_optim or clone_optimizer(critic_optim, critic2.parameters())
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim

        self.policy_noise = policy_noise
        self.update_actor_freq = update_actor_freq
        self.noise_clip = noise_clip
        self._cnt = 0
        self._last = 0

    def train(self, mode: bool = True) -> Self:
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        self.soft_update(self.critic_old, self.critic, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)
        self.soft_update(self.actor_old, self.actor, self.tau)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # obs_next: s_{t+n}
        act_ = self(obs_next_batch, model="actor_old").act
        noise = torch.randn(size=act_.shape, device=act_.device) * self.policy_noise
        if self.noise_clip > 0.0:
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
        act_ += noise
        return torch.min(
            self.critic_old(obs_next_batch.obs, act_),
            self.critic2_old(obs_next_batch.obs, act_),
        )

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TTD3TrainingStats:  # type: ignore
        # critic 1&2
        td1, critic1_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        td2, critic2_loss = self._mse_optimizer(batch, self.critic2, self.critic2_optim)
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        if self._cnt % self.update_actor_freq == 0:
            actor_loss = -self.critic(batch.obs, self(batch, eps=0.0).act).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self._last = actor_loss.item()
            self.actor_optim.step()
            self.sync_weight()
        self._cnt += 1

        return TD3TrainingStats(  # type: ignore[return-value]
            actor_loss=self._last,
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
        )
