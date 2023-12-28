from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import gymnasium as gym
import torch
import torch.nn.functional as F

from tianshou.data import to_torch_as
from tianshou.data.types import RolloutBatchProtocol
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.policy import TD3Policy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.policy.modelfree.td3 import TD3TrainingStats


@dataclass(kw_only=True)
class TD3BCTrainingStats(TD3TrainingStats):
    pass


TTD3BCTrainingStats = TypeVar("TTD3BCTrainingStats", bound=TD3BCTrainingStats)


class TD3BCPolicy(TD3Policy[TTD3BCTrainingStats]):
    """Implementation of TD3+BC. arXiv:2106.06860.

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
    :param alpha: the value of alpha, which controls the weight for TD3 learning
        relative to behavior cloning.
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
        exploration_noise: BaseNoise | None = GaussianNoise(sigma=0.1),
        policy_noise: float = 0.2,
        update_actor_freq: int = 2,
        noise_clip: float = 0.5,
        # TODO: same name as alpha in SAC and REDQ, which also inherit from DDPGPolicy. Rename?
        alpha: float = 2.5,
        estimation_step: int = 1,
        observation_space: gym.Space | None = None,
        action_scaling: bool = True,
        action_bound_method: Literal["clip"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            actor_optim=actor_optim,
            critic=critic,
            critic_optim=critic_optim,
            action_space=action_space,
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=tau,
            gamma=gamma,
            exploration_noise=exploration_noise,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            update_actor_freq=update_actor_freq,
            estimation_step=estimation_step,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=observation_space,
            lr_scheduler=lr_scheduler,
        )
        self.alpha = alpha

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TTD3BCTrainingStats:  # type: ignore
        # critic 1&2
        td1, critic1_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        td2, critic2_loss = self._mse_optimizer(batch, self.critic2, self.critic2_optim)
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        if self._cnt % self.update_actor_freq == 0:
            act = self(batch, eps=0.0).act
            q_value = self.critic(batch.obs, act)
            lmbda = self.alpha / q_value.abs().mean().detach()
            actor_loss = -lmbda * q_value.mean() + F.mse_loss(act, to_torch_as(batch.act, act))
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self._last = actor_loss.item()
            self.actor_optim.step()
            self.sync_weight()
        self._cnt += 1

        return TD3BCTrainingStats(  # type: ignore[return-value]
            actor_loss=self._last,
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
        )
