from dataclasses import dataclass
from typing import Any, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from overrides import override
from torch.distributions import Categorical

from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ObsBatchProtocol, RolloutBatchProtocol
from tianshou.policy import SACPolicy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.policy.modelfree.sac import SACTrainingStats


@dataclass
class DiscreteSACTrainingStats(SACTrainingStats):
    pass


TDiscreteSACTrainingStats = TypeVar("TDiscreteSACTrainingStats", bound=DiscreteSACTrainingStats)


class DiscreteSACPolicy(SACPolicy[TDiscreteSACTrainingStats]):
    """Implementation of SAC for Discrete Action Settings. arXiv:1910.07207.

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
    :param alpha: entropy regularization coefficient.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided,
        then alpha is automatically tuned.
    :param estimation_step: the number of steps to look ahead for calculating
    :param observation_space: Env's observation space.
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
        action_space: gym.spaces.Discrete,
        critic2: torch.nn.Module | None = None,
        critic2_optim: torch.optim.Optimizer | None = None,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float | tuple[float, torch.Tensor, torch.optim.Optimizer] = 0.2,
        estimation_step: int = 1,
        observation_space: gym.Space | None = None,
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
            alpha=alpha,
            estimation_step=estimation_step,
            # Note: inheriting from continuous sac reduces code duplication,
            # but continuous stuff has to be disabled
            exploration_noise=None,
            action_scaling=False,
            action_bound_method=None,
            observation_space=observation_space,
            lr_scheduler=lr_scheduler,
        )

    # TODO: violates Liskov substitution principle, incompatible action space with SAC
    #   Not too urgent, but still..
    @override
    def _check_field_validity(self) -> None:
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"DiscreteSACPolicy only supports gym.spaces.Discrete, but got {self.action_space=}."
                f"Please use SACPolicy for continuous action spaces.",
            )

    def forward(  # type: ignore
        self,
        batch: ObsBatchProtocol,
        state: dict | Batch | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Batch:
        logits, hidden = self.actor(batch.obs, state=state, info=batch.info)
        dist = Categorical(logits=logits)
        if self.deterministic_eval and not self.training:
            act = dist.mode
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # obs_next: s_{t+n}
        obs_next_result = self(obs_next_batch)
        dist = obs_next_result.dist
        target_q = dist.probs * torch.min(
            self.critic_old(obs_next_batch.obs),
            self.critic2_old(obs_next_batch.obs),
        )
        return target_q.sum(dim=-1) + self.alpha * dist.entropy()

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TDiscreteSACTrainingStats:  # type: ignore
        weight = batch.pop("weight", 1.0)
        target_q = batch.returns.flatten()
        act = to_torch(batch.act[:, np.newaxis], device=target_q.device, dtype=torch.long)

        # critic 1
        current_q1 = self.critic(batch.obs).gather(1, act).flatten()
        td1 = current_q1 - target_q
        critic1_loss = (td1.pow(2) * weight).mean()

        self.critic_optim.zero_grad()
        critic1_loss.backward()
        self.critic_optim.step()

        # critic 2
        current_q2 = self.critic2(batch.obs).gather(1, act).flatten()
        td2 = current_q2 - target_q
        critic2_loss = (td2.pow(2) * weight).mean()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        dist = self(batch).dist
        entropy = dist.entropy()
        with torch.no_grad():
            current_q1a = self.critic(batch.obs)
            current_q2a = self.critic2(batch.obs)
            q = torch.min(current_q1a, current_q2a)
        actor_loss = -(self.alpha * entropy + (dist.probs * q).sum(dim=-1)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.is_auto_alpha:
            log_prob = -entropy.detach() + self.target_entropy
            alpha_loss = -(self.log_alpha * log_prob).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()

        self.sync_weight()

        if self.is_auto_alpha:
            self.alpha = cast(torch.Tensor, self.alpha)

        return DiscreteSACTrainingStats(  # type: ignore[return-value]
            actor_loss=actor_loss.item(),
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
            alpha=self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
            alpha_loss=None if not self.is_auto_alpha else alpha_loss.item(),
        )

    def exploration_noise(
        self,
        act: np.ndarray | BatchProtocol,
        batch: RolloutBatchProtocol,
    ) -> np.ndarray | BatchProtocol:
        return act
