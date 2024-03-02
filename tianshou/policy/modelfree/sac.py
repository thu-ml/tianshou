from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Independent, Normal

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.types import (
    DistLogProbBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.exploration import BaseNoise
from tianshou.policy import DDPGPolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.utils.conversion import to_optional_float
from tianshou.utils.optim import clone_optimizer


@dataclass(kw_only=True)
class SACTrainingStats(TrainingStats):
    actor_loss: float
    critic1_loss: float
    critic2_loss: float
    alpha: float | None = None
    alpha_loss: float | None = None


TSACTrainingStats = TypeVar("TSACTrainingStats", bound=SACTrainingStats)


# TODO: the type ignore here is needed b/c the hierarchy is actually broken! Should reconsider the inheritance structure.
class SACPolicy(DDPGPolicy[TSACTrainingStats], Generic[TSACTrainingStats]):  # type: ignore[type-var]
    """Implementation of Soft Actor-Critic. arXiv:1812.05905.

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
    :param estimation_step: The number of steps to look ahead.
    :param exploration_noise: add noise to action for exploration.
        This is useful when solving "hard exploration" problems.
        "default" is equivalent to GaussianNoise(sigma=0.1).
    :param deterministic_eval: whether to use deterministic action
        (mode of Gaussian policy) in evaluation mode instead of stochastic
        action sampled by the policy. Does not affect training.
    :param action_scaling: whether to map actions from range [-1, 1]
        to range[action_spaces.low, action_spaces.high].
    :param action_bound_method: method to bound action to range [-1, 1],
        can be either "clip" (for simply clipping the action)
        or empty string for no bounding. Only used if the action_space is continuous.
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
        action_space: gym.Space,
        critic2: torch.nn.Module | None = None,
        critic2_optim: torch.optim.Optimizer | None = None,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float | tuple[float, torch.Tensor, torch.optim.Optimizer] = 0.2,
        estimation_step: int = 1,
        exploration_noise: BaseNoise | Literal["default"] | None = None,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        # TODO: some papers claim that tanh is crucial for SAC, yet DDPG will raise an
        #  error if tanh is used. Should be investigated.
        action_bound_method: Literal["clip"] | None = "clip",
        observation_space: gym.Space | None = None,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
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
        critic2 = critic2 or deepcopy(critic)
        critic2_optim = critic2_optim or clone_optimizer(critic_optim, critic2.parameters())
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self.deterministic_eval = deterministic_eval
        self.__eps = np.finfo(np.float32).eps.item()

        self.alpha: float | torch.Tensor
        self._is_auto_alpha = not isinstance(alpha, float)
        if self._is_auto_alpha:
            # TODO: why doesn't mypy understand that this must be a tuple?
            alpha = cast(tuple[float, torch.Tensor, torch.optim.Optimizer], alpha)
            if alpha[1].shape != torch.Size([1]):
                raise ValueError(
                    f"Expected log_alpha to have shape torch.Size([1]), "
                    f"but got {alpha[1].shape} instead.",
                )
            if not alpha[1].requires_grad:
                raise ValueError("Expected log_alpha to require gradient, but it doesn't.")

            self.target_entropy, self.log_alpha, self.alpha_optim = alpha
            self.alpha = self.log_alpha.detach().exp()
        else:
            alpha = cast(
                float,
                alpha,
            )  # can we convert alpha to a constant tensor here? then mypy wouldn't complain
            self.alpha = alpha

        # TODO or not TODO: add to BasePolicy?
        self._check_field_validity()

    def _check_field_validity(self) -> None:
        if not isinstance(self.action_space, gym.spaces.Box):
            raise ValueError(
                f"SACPolicy only supports gym.spaces.Box, but got {self.action_space=}."
                f"Please use DiscreteSACPolicy for discrete action spaces.",
            )

    @property
    def is_auto_alpha(self) -> bool:
        return self._is_auto_alpha

    def train(self, mode: bool = True) -> Self:
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        self.soft_update(self.critic_old, self.critic, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)

    # TODO: violates Liskov substitution principle
    def forward(  # type: ignore
        self,
        batch: ObsBatchProtocol,
        state: dict | Batch | np.ndarray | None = None,
        **kwargs: Any,
    ) -> DistLogProbBatchProtocol:
        logits, hidden = self.actor(batch.obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)
        dist = Independent(Normal(*logits), 1)
        if self.deterministic_eval and not self.training:
            act = dist.mode
        else:
            act = dist.rsample()
        log_prob = dist.log_prob(act).unsqueeze(-1)
        # apply correction for Tanh squashing when computing logprob from Gaussian
        # You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
        # in appendix C to get some understanding of this equation.
        squashed_action = torch.tanh(act)
        log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) + self.__eps).sum(
            -1,
            keepdim=True,
        )
        result = Batch(
            logits=logits,
            act=squashed_action,
            state=hidden,
            dist=dist,
            log_prob=log_prob,
        )
        return cast(DistLogProbBatchProtocol, result)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # obs_next: s_{t+n}
        obs_next_result = self(obs_next_batch)
        act_ = obs_next_result.act
        return (
            torch.min(
                self.critic_old(obs_next_batch.obs, act_),
                self.critic2_old(obs_next_batch.obs, act_),
            )
            - self.alpha * obs_next_result.log_prob
        )

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TSACTrainingStats:  # type: ignore
        # critic 1&2
        td1, critic1_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        td2, critic2_loss = self._mse_optimizer(batch, self.critic2, self.critic2_optim)
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        obs_result = self(batch)
        act = obs_result.act
        current_q1a = self.critic(batch.obs, act).flatten()
        current_q2a = self.critic2(batch.obs, act).flatten()
        actor_loss = (
            self.alpha * obs_result.log_prob.flatten() - torch.min(current_q1a, current_q2a)
        ).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        alpha_loss = None

        if self.is_auto_alpha:
            log_prob = obs_result.log_prob.detach() + self.target_entropy
            # please take a look at issue #258 if you'd like to change this line
            alpha_loss = -(self.log_alpha * log_prob).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()

        self.sync_weight()

        return SACTrainingStats(  # type: ignore[return-value]
            actor_loss=actor_loss.item(),
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
            alpha=to_optional_float(self.alpha),
            alpha_loss=to_optional_float(alpha_loss),
        )
