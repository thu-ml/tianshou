from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Independent, Normal

from tianshou.data import Batch
from tianshou.data.types import (
    DistLogProbBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.exploration import BaseNoise
from tianshou.policy.base import Policy, TLearningRateScheduler, TrainingStats
from tianshou.policy.modelfree.td3 import ActorDualCriticsOffPolicyAlgorithm
from tianshou.utils.conversion import to_optional_float
from tianshou.utils.net.continuous import ActorProb


def correct_log_prob_gaussian_tanh(
    log_prob: torch.Tensor,
    tanh_squashed_action: torch.Tensor,
    eps: float = np.finfo(np.float32).eps.item(),
) -> torch.Tensor:
    """Apply correction for Tanh squashing when computing `log_prob` from Gaussian.

    See equation 21 in the original `SAC paper <https://arxiv.org/abs/1801.01290>`_.

    :param log_prob: log probability of the action
    :param tanh_squashed_action: action squashed to values in (-1, 1) range by tanh
    :param eps: epsilon for numerical stability
    """
    log_prob_correction = torch.log(1 - tanh_squashed_action.pow(2) + eps).sum(-1, keepdim=True)
    return log_prob - log_prob_correction


@dataclass(kw_only=True)
class SACTrainingStats(TrainingStats):
    actor_loss: float
    critic1_loss: float
    critic2_loss: float
    alpha: float | None = None
    alpha_loss: float | None = None


TSACTrainingStats = TypeVar("TSACTrainingStats", bound=SACTrainingStats)


class SACPolicy(Policy):
    def __init__(
        self,
        *,
        actor: torch.nn.Module | ActorProb,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: Literal["clip"] | None = "clip",
        action_space: gym.Space,
        observation_space: gym.Space | None = None,
    ):
        """
        :param actor: the actor network following the rules (s -> dist_input_BD)
        :param deterministic_eval: whether to use deterministic action
            (mode of Gaussian policy) in evaluation mode instead of stochastic
            action sampled by the policy. Does not affect training.
        :param action_scaling: whether to map actions from range [-1, 1]
            to range[action_spaces.low, action_spaces.high].
        :param action_bound_method: method to bound action to range [-1, 1],
            can be either "clip" (for simply clipping the action)
            or empty string for no bounding. Only used if the action_space is continuous.
            This parameter is ignored in SAC, which used tanh squashing after sampling
            unbounded from the gaussian policy (as in (arXiv 1801.01290): Equation 21.).
        :param action_space: the action space of the environment
        :param observation_space: the observation space of the environment
        """
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
        )
        self.actor = actor
        self.deterministic_eval = deterministic_eval

    def forward(  # type: ignore
        self,
        batch: ObsBatchProtocol,
        state: dict | Batch | np.ndarray | None = None,
        **kwargs: Any,
    ) -> DistLogProbBatchProtocol:
        (loc_B, scale_B), hidden_BH = self.actor(batch.obs, state=state, info=batch.info)
        dist = Independent(Normal(loc=loc_B, scale=scale_B), 1)
        if self.deterministic_eval and not self.is_within_training_step:
            act_B = dist.mode
        else:
            act_B = dist.rsample()
        log_prob = dist.log_prob(act_B).unsqueeze(-1)

        squashed_action = torch.tanh(act_B)
        log_prob = correct_log_prob_gaussian_tanh(log_prob, squashed_action)
        result = Batch(
            logits=(loc_B, scale_B),
            act=squashed_action,
            state=hidden_BH,
            dist=dist,
            log_prob=log_prob,
        )
        return cast(DistLogProbBatchProtocol, result)


class Alpha(ABC):
    """Defines the interface for the entropy regularization coefficient alpha."""

    @property
    @abstractmethod
    def value(self) -> float:
        """Retrieves the current value of alpha."""

    @abstractmethod
    def update(self, entropy: torch.Tensor) -> float | None:
        """
        Updates the alpha value based on the entropy.

        :param entropy: the entropy of the policy.
        :return: the loss value if alpha is auto-tuned, otherwise None.
        """
        return None


class FixedAlpha(Alpha):
    """Represents a fixed entropy regularization coefficient alpha."""

    def __init__(self, alpha: float):
        self._value = alpha

    @property
    def value(self) -> float:
        return self._value

    def update(self, entropy: torch.Tensor) -> float | None:
        return None


class AutoAlpha(torch.nn.Module, Alpha):
    """Represents an entropy regularization coefficient alpha that is automatically tuned."""

    def __init__(
        self, target_entropy: float, log_alpha: torch.Tensor, optim: torch.optim.Optimizer
    ):
        """
        :param target_entropy: the target entropy value.
            For discrete action spaces, it is usually -log(|A|) for a balance between stochasticity
            and determinism or -log(1/|A|)=log(|A|) for maximum stochasticity or, more generally,
            lambda*log(|A|), e.g. with lambda close to 1 (e.g. 0.98) for pronounced stochasticity.
            For continuous action spaces, it is usually -dim(A) for a balance between stochasticity
            and determinism, with similar generalizations as for discrete action spaces.
        :param log_alpha: the (initial) log of the entropy regularization coefficient alpha.
            This must be a scalar tensor with requires_grad=True.
        :param optim: the optimizer for `log_alpha`.
        """
        super().__init__()
        if not log_alpha.requires_grad:
            raise ValueError("Expected log_alpha to require gradient, but it doesn't.")
        if log_alpha.shape != torch.Size([1]):
            raise ValueError(
                f"Expected log_alpha to have shape torch.Size([1]), "
                f"but got {log_alpha.shape} instead.",
            )
        self._target_entropy = target_entropy
        self._log_alpha = log_alpha
        self._optim = optim

    @property
    def value(self) -> float:
        return self._log_alpha.detach().exp().item()

    def update(self, entropy: torch.Tensor) -> float:
        entropy_deficit = self._target_entropy - entropy
        alpha_loss = -(self._log_alpha * entropy_deficit).mean()
        self._optim.zero_grad()
        alpha_loss.backward()
        self._optim.step()
        return alpha_loss.item()


class SAC(
    ActorDualCriticsOffPolicyAlgorithm[SACPolicy, TSACTrainingStats, DistLogProbBatchProtocol],
    Generic[TSACTrainingStats],
):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905."""

    def __init__(
        self,
        *,
        policy: SACPolicy,
        policy_optim: torch.optim.Optimizer,
        critic: torch.nn.Module,
        critic_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module | None = None,
        critic2_optim: torch.optim.Optimizer | None = None,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float | Alpha = 0.2,
        estimation_step: int = 1,
        exploration_noise: BaseNoise | Literal["default"] | None = None,
        deterministic_eval: bool = True,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        """
        :param policy: the policy
        :param policy_optim: the optimizer for actor network.
        :param critic: the first critic network. (s, a -> Q(s, a))
        :param critic_optim: the optimizer for the first critic network.
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
        :param lr_scheduler: a learning rate scheduler that adjusts the learning rate
            in optimizer in each policy.update()
        """
        super().__init__(
            policy=policy,
            policy_optim=policy_optim,
            critic=critic,
            critic_optim=critic_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=tau,
            gamma=gamma,
            exploration_noise=exploration_noise,
            estimation_step=estimation_step,
            lr_scheduler=lr_scheduler,
        )
        self.deterministic_eval = deterministic_eval
        self.alpha = FixedAlpha(alpha) if isinstance(alpha, float) else alpha
        assert isinstance(self.alpha, Alpha)
        self._check_field_validity()

    def _check_field_validity(self) -> None:
        if not isinstance(self.policy.action_space, gym.spaces.Box):
            raise ValueError(
                f"SACPolicy only supports gym.spaces.Box, but got {self.action_space=}."
                f"Please use DiscreteSACPolicy for discrete action spaces.",
            )

    def _target_q_compute_value(
        self, obs_batch: Batch, act_batch: DistLogProbBatchProtocol
    ) -> torch.Tensor:
        min_q_value = super()._target_q_compute_value(obs_batch, act_batch)
        return min_q_value - self.alpha.value * act_batch.log_prob

    def _update_with_batch(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TSACTrainingStats:  # type: ignore
        # critic 1&2
        td1, critic1_loss = self._minimize_critic_squared_loss(
            batch, self.critic, self.critic_optim
        )
        td2, critic2_loss = self._minimize_critic_squared_loss(
            batch, self.critic2, self.critic2_optim
        )
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        obs_result = self.policy(batch)
        act = obs_result.act
        current_q1a = self.critic(batch.obs, act).flatten()
        current_q2a = self.critic2(batch.obs, act).flatten()
        actor_loss = (
            self.alpha.value * obs_result.log_prob.flatten() - torch.min(current_q1a, current_q2a)
        ).mean()
        self.policy_optim.zero_grad()
        actor_loss.backward()
        self.policy_optim.step()

        # The entropy of a Gaussian policy can be expressed as -log_prob + a constant (which we ignore)
        entropy = -obs_result.log_prob.detach()
        alpha_loss = self.alpha.update(entropy)

        self._update_lagged_network_weights()

        return SACTrainingStats(  # type: ignore[return-value]
            actor_loss=actor_loss.item(),
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
            alpha=to_optional_float(self.alpha.value),
            alpha_loss=to_optional_float(alpha_loss),
        )
