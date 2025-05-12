from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, Union, cast

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Independent, Normal
from torch.nn import ParameterList

from tianshou.data import Batch
from tianshou.data.types import (
    DistLogProbBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.exploration import BaseNoise
from tianshou.policy.base import TrainingStats
from tianshou.policy.modelfree.ddpg import ContinuousPolicyWithExplorationNoise
from tianshou.policy.modelfree.td3 import ActorDualCriticsOffPolicyAlgorithm
from tianshou.policy.optim import OptimizerFactory
from tianshou.utils.conversion import to_optional_float
from tianshou.utils.net.continuous import ContinuousActorProb


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


class SACPolicy(ContinuousPolicyWithExplorationNoise):
    def __init__(
        self,
        *,
        actor: torch.nn.Module | ContinuousActorProb,
        exploration_noise: BaseNoise | Literal["default"] | None = None,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: Literal["clip"] | None = "clip",
        action_space: gym.Space,
        observation_space: gym.Space | None = None,
    ):
        """
        :param actor: the actor network following the rules (s -> dist_input_BD)
        :param exploration_noise: add noise to action for exploration.
            This is useful when solving "hard exploration" problems.
            "default" is equivalent to GaussianNoise(sigma=0.1).
        :param deterministic_eval: flag indicating whether the policy should use deterministic
            actions (using the mode of the action distribution) instead of stochastic ones
            (using random sampling) during evaluation.
            When enabled, the policy will always select the most probable action according to
            the learned distribution during evaluation phases, while still using stochastic
            sampling during training. This creates a clear distinction between exploration
            (training) and exploitation (evaluation) behaviors.
            Deterministic actions are generally preferred for final deployment and reproducible
            evaluation as they provide consistent behavior, reduce variance in performance
            metrics, and are more interpretable for human observers.
            Note that this parameter only affects behavior when the policy is not within a
            training step. When collecting rollouts for training, actions remain stochastic
            regardless of this setting to maintain proper exploration behaviour.
        :param action_scaling: flag indicating whether, for continuous action spaces, actions
            should be scaled from the standard neural network output range [-1, 1] to the
            environment's action space range [action_space.low, action_space.high].
            This applies to continuous action spaces only (gym.spaces.Box) and has no effect
            for discrete spaces.
            When enabled, policy outputs are expected to be in the normalized range [-1, 1]
            (after bounding), and are then linearly transformed to the actual required range.
            This improves neural network training stability, allows the same algorithm to work
            across environments with different action ranges, and standardizes exploration
            strategies.
            Should be disabled if the actor model already produces outputs in the correct range.
        :param action_bound_method: the method used for bounding actions in continuous action spaces
            to the range [-1, 1] before scaling them to the environment's action space (provided
            that `action_scaling` is enabled).
            This applies to continuous action spaces only (`gym.spaces.Box`) and should be set to None
            for discrete spaces.
            When set to "clip", actions exceeding the [-1, 1] range are simply clipped to this
            range. When set to "tanh", a hyperbolic tangent function is applied, which smoothly
            constrains outputs to [-1, 1] while preserving gradients.
            The choice of bounding method affects both training dynamics and exploration behavior.
            Clipping provides hard boundaries but may create plateau regions in the gradient
            landscape, while tanh provides smoother transitions but can compress sensitivity
            near the boundaries.
            Should be set to None if the actor model inherently produces bounded outputs.
            Typically used together with `action_scaling=True`.
            NOTE: This parameter has negligible effect since actions are already bounded by tanh
            squashing in the forward method (as in arXiv 1801.01290, Equation 21).
        :param action_space: the environment's action_space.
        :param observation_space: the environment's observation space
        """
        super().__init__(
            exploration_noise=exploration_noise,
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

    @staticmethod
    def from_float_or_instance(alpha: Union[float, "Alpha"]) -> "Alpha":
        if isinstance(alpha, float):
            return FixedAlpha(alpha)
        elif isinstance(alpha, Alpha):
            return alpha
        else:
            raise ValueError(f"Expected float or Alpha instance, but got {alpha=}")

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

    def __init__(self, target_entropy: float, log_alpha: float, optim: OptimizerFactory):
        """
        :param target_entropy: the target entropy value.
            For discrete action spaces, it is usually -log(|A|) for a balance between stochasticity
            and determinism or -log(1/|A|)=log(|A|) for maximum stochasticity or, more generally,
            lambda*log(|A|), e.g. with lambda close to 1 (e.g. 0.98) for pronounced stochasticity.
            For continuous action spaces, it is usually -dim(A) for a balance between stochasticity
            and determinism, with similar generalizations as for discrete action spaces.
        :param log_alpha: the (initial) value of the log of the entropy regularization coefficient alpha.
        :param optim: the factory with which to create the optimizer for `log_alpha`.
        """
        super().__init__()
        self._target_entropy = target_entropy
        self._log_alpha = torch.tensor(log_alpha, requires_grad=True)
        self._optim, lr_scheduler = optim.create_instances(ParameterList([self._log_alpha]))
        if lr_scheduler is not None:
            raise ValueError(
                f"Learning rate schedulers are not supported by {self.__class__.__name__}"
            )

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
    ActorDualCriticsOffPolicyAlgorithm[SACPolicy, DistLogProbBatchProtocol],
    Generic[TSACTrainingStats],
):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905."""

    def __init__(
        self,
        *,
        policy: SACPolicy,
        policy_optim: OptimizerFactory,
        critic: torch.nn.Module,
        critic_optim: OptimizerFactory,
        critic2: torch.nn.Module | None = None,
        critic2_optim: OptimizerFactory | None = None,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float | Alpha = 0.2,
        estimation_step: int = 1,
        deterministic_eval: bool = True,
    ) -> None:
        """
        :param policy: the policy
        :param policy_optim: the optimizer factory for the policy's model.
        :param critic: the first critic network. (s, a -> Q(s, a))
        :param critic_optim: the optimizer factory for the first critic network.
        :param critic2: the second critic network. (s, a -> Q(s, a)).
            If None, copy the first critic (via deepcopy).
        :param critic2_optim: the optimizer factory for the second critic network.
            If None, use the first critic's factory.
        :param tau: the soft update coefficient for target networks, controlling the rate at which
            target networks track the learned networks.
            When the parameters of the target network are updated with the current (source) network's
            parameters, a weighted average is used: target = tau * source + (1 - tau) * target.
            Smaller values (closer to 0) create more stable but slower learning as target networks
            change more gradually. Higher values (closer to 1) allow faster learning but may reduce
            stability.
            Typically set to a small value (0.001 to 0.01) for most reinforcement learning tasks.
        :param gamma: the discount factor in [0, 1] for future rewards.
            This determines how much future rewards are valued compared to immediate ones.
            Lower values (closer to 0) make the agent focus on immediate rewards, creating "myopic"
            behavior. Higher values (closer to 1) make the agent value long-term rewards more,
            potentially improving performance in tasks where delayed rewards are important but
            increasing training variance by incorporating more environmental stochasticity.
            Typically set between 0.9 and 0.99 for most reinforcement learning tasks
        :param alpha: the entropy regularization coefficient, which balances exploration and exploitation.
            This coefficient controls how much the agent values randomness in its policy versus
            pursuing higher rewards.
            Higher values (e.g., 0.5-1.0) strongly encourage exploration by rewarding the agent
            for maintaining diverse action choices, even if this means selecting some lower-value actions.
            Lower values (e.g., 0.01-0.1) prioritize exploitation, allowing the policy to become
            more focused on the highest-value actions.
            A value of 0 would completely remove entropy regularization, potentially leading to
            premature convergence to suboptimal deterministic policies.
            Can be provided as a fixed float (0.2 is a reasonable default) or as an instance of,
            in particular, class `AutoAlpha` for automatic tuning during training.
        :param estimation_step: the number of future steps (> 0) to consider when computing temporal
            difference (TD) targets. Controls the balance between TD learning and Monte Carlo methods:
            higher values reduce bias (by relying less on potentially inaccurate value estimates)
            but increase variance (by incorporating more environmental stochasticity and reducing
            the averaging effect). A value of 1 corresponds to standard TD learning with immediate
            bootstrapping, while very large values approach Monte Carlo-like estimation that uses
            complete episode returns.
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
            estimation_step=estimation_step,
        )
        self.deterministic_eval = deterministic_eval
        self.alpha = Alpha.from_float_or_instance(alpha)
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

    def _update_with_batch(self, batch: RolloutBatchProtocol) -> TSACTrainingStats:  # type: ignore
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
        self.policy_optim.step(actor_loss)

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
