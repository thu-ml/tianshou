import warnings
from abc import ABC
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from sensai.util.helper import mark_used

from tianshou.algorithm import Algorithm
from tianshou.algorithm.algorithm_base import (
    LaggedNetworkPolyakUpdateAlgorithmMixin,
    OffPolicyAlgorithm,
    Policy,
    TArrOrActBatch,
    TPolicy,
    TrainingStats,
)
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import Batch, ReplayBuffer
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    ActBatchProtocol,
    ActStateBatchProtocol,
    BatchWithReturnsProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.utils.net.continuous import (
    ContinuousActorDeterministicInterface,
    ContinuousCritic,
)

mark_used(ActBatchProtocol)


@dataclass(kw_only=True)
class DDPGTrainingStats(TrainingStats):
    actor_loss: float
    critic_loss: float


class ContinuousPolicyWithExplorationNoise(Policy, ABC):
    def __init__(
        self,
        *,
        exploration_noise: BaseNoise | Literal["default"] | None = None,
        action_space: gym.Space,
        observation_space: gym.Space | None = None,
        action_scaling: bool = True,
        action_bound_method: Literal["clip"] | None = "clip",
    ):
        """
        :param exploration_noise: noise model for adding noise to continuous actions
            for exploration. This is useful when solving "hard exploration" problems.
            "default" is equivalent to GaussianNoise(sigma=0.1).
        :param action_space: the environment's action_space.
        :param observation_space: the environment's observation space
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
        """
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
        )
        if exploration_noise == "default":
            exploration_noise = GaussianNoise(sigma=0.1)
        self.exploration_noise = exploration_noise

    def set_exploration_noise(self, noise: BaseNoise | None) -> None:
        """Set the exploration noise."""
        self.exploration_noise = noise

    def add_exploration_noise(
        self,
        act: TArrOrActBatch,
        batch: ObsBatchProtocol,
    ) -> TArrOrActBatch:
        if self.exploration_noise is None:
            return act
        if isinstance(act, np.ndarray):
            return act + self.exploration_noise(act.shape)
        warnings.warn("Cannot add exploration noise to non-numpy_array action.")
        return act


class ContinuousDeterministicPolicy(ContinuousPolicyWithExplorationNoise):
    """A policy for continuous action spaces that uses an actor which directly maps states to actions."""

    def __init__(
        self,
        *,
        actor: ContinuousActorDeterministicInterface,
        exploration_noise: BaseNoise | Literal["default"] | None = None,
        action_space: gym.Space,
        observation_space: gym.Space | None = None,
        action_scaling: bool = True,
        action_bound_method: Literal["clip"] | None = "clip",
    ):
        """
        :param actor: The actor network following the rules (s -> actions)
        :param exploration_noise: add noise to continuous actions for exploration;
            set to None for discrete action spaces.
            This is useful when solving "hard exploration" problems.
            "default" is equivalent to GaussianNoise(sigma=0.1).
        :param action_space: the environment's action space.
        :param tau: the soft update coefficient for target networks, controlling the rate at which
            target networks track the learned networks.
            When the parameters of the target network are updated with the current (source) network's
            parameters, a weighted average is used: target = tau * source + (1 - tau) * target.
            Smaller values (closer to 0) create more stable but slower learning as target networks
            change more gradually. Higher values (closer to 1) allow faster learning but may reduce
            stability.
            Typically set to a small value (0.001 to 0.01) for most reinforcement learning tasks.
        :param observation_space: the environment's observation space.
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
        :param action_bound_method: method to bound action to range [-1, 1].
        """
        if action_scaling and not np.isclose(actor.max_action, 1.0):
            warnings.warn(
                "action_scaling and action_bound_method are only intended to deal"
                "with unbounded model action space, but find actor model bound"
                f"action space with max_action={actor.max_action}."
                "Consider using unbounded=True option of the actor model,"
                "or set action_scaling to False and action_bound_method to None.",
            )
        super().__init__(
            exploration_noise=exploration_noise,
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
        )
        self.actor = actor

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> ActStateBatchProtocol:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.
        """
        if model is None:
            model = self.actor
        actions, hidden = model(batch.obs, state=state, info=batch.info)
        return cast(ActStateBatchProtocol, Batch(act=actions, state=hidden))


TActBatchProtocol = TypeVar("TActBatchProtocol", bound=ActBatchProtocol)


class ActorCriticOffPolicyAlgorithm(
    OffPolicyAlgorithm[TPolicy],
    LaggedNetworkPolyakUpdateAlgorithmMixin,
    Generic[TPolicy, TActBatchProtocol],
    ABC,
):
    """Base class for actor-critic off-policy algorithms that use a lagged critic
    as a target network.

    Its implementation of `process_fn` adds the n-step return to the batch, using the
    Q-values computed by the target network (lagged critic, `critic_old`) in order to compute the
    reward-to-go.

    Specializations can override the action computation (`_target_q_compute_action`) or the
    Q-value computation based on these actions (`_target_q_compute_value`) to customize the
    target Q-value computation.
    The default implementation assumes a continuous action space where a critic receives a
    state/observation and an action; for discrete action spaces, where the critic receives only
    a state/observation, the method `_target_q_compute_value` must be overridden.
    """

    def __init__(
        self,
        *,
        policy: TPolicy,
        policy_optim: OptimizerFactory,
        critic: torch.nn.Module,
        critic_optim: OptimizerFactory,
        tau: float = 0.005,
        gamma: float = 0.99,
        n_step_return_horizon: int = 1,
    ) -> None:
        """
        :param policy: the policy
        :param policy_optim: the optimizer factory for the policy's model.
        :param critic: the critic network.
            For continuous action spaces: (s, a -> Q(s, a)).
            For discrete action spaces: (s -> <Q(s, a_1), ..., Q(s, a_N)>).
            NOTE: The default implementation of `_target_q_compute_value` assumes
                a continuous action space; override this method if using discrete actions.
        :param critic_optim: the optimizer factory for the critic network.
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
        """
        assert 0.0 <= tau <= 1.0, f"tau should be in [0, 1] but got: {tau}"
        assert 0.0 <= gamma <= 1.0, f"gamma should be in [0, 1] but got: {gamma}"
        super().__init__(
            policy=policy,
        )
        LaggedNetworkPolyakUpdateAlgorithmMixin.__init__(self, tau=tau)
        self.policy_optim = self._create_optimizer(policy, policy_optim)
        self.critic = critic
        self.critic_old = self._add_lagged_network(self.critic)
        self.critic_optim = self._create_optimizer(self.critic, critic_optim)
        self.gamma = gamma
        self.n_step_return_horizon = n_step_return_horizon

    @staticmethod
    def _minimize_critic_squared_loss(
        batch: RolloutBatchProtocol,
        critic: torch.nn.Module,
        optimizer: Algorithm.Optimizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Takes an optimizer step to minimize the squared loss of the critic given a batch of data.

        :param batch: the batch containing the observations, actions, returns, and (optionally) weights.
        :param critic: the critic network to minimize the loss for.
        :param optimizer: the optimizer for the critic's parameters.
        :return: a pair (`td`, `loss`), where `td` is the tensor of errors (current - target) and `loss` is the MSE loss.
        """
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        critic_loss = (td.pow(2) * weight).mean()
        optimizer.step(critic_loss)
        return td, critic_loss

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol | BatchWithReturnsProtocol:
        # add the n-step return to the batch, which the critic (Q-functions) seeks to match,
        # based the Q-values computed by the target network (lagged critic)
        return self.compute_nstep_return(
            batch=batch,
            buffer=buffer,
            indices=indices,
            target_q_fn=self._target_q,
            gamma=self.gamma,
            n_step=self.n_step_return_horizon,
        )

    def _target_q_compute_action(self, obs_batch: Batch) -> TActBatchProtocol:
        """
        Computes the action to be taken for the given batch (containing the observations)
        within the context of Q-value target computation.

        :param obs_batch: the batch containing the observations.
        :return: batch containing the actions to be taken.
        """
        return self.policy(obs_batch)

    def _target_q_compute_value(
        self, obs_batch: Batch, act_batch: TActBatchProtocol
    ) -> torch.Tensor:
        """
        Computes the target Q-value given a batch with observations and actions taken.

        :param obs_batch: the batch containing the observations.
        :param act_batch: the batch containing the actions taken.
        :return: a tensor containing the target Q-values.
        """
        # compute the target Q-value using the lagged critic network (target network)
        return self.critic_old(obs_batch.obs, act_batch.act)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """
        Computes the target Q-value for the given buffer and indices.

        :param buffer: the replay buffer
        :param indices: the indices within the buffer to compute the target Q-value for
        """
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # obs_next: s_{t+n}
        act_batch = self._target_q_compute_action(obs_next_batch)
        return self._target_q_compute_value(obs_next_batch, act_batch)


class DDPG(
    ActorCriticOffPolicyAlgorithm[ContinuousDeterministicPolicy, ActBatchProtocol],
):
    """Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971."""

    def __init__(
        self,
        *,
        policy: ContinuousDeterministicPolicy,
        policy_optim: OptimizerFactory,
        critic: torch.nn.Module | ContinuousCritic,
        critic_optim: OptimizerFactory,
        tau: float = 0.005,
        gamma: float = 0.99,
        n_step_return_horizon: int = 1,
    ) -> None:
        """
        :param policy: the policy
        :param policy_optim: the optimizer factory for the policy's model.
        :param critic: the critic network. (s, a -> Q(s, a))
        :param critic_optim: the optimizer factory for the critic network.
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
        :param n_step_return_horizon: the number of future steps (> 0) to consider when computing temporal
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
            tau=tau,
            gamma=gamma,
            n_step_return_horizon=n_step_return_horizon,
        )
        self.actor_old = self._add_lagged_network(self.policy.actor)

    def _target_q_compute_action(self, obs_batch: Batch) -> ActBatchProtocol:
        # compute the action using the lagged actor network
        return self.policy(obs_batch, model=self.actor_old)

    def _update_with_batch(self, batch: RolloutBatchProtocol) -> DDPGTrainingStats:
        # critic
        td, critic_loss = self._minimize_critic_squared_loss(batch, self.critic, self.critic_optim)
        batch.weight = td  # prio-buffer
        # actor
        actor_loss = -self.critic(batch.obs, self.policy(batch).act).mean()
        self.policy_optim.step(actor_loss)
        self._update_lagged_network_weights()

        return DDPGTrainingStats(actor_loss=actor_loss.item(), critic_loss=critic_loss.item())
