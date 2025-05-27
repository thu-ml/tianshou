from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from tianshou.algorithm.algorithm_base import (
    TPolicy,
    TrainingStats,
)
from tianshou.algorithm.modelfree.ddpg import (
    ActorCriticOffPolicyAlgorithm,
    ContinuousDeterministicPolicy,
    TActBatchProtocol,
)
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import Batch
from tianshou.data.types import (
    ActStateBatchProtocol,
    RolloutBatchProtocol,
)


@dataclass(kw_only=True)
class TD3TrainingStats(TrainingStats):
    actor_loss: float
    critic1_loss: float
    critic2_loss: float


class ActorDualCriticsOffPolicyAlgorithm(
    ActorCriticOffPolicyAlgorithm[TPolicy, TActBatchProtocol],
    ABC,
):
    """A base class for off-policy algorithms with two critics, where the target Q-value is computed as the minimum
    of the two lagged critics' values.
    """

    def __init__(
        self,
        *,
        policy: Any,
        policy_optim: OptimizerFactory,
        critic: torch.nn.Module,
        critic_optim: OptimizerFactory,
        critic2: torch.nn.Module | None = None,
        critic2_optim: OptimizerFactory | None = None,
        tau: float = 0.005,
        gamma: float = 0.99,
        n_step_return_horizon: int = 1,
    ) -> None:
        """
        :param policy: the policy
        :param policy_optim: the optimizer factory for the policy's model.
        :param critic: the first critic network.
            For continuous action spaces: (s, a -> Q(s, a)).
            **NOTE**: The default implementation of `_target_q_compute_value` assumes
            a continuous action space; override this method if using discrete actions.
        :param critic_optim: the optimizer factory for the first critic network.
        :param critic2: the second critic network (analogous functionality to the first).
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
        self.critic2 = critic2 or deepcopy(critic)
        self.critic2_old = self._add_lagged_network(self.critic2)
        self.critic2_optim = self._create_optimizer(self.critic2, critic2_optim or critic_optim)

    def _target_q_compute_value(
        self, obs_batch: Batch, act_batch: TActBatchProtocol
    ) -> torch.Tensor:
        # compute the Q-value as the minimum of the two lagged critics
        act = act_batch.act
        return torch.min(
            self.critic_old(obs_batch.obs, act),
            self.critic2_old(obs_batch.obs, act),
        )


class TD3(
    ActorDualCriticsOffPolicyAlgorithm[ContinuousDeterministicPolicy, ActStateBatchProtocol],
):
    """Implementation of TD3, arXiv:1802.09477."""

    def __init__(
        self,
        *,
        policy: ContinuousDeterministicPolicy,
        policy_optim: OptimizerFactory,
        critic: torch.nn.Module,
        critic_optim: OptimizerFactory,
        critic2: torch.nn.Module | None = None,
        critic2_optim: OptimizerFactory | None = None,
        tau: float = 0.005,
        gamma: float = 0.99,
        policy_noise: float = 0.2,
        update_actor_freq: int = 2,
        noise_clip: float = 0.5,
        n_step_return_horizon: int = 1,
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
        :param policy_noise: scaling factor for the Gaussian noise added to target policy actions.
            This parameter implements target policy smoothing, a regularization technique described in the TD3 paper.
            The noise is sampled from a normal distribution and multiplied by this value before being added to actions.
            Higher values increase exploration in the target policy, helping to address function approximation error.
            The added noise is optionally clipped to a range determined by the noise_clip parameter.
            Typically set between 0.1 and 0.5 relative to the action scale of the environment.
        :param update_actor_freq: the frequency of actor network updates relative to critic network updates
            (the actor network is only updated once for every `update_actor_freq` critic updates).
            This implements the "delayed" policy updates from the TD3 algorithm, where the actor is
            updated less frequently than the critics.
            Higher values (e.g., 2-5) help stabilize training by allowing the critic to become more
            accurate before updating the policy.
            The default value of 2 follows the original TD3 paper's recommendation of updating the
            policy at half the rate of the Q-functions.
        :param noise_clip: defines the maximum absolute value of the noise added to target policy actions, i.e. noise values
            are clipped to the range [-noise_clip, noise_clip] (after generating and scaling the noise
            via `policy_noise`).
            This parameter implements bounded target policy smoothing as described in the TD3 paper.
            It prevents extreme noise values from causing unrealistic target values during training.
            Setting it 0.0 (or a negative value) disables clipping entirely.
            It is typically set to about twice the `policy_noise` value (e.g. 0.5 when `policy_noise` is 0.2).
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
            n_step_return_horizon=n_step_return_horizon,
        )
        self.actor_old = self._add_lagged_network(self.policy.actor)
        self.policy_noise = policy_noise
        self.update_actor_freq = update_actor_freq
        self.noise_clip = noise_clip
        self._cnt = 0
        self._last = 0

    def _target_q_compute_action(self, obs_batch: Batch) -> ActStateBatchProtocol:
        # compute action using lagged actor
        act_batch = self.policy(obs_batch, model=self.actor_old)
        act_ = act_batch.act

        # add noise
        noise = torch.randn(size=act_.shape, device=act_.device) * self.policy_noise
        if self.noise_clip > 0.0:
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
        act_ += noise

        act_batch.act = act_
        return act_batch

    def _update_with_batch(self, batch: RolloutBatchProtocol) -> TD3TrainingStats:
        # critic 1&2
        td1, critic1_loss = self._minimize_critic_squared_loss(
            batch, self.critic, self.critic_optim
        )
        td2, critic2_loss = self._minimize_critic_squared_loss(
            batch, self.critic2, self.critic2_optim
        )
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        if self._cnt % self.update_actor_freq == 0:
            actor_loss = -self.critic(batch.obs, self.policy(batch, eps=0.0).act).mean()
            self._last = actor_loss.item()
            self.policy_optim.step(actor_loss)
            self._update_lagged_network_weights()
        self._cnt += 1

        return TD3TrainingStats(
            actor_loss=self._last,
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
        )
