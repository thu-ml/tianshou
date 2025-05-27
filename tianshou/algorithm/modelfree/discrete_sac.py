from dataclasses import dataclass
from typing import Any, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from tianshou.algorithm.algorithm_base import Policy
from tianshou.algorithm.modelfree.sac import Alpha, SACTrainingStats
from tianshou.algorithm.modelfree.td3 import ActorDualCriticsOffPolicyAlgorithm
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import Batch, to_torch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    DistBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.utils.net.discrete import DiscreteCritic


@dataclass
class DiscreteSACTrainingStats(SACTrainingStats):
    pass


TDiscreteSACTrainingStats = TypeVar("TDiscreteSACTrainingStats", bound=DiscreteSACTrainingStats)


class DiscreteSACPolicy(Policy):
    def __init__(
        self,
        *,
        actor: torch.nn.Module,
        deterministic_eval: bool = True,
        action_space: gym.Space,
        observation_space: gym.Space | None = None,
    ):
        """
        :param actor: the actor network following the rules (s -> dist_input_BD),
            where the distribution input is for a `Categorical` distribution.
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
        :param action_space: the environment's action_space.
        :param observation_space: the environment's observation space
        """
        assert isinstance(action_space, gym.spaces.Discrete)
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
        )
        self.actor = actor
        self.deterministic_eval = deterministic_eval

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Batch:
        logits_BA, hidden_BH = self.actor(batch.obs, state=state, info=batch.info)
        dist = Categorical(logits=logits_BA)
        act_B = (
            dist.mode
            if self.deterministic_eval and not self.is_within_training_step
            else dist.sample()
        )
        return Batch(logits=logits_BA, act=act_B, state=hidden_BH, dist=dist)


class DiscreteSAC(ActorDualCriticsOffPolicyAlgorithm[DiscreteSACPolicy, DistBatchProtocol]):
    """Implementation of SAC for Discrete Action Settings. arXiv:1910.07207."""

    def __init__(
        self,
        *,
        policy: DiscreteSACPolicy,
        policy_optim: OptimizerFactory,
        critic: torch.nn.Module | DiscreteCritic,
        critic_optim: OptimizerFactory,
        critic2: torch.nn.Module | DiscreteCritic | None = None,
        critic2_optim: OptimizerFactory | None = None,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float | Alpha = 0.2,
        n_step_return_horizon: int = 1,
    ) -> None:
        """
        :param policy: the policy
        :param policy_optim: the optimizer factory for the policy's model.
        :param critic: the first critic network. (s -> <Q(s, a_1), ..., Q(s, a_N)>).
        :param critic_optim: the optimizer factory for the first critic network.
        :param critic2: the second critic network. (s -> <Q(s, a_1), ..., Q(s, a_N)>).
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
        :param alpha: the entropy regularization coefficient alpha or an object
            which can be used to automatically tune it (e.g. an instance of `AutoAlpha`).
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
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=tau,
            gamma=gamma,
            n_step_return_horizon=n_step_return_horizon,
        )
        self.alpha = Alpha.from_float_or_instance(alpha)

    def _target_q_compute_value(
        self, obs_batch: Batch, act_batch: DistBatchProtocol
    ) -> torch.Tensor:
        dist = cast(Categorical, act_batch.dist)
        target_q = dist.probs * torch.min(
            self.critic_old(obs_batch.obs),
            self.critic2_old(obs_batch.obs),
        )
        return target_q.sum(dim=-1) + self.alpha.value * dist.entropy()

    def _update_with_batch(self, batch: RolloutBatchProtocol) -> TDiscreteSACTrainingStats:  # type: ignore
        weight = batch.pop("weight", 1.0)
        target_q = batch.returns.flatten()
        act = to_torch(batch.act[:, np.newaxis], device=target_q.device, dtype=torch.long)

        # critic 1
        current_q1 = self.critic(batch.obs).gather(1, act).flatten()
        td1 = current_q1 - target_q
        critic1_loss = (td1.pow(2) * weight).mean()
        self.critic_optim.step(critic1_loss)

        # critic 2
        current_q2 = self.critic2(batch.obs).gather(1, act).flatten()
        td2 = current_q2 - target_q
        critic2_loss = (td2.pow(2) * weight).mean()
        self.critic2_optim.step(critic2_loss)

        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        dist = self.policy(batch).dist
        entropy = dist.entropy()
        with torch.no_grad():
            current_q1a = self.critic(batch.obs)
            current_q2a = self.critic2(batch.obs)
            q = torch.min(current_q1a, current_q2a)
        actor_loss = -(self.alpha.value * entropy + (dist.probs * q).sum(dim=-1)).mean()
        self.policy_optim.step(actor_loss)

        alpha_loss = self.alpha.update(entropy.detach())

        self._update_lagged_network_weights()

        return DiscreteSACTrainingStats(  # type: ignore[return-value]
            actor_loss=actor_loss.item(),
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
            alpha=self.alpha.value,
            alpha_loss=alpha_loss,
        )
