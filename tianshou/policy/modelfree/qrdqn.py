import warnings
from typing import Generic, TypeVar

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy.modelfree.dqn import (
    DQNPolicy,
    QLearningOffPolicyAlgorithm,
)
from tianshou.policy.modelfree.pg import SimpleLossTrainingStats
from tianshou.policy.optim import OptimizerFactory


class QRDQNPolicy(DQNPolicy):
    def compute_q_value(self, logits: torch.Tensor, mask: np.ndarray | None) -> torch.Tensor:
        return super().compute_q_value(logits.mean(2), mask)


TQRDQNPolicy = TypeVar("TQRDQNPolicy", bound=QRDQNPolicy)


class QRDQN(
    QLearningOffPolicyAlgorithm[TQRDQNPolicy],
    Generic[TQRDQNPolicy],
):
    """Implementation of Quantile Regression Deep Q-Network. arXiv:1710.10044."""

    def __init__(
        self,
        *,
        policy: TQRDQNPolicy,
        optim: OptimizerFactory,
        gamma: float = 0.99,
        num_quantiles: int = 200,
        estimation_step: int = 1,
        target_update_freq: int = 0,
    ) -> None:
        """
        :param policy: the policy
        :param optim: the optimizer factory for the policy's model.
        :param gamma: the discount factor in [0, 1] for future rewards.
            This determines how much future rewards are valued compared to immediate ones.
            Lower values (closer to 0) make the agent focus on immediate rewards, creating "myopic"
            behavior. Higher values (closer to 1) make the agent value long-term rewards more,
            potentially improving performance in tasks where delayed rewards are important but
            increasing training variance by incorporating more environmental stochasticity.
            Typically set between 0.9 and 0.99 for most reinforcement learning tasks
        :param num_quantiles: the number of quantiles used to represent the return distribution for each action.
            Determines the granularity of the approximated distribution function.
            Higher values provide a more fine-grained approximation of the true return distribution but
            increase computational and memory requirements.
            Lower values reduce computational cost but may not capture the distribution accurately enough.
            The original QRDQN paper used 200 quantiles for Atari environments.
            Must be greater than 1, as at least two quantiles are needed to represent a distribution.
        :param estimation_step: the number of future steps (> 0) to consider when computing temporal
            difference (TD) targets. Controls the balance between TD learning and Monte Carlo methods:
            higher values reduce bias (by relying less on potentially inaccurate value estimates)
            but increase variance (by incorporating more environmental stochasticity and reducing
            the averaging effect). A value of 1 corresponds to standard TD learning with immediate
            bootstrapping, while very large values approach Monte Carlo-like estimation that uses
            complete episode returns.
        :param target_update_freq: the number of training iterations between each complete update of
            the target network.
            Controls how frequently the target Q-network parameters are updated with the current
            Q-network values.
            A value of 0 disables the target network entirely, using only a single network for both
            action selection and bootstrap targets.
            Higher values provide more stable learning targets but slow down the propagation of new
            value estimates. Lower positive values allow faster learning but may lead to instability
            due to rapidly changing targets.
            Typically set between 100-10000 for DQN variants, with exact values depending on environment
            complexity.
        """
        assert num_quantiles > 1, f"num_quantiles should be greater than 1 but got: {num_quantiles}"
        super().__init__(
            policy=policy,
            optim=optim,
            gamma=gamma,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
        )
        self.num_quantiles = num_quantiles
        tau = torch.linspace(0, 1, self.num_quantiles + 1)
        self.tau_hat = torch.nn.Parameter(
            ((tau[:-1] + tau[1:]) / 2).view(1, -1, 1),
            requires_grad=False,
        )
        warnings.filterwarnings("ignore", message="Using a target size")

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # obs_next: s_{t+n}
        if self.use_target_network:
            act = self.policy(obs_next_batch).act
            next_dist = self.policy(obs_next_batch, model=self.model_old).logits
        else:
            next_batch = self.policy(obs_next_batch)
            act = next_batch.act
            next_dist = next_batch.logits
        return next_dist[np.arange(len(act)), act, :]

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> SimpleLossTrainingStats:
        self._periodically_update_lagged_network_weights()
        weight = batch.pop("weight", 1.0)
        curr_dist = self.policy(batch).logits
        act = batch.act
        curr_dist = curr_dist[np.arange(len(act)), act, :].unsqueeze(2)
        target_dist = batch.returns.unsqueeze(1)
        # calculate each element's difference between curr_dist and target_dist
        dist_diff = F.smooth_l1_loss(target_dist, curr_dist, reduction="none")
        huber_loss = (
            (dist_diff * (self.tau_hat - (target_dist - curr_dist).detach().le(0.0).float()).abs())
            .sum(-1)
            .mean(1)
        )
        loss = (huber_loss * weight).mean()
        # ref: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/
        # blob/master/fqf_iqn_qrdqn/agent/qrdqn_agent.py L130
        batch.weight = dist_diff.detach().abs().sum(-1).mean(1)  # prio-buffer
        self.optim.step(loss)

        return SimpleLossTrainingStats(loss=loss.item())
