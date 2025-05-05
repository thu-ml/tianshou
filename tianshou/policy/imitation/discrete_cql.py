from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import to_torch
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import QRDQN
from tianshou.policy.base import OfflineAlgorithm
from tianshou.policy.modelfree.pg import SimpleLossTrainingStats
from tianshou.policy.modelfree.qrdqn import QRDQNPolicy
from tianshou.policy.optim import OptimizerFactory


@dataclass(kw_only=True)
class DiscreteCQLTrainingStats(SimpleLossTrainingStats):
    cql_loss: float
    qr_loss: float


# NOTE: This uses diamond inheritance to convert from off-policy to offline
class DiscreteCQL(OfflineAlgorithm[QRDQNPolicy], QRDQN[QRDQNPolicy]):  # type: ignore[misc]
    """Implementation of discrete Conservative Q-Learning algorithm. arXiv:2006.04779."""

    def __init__(
        self,
        *,
        policy: QRDQNPolicy,
        optim: OptimizerFactory,
        min_q_weight: float = 10.0,
        gamma: float = 0.99,
        num_quantiles: int = 200,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
    ) -> None:
        """
        :param policy: the policy
        :param optim: the optimizer factory for the policy's model.
        :param min_q_weight: the weight for the cql loss.
        :param gamma: the discount factor in [0, 1] for future rewards.
            This determines how much future rewards are valued compared to immediate ones.
            Lower values (closer to 0) make the agent focus on immediate rewards, creating "myopic"
            behavior. Higher values (closer to 1) make the agent value long-term rewards more,
            potentially improving performance in tasks where delayed rewards are important but
            increasing training variance by incorporating more environmental stochasticity.
            Typically set between 0.9 and 0.99 for most reinforcement learning tasks
        :param num_quantiles: the number of quantile midpoints in the inverse
            cumulative distribution function of the value.
        :param estimation_step: the number of future steps (> 0) to consider when computing temporal
            difference (TD) targets. Controls the balance between TD learning and Monte Carlo methods:
            higher values reduce bias (by relying less on potentially inaccurate value estimates)
            but increase variance (by incorporating more environmental stochasticity and reducing
            the averaging effect). A value of 1 corresponds to standard TD learning with immediate
            bootstrapping, while very large values approach Monte Carlo-like estimation that uses
            complete episode returns.
        :param target_update_freq: the target network update frequency (0 if
            you do not use the target network).
        :param reward_normalization: normalize the **returns** to Normal(0, 1).
            TODO: rename to return_normalization?
        """
        QRDQN.__init__(
            self,
            policy=policy,
            optim=optim,
            gamma=gamma,
            num_quantiles=num_quantiles,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
        )
        self.min_q_weight = min_q_weight

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> DiscreteCQLTrainingStats:
        self._periodically_update_lagged_network_weights()
        weight = batch.pop("weight", 1.0)
        all_dist = self.policy(batch).logits
        act = to_torch(batch.act, dtype=torch.long, device=all_dist.device)
        curr_dist = all_dist[np.arange(len(act)), act, :].unsqueeze(2)
        target_dist = batch.returns.unsqueeze(1)
        # calculate each element's difference between curr_dist and target_dist
        dist_diff = F.smooth_l1_loss(target_dist, curr_dist, reduction="none")
        huber_loss = (
            (dist_diff * (self.tau_hat - (target_dist - curr_dist).detach().le(0.0).float()).abs())
            .sum(-1)
            .mean(1)
        )
        qr_loss = (huber_loss * weight).mean()
        # ref: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/
        # blob/master/fqf_iqn_qrdqn/agent/qrdqn_agent.py L130
        batch.weight = dist_diff.detach().abs().sum(-1).mean(1)  # prio-buffer
        # add CQL loss
        q = self.policy.compute_q_value(all_dist, None)
        dataset_expec = q.gather(1, act.unsqueeze(1)).mean()
        negative_sampling = q.logsumexp(1).mean()
        min_q_loss = negative_sampling - dataset_expec
        loss = qr_loss + min_q_loss * self.min_q_weight
        self.optim.step(loss)

        return DiscreteCQLTrainingStats(
            loss=loss.item(),
            qr_loss=qr_loss.item(),
            cql_loss=min_q_loss.item(),
        )
