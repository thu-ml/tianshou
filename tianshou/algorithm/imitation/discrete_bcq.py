import math
from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from tianshou.algorithm.algorithm_base import (
    LaggedNetworkFullUpdateAlgorithmMixin,
    OfflineAlgorithm,
)
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.modelfree.pg import SimpleLossTrainingStats
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.data.types import (
    BatchWithReturnsProtocol,
    ImitationBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)

float_info = torch.finfo(torch.float32)
INF = float_info.max


@dataclass(kw_only=True)
class DiscreteBCQTrainingStats(SimpleLossTrainingStats):
    q_loss: float
    i_loss: float
    reg_loss: float


class DiscreteBCQPolicy(DiscreteQLearningPolicy):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        imitator: torch.nn.Module,
        target_update_freq: int = 8000,
        unlikely_action_threshold: float = 0.3,
        action_space: gym.spaces.Discrete,
        observation_space: gym.Space | None = None,
        eps_inference: float = 0.0,
    ) -> None:
        """
        :param model: a model following the rules (s_B -> action_values_BA)
        :param imitator: a model following the rules (s -> imitation_logits)
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
        :param unlikely_action_threshold: the threshold (tau) for unlikely
            actions, as shown in Equ. (17) in the paper.
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
        :param action_space: the environment's action space.
        :param observation_space: the environment's observation space.
        :param eps_inference: the epsilon value for epsilon-greedy exploration during inference,
            i.e. non-training cases (such as evaluation during test steps).
            The epsilon value is the probability of choosing a random action instead of the action
            chosen by the policy.
            A value of 0.0 means no exploration (fully greedy) and a value of 1.0 means full
            exploration (fully random).
        """
        super().__init__(
            model=model,
            action_space=action_space,
            observation_space=observation_space,
            eps_training=0.0,  # no training data collection (offline)
            eps_inference=eps_inference,
        )
        self.imitator = imitator
        assert (
            target_update_freq > 0
        ), f"BCQ needs target_update_freq>0 but got: {target_update_freq}."
        assert (
            0.0 <= unlikely_action_threshold < 1.0
        ), f"unlikely_action_threshold should be in [0, 1) but got: {unlikely_action_threshold}"
        if unlikely_action_threshold > 0:
            self._log_tau = math.log(unlikely_action_threshold)
        else:
            self._log_tau = -np.inf
        self.max_action_num: int | None = None

    def forward(  # type: ignore
        self,
        batch: ObsBatchProtocol,
        state: dict | Batch | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ImitationBatchProtocol:
        q_value, state = self.model(batch.obs, state=state, info=batch.info)
        if self.max_action_num is None:
            self.max_action_num = q_value.shape[1]
        imitation_logits, _ = self.imitator(batch.obs, state=state, info=batch.info)

        # mask actions for argmax
        ratio = imitation_logits - imitation_logits.max(dim=-1, keepdim=True).values
        mask = (ratio < self._log_tau).float()
        act = (q_value - INF * mask).argmax(dim=-1)

        result = Batch(act=act, state=state, q_value=q_value, imitation_logits=imitation_logits)
        return cast(ImitationBatchProtocol, result)


class DiscreteBCQ(
    OfflineAlgorithm[DiscreteBCQPolicy],
    LaggedNetworkFullUpdateAlgorithmMixin,
):
    """Implementation of the discrete batch-constrained deep Q-learning (BCQ) algorithm. arXiv:1910.01708."""

    def __init__(
        self,
        *,
        policy: DiscreteBCQPolicy,
        optim: OptimizerFactory,
        gamma: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 8000,
        imitation_logits_penalty: float = 1e-2,
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
        :param imitation_logits_penalty: regularization weight for imitation
            logits.
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
        super().__init__(
            policy=policy,
        )
        LaggedNetworkFullUpdateAlgorithmMixin.__init__(self)
        self.optim = self._create_optimizer(self.policy, optim)
        assert 0.0 <= gamma <= 1.0, f"discount factor should be in [0, 1] but got: {gamma}"
        self.gamma = gamma
        assert (
            estimation_step > 0
        ), f"estimation_step should be greater than 0 but got: {estimation_step}"
        self.n_step = estimation_step
        self._target = target_update_freq > 0
        self.freq = target_update_freq
        self._iter = 0
        if self._target:
            self.model_old = self._add_lagged_network(self.policy.model)
        self._weight_reg = imitation_logits_penalty

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithReturnsProtocol:
        return self.compute_nstep_return(
            batch=batch,
            buffer=buffer,
            indices=indices,
            target_q_fn=self._target_q,
            gamma=self.gamma,
            n_step=self.n_step,
        )

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        next_obs_batch = Batch(obs=batch.obs_next, info=[None] * len(batch))
        # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
        act = self.policy(next_obs_batch).act
        target_q, _ = self.model_old(batch.obs_next)
        return target_q[np.arange(len(act)), act]

    def _update_with_batch(  # type: ignore[override]
        self,
        batch: BatchWithReturnsProtocol,
    ) -> DiscreteBCQTrainingStats:
        if self._iter % self.freq == 0:
            self._update_lagged_network_weights()
        self._iter += 1

        target_q = batch.returns.flatten()
        result = self.policy(batch)
        imitation_logits = result.imitation_logits
        current_q = result.q_value[np.arange(len(target_q)), batch.act]
        act = to_torch(batch.act, dtype=torch.long, device=target_q.device)
        q_loss = F.smooth_l1_loss(current_q, target_q)
        i_loss = F.nll_loss(F.log_softmax(imitation_logits, dim=-1), act)
        reg_loss = imitation_logits.pow(2).mean()
        loss = q_loss + i_loss + self._weight_reg * reg_loss

        self.optim.step(loss)

        return DiscreteBCQTrainingStats(
            loss=loss.item(),
            q_loss=q_loss.item(),
            i_loss=i_loss.item(),
            reg_loss=reg_loss.item(),
        )
