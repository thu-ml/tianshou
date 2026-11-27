import math
from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.algorithm.algorithm_base import (
    LaggedNetworkFullUpdateAlgorithmMixin,
    OfflineAlgorithm,
)
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.modelfree.reinforce import SimpleLossTrainingStats
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


def _validate_target_update_freq(target_update_freq: int | None) -> None:
    if target_update_freq is not None and target_update_freq < 0:
        raise ValueError("target_update_freq must be nonnegative when specified")


@dataclass(kw_only=True)
class DiscreteBCQTrainingStats(SimpleLossTrainingStats):
    q_loss: float
    i_loss: float
    reg_loss: float


class DiscreteBCQPolicy(DiscreteQLearningPolicy):
    # Older pickled policies did not store the interval.
    _default_target_update_freq: int | None = None

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        imitator: torch.nn.Module,
        target_update_freq: int | None = None,
        unlikely_action_threshold: float = 0.3,
        action_space: gym.spaces.Discrete,
        observation_space: gym.Space | None = None,
        eps_inference: float = 0.0,
    ) -> None:
        """
        :param model: a model following the rules (s_B -> action_values_BA)
        :param imitator: a model following the rules (s -> imitation_logits)
        :param target_update_freq: optional target network update interval for :class:`DiscreteBCQ`.
            ``None`` leaves the interval unspecified. Zero disables the target network.
            An explicit interval on the algorithm takes precedence.
        :param unlikely_action_threshold: the threshold (tau) for unlikely
            actions, as shown in Equ. (17) in the paper.
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
        _validate_target_update_freq(target_update_freq)
        self._default_target_update_freq = target_update_freq
        assert 0.0 <= unlikely_action_threshold < 1.0, (
            f"unlikely_action_threshold should be in [0, 1) but got: {unlikely_action_threshold}"
        )
        if unlikely_action_threshold > 0:
            self._log_tau = math.log(unlikely_action_threshold)
        else:
            self._log_tau = -np.inf

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: Any | None = None,
        model: nn.Module | None = None,
    ) -> ImitationBatchProtocol:
        if model is None:
            model = self.model
        q_value, state = model(batch.obs, state=state, info=batch.info)
        imitation_logits, _ = self.imitator(batch.obs, state=state, info=batch.info)

        # mask actions for argmax
        ratio = imitation_logits - imitation_logits.max(dim=-1, keepdim=True).values
        mask = (ratio < self._log_tau).float()
        act = (q_value - INF * mask).argmax(dim=-1)

        result = Batch(
            act=act,
            state=state,
            q_value=q_value,
            imitation_logits=imitation_logits,
            logits=imitation_logits,
        )
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
        n_step_return_horizon: int = 1,
        target_update_freq: int | None = None,
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
        :param n_step_return_horizon: the number of future steps (> 0) to consider when computing temporal
            difference (TD) targets. Controls the balance between TD learning and Monte Carlo methods:
            higher values reduce bias (by relying less on potentially inaccurate value estimates)
            but increase variance (by incorporating more environmental stochasticity and reducing
            the averaging effect). A value of 1 corresponds to standard TD learning with immediate
            bootstrapping, while very large values approach Monte Carlo-like estimation that uses
            complete episode returns.
        :param target_update_freq: number of gradient steps between target network updates.
            ``None`` uses the policy's interval. If both are unspecified, the default is
            8000, the Atari setting in the paper (Appendix A.2, Table 1). Zero disables
            the target network and uses the online Q-value of the action selected by the policy.
        :param imitation_logits_penalty: regularization weight for imitation
            logits.
        """
        _validate_target_update_freq(target_update_freq)
        super().__init__(
            policy=policy,
        )
        LaggedNetworkFullUpdateAlgorithmMixin.__init__(self)
        self.optim = self._create_optimizer(self.policy, optim)
        assert 0.0 <= gamma <= 1.0, f"discount factor should be in [0, 1] but got: {gamma}"
        self.gamma = gamma
        assert n_step_return_horizon > 0, (
            f"n_step_return_horizon should be greater than 0 but got: {n_step_return_horizon}"
        )
        self.n_step = n_step_return_horizon
        if target_update_freq is None:
            target_update_freq = policy._default_target_update_freq
        self.freq = 8000 if target_update_freq is None else target_update_freq
        self._target = self.freq > 0
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
        result = self.policy(next_obs_batch)
        if self._target:
            target_q, _ = self.model_old(batch.obs_next)
        else:
            target_q = result.q_value
        return target_q[np.arange(len(indices)), result.act]

    def _update_with_batch(  # type: ignore[override]
        self,
        batch: BatchWithReturnsProtocol,
    ) -> DiscreteBCQTrainingStats:
        if self._target and self._iter % self.freq == 0:
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
