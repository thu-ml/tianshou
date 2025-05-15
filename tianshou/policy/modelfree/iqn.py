from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, to_numpy
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    ObsBatchProtocol,
    QuantileRegressionBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.policy import QRDQN
from tianshou.policy.modelfree.pg import SimpleLossTrainingStats
from tianshou.policy.modelfree.qrdqn import QRDQNPolicy
from tianshou.policy.optim import OptimizerFactory


class IQNPolicy(QRDQNPolicy):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        action_space: gym.spaces.Space,
        sample_size: int = 32,
        online_sample_size: int = 8,
        target_sample_size: int = 8,
        observation_space: gym.Space | None = None,
        eps_training: float = 0.0,
        eps_inference: float = 0.0,
    ) -> None:
        """
        :param model:
        :param action_space: the environment's action space
        :param sample_size:
        :param online_sample_size:
        :param target_sample_size:
        :param observation_space: the environment's observation space
        :param eps_training: the epsilon value for epsilon-greedy exploration during training.
            When collecting data for training, this is the probability of choosing a random action
            instead of the action chosen by the policy.
            A value of 0.0 means no exploration (fully greedy) and a value of 1.0 means full
            exploration (fully random).
        :param eps_inference: the epsilon value for epsilon-greedy exploration during inference,
            i.e. non-training cases (such as evaluation during test steps).
            The epsilon value is the probability of choosing a random action instead of the action
            chosen by the policy.
            A value of 0.0 means no exploration (fully greedy) and a value of 1.0 means full
            exploration (fully random).
        """
        assert isinstance(action_space, gym.spaces.Discrete)
        assert sample_size > 1, f"sample_size should be greater than 1 but got: {sample_size}"
        assert (
            online_sample_size > 1
        ), f"online_sample_size should be greater than 1 but got: {online_sample_size}"
        assert (
            target_sample_size > 1
        ), f"target_sample_size should be greater than 1 but got: {target_sample_size}"
        super().__init__(
            model=model,
            action_space=action_space,
            observation_space=observation_space,
            eps_training=eps_training,
            eps_inference=eps_inference,
        )
        self.sample_size = sample_size
        self.online_sample_size = online_sample_size
        self.target_sample_size = target_sample_size

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> QuantileRegressionBatchProtocol:
        is_model_old = model is not None
        if is_model_old:
            sample_size = self.target_sample_size
        elif self.training:
            sample_size = self.online_sample_size
        else:
            sample_size = self.sample_size
        if model is None:
            model = self.model
        obs = batch.obs
        # TODO: this seems very contrived!
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        (logits, taus), hidden = model(
            obs_next,
            sample_size=sample_size,
            state=state,
            info=batch.info,
        )
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if self.max_action_num is None:  # type: ignore
            # TODO: see same thing in DQNPolicy!
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        result = Batch(logits=logits, act=act, rnn_hidden_state=hidden, taus=taus)
        return cast(QuantileRegressionBatchProtocol, result)


class IQN(QRDQN[IQNPolicy]):
    """Implementation of Implicit Quantile Network. arXiv:1806.06923."""

    def __init__(
        self,
        *,
        policy: IQNPolicy,
        optim: OptimizerFactory,
        gamma: float = 0.99,
        num_quantiles: int = 200,
        estimation_step: int = 1,
        target_update_freq: int = 0,
    ) -> None:
        """
        :param policy: the policy
        :param optim: the optimizer factory for the policy's model
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
            optim=optim,
            gamma=gamma,
            num_quantiles=num_quantiles,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
        )

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> SimpleLossTrainingStats:
        self._periodically_update_lagged_network_weights()
        weight = batch.pop("weight", 1.0)
        action_batch = self.policy(batch)
        curr_dist, taus = action_batch.logits, action_batch.taus
        act = batch.act
        curr_dist = curr_dist[np.arange(len(act)), act, :].unsqueeze(2)
        target_dist = batch.returns.unsqueeze(1)
        # calculate each element's difference between curr_dist and target_dist
        dist_diff = F.smooth_l1_loss(target_dist, curr_dist, reduction="none")
        huber_loss = (
            (
                dist_diff
                * (taus.unsqueeze(2) - (target_dist - curr_dist).detach().le(0.0).float()).abs()
            )
            .sum(-1)
            .mean(1)
        )
        loss = (huber_loss * weight).mean()
        # ref: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/
        # blob/master/fqf_iqn_qrdqn/agent/qrdqn_agent.py L130
        batch.weight = dist_diff.detach().abs().sum(-1).mean(1)  # prio-buffer
        self.optim.step(loss)

        return SimpleLossTrainingStats(loss=loss.item())
