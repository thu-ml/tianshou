from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from sensai.util.helper import mark_used

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch, to_torch_as
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    ActBatchProtocol,
    BatchWithReturnsProtocol,
    ModelOutputBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.policy.base import TArrOrActBatch
from tianshou.policy.modelfree.dqn import (
    DiscreteQLearningPolicy,
    QLearningOffPolicyAlgorithm,
)
from tianshou.policy.modelfree.pg import SimpleLossTrainingStats
from tianshou.policy.optim import OptimizerFactory
from tianshou.utils.net.common import BranchingNet

mark_used(ActBatchProtocol)


class BDQNPolicy(DiscreteQLearningPolicy[BranchingNet]):
    def __init__(
        self,
        *,
        model: BranchingNet,
        action_space: gym.spaces.Discrete,
        observation_space: gym.Space | None = None,
        eps_training: float = 0.0,
        eps_inference: float = 0.0,
    ):
        """
        :param model: BranchingNet mapping (obs, state, info) -> action_values_BA.
        :param action_space: the environment's action space
        :param observation_space: the environment's observation space.
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
        super().__init__(
            model=model,
            action_space=action_space,
            observation_space=observation_space,
            eps_training=eps_training,
            eps_inference=eps_inference,
        )

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> ModelOutputBatchProtocol:
        if model is None:
            model = self.model
        obs = batch.obs
        # TODO: this is very contrived, see also iqn.py
        obs_next_BO = obs.obs if hasattr(obs, "obs") else obs
        action_values_BA, hidden_BH = model(obs_next_BO, rnn_hidden_state=state, info=batch.info)
        act_B = to_numpy(action_values_BA.argmax(dim=-1))
        result = Batch(logits=action_values_BA, act=act_B, rnn_hidden_state=hidden_BH)
        return cast(ModelOutputBatchProtocol, result)

    def add_exploration_noise(
        self,
        act: TArrOrActBatch,
        batch: ObsBatchProtocol,
    ) -> TArrOrActBatch:
        eps = self.eps_training if self.is_within_training_step else self.eps_inference
        # TODO: This looks problematic; the non-array case is silently ignored
        if isinstance(act, np.ndarray) and not np.isclose(eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < eps
            rand_act = np.random.randint(
                low=0,
                high=self.model.action_per_branch,
                size=(bsz, act.shape[-1]),
            )
            if hasattr(batch.obs, "mask"):
                rand_act += batch.obs.mask
            act[rand_mask] = rand_act[rand_mask]
        return act


class BDQN(QLearningOffPolicyAlgorithm[BDQNPolicy]):
    """Implementation of the Branching Dueling Q-Network (BDQN) algorithm arXiv:1711.08946."""

    def __init__(
        self,
        *,
        policy: BDQNPolicy,
        optim: OptimizerFactory,
        gamma: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        is_double: bool = True,
    ) -> None:
        """
        :param policy: policy
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
        :param is_double: flag indicating whether to use Double Q-learning for target value calculation.
            If True, the algorithm uses the online network to select actions and the target network to evaluate their Q-values.
            This decoupling helps reduce the overestimation bias that standard Q-learning is prone to.
            If False, the algorithm selects actions by directly taking the maximum Q-value from the target network.
            Note: This parameter is most effective when used with a target network (target_update_freq > 0).
        """
        assert (
            estimation_step == 1
        ), f"N-step bigger than one is not supported by BDQ but got: {estimation_step}"
        super().__init__(
            policy=policy,
            optim=optim,
            gamma=gamma,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
        )
        self.is_double = is_double

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # obs_next: s_{t+n}
        result = self.policy(obs_next_batch)
        if self.use_target_network:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self.policy(obs_next_batch, model=self.model_old).logits
        else:
            target_q = result.logits
        if self.is_double:
            act = np.expand_dims(self.policy(obs_next_batch).act, -1)
            act = to_torch(act, dtype=torch.long, device=target_q.device)
        else:
            act = target_q.max(-1).indices.unsqueeze(-1)
        return torch.gather(target_q, -1, act).squeeze()

    def _compute_return(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        gamma: float = 0.99,
    ) -> BatchWithReturnsProtocol:
        rew = batch.rew
        with torch.no_grad():
            target_q_torch = self._target_q(buffer, indice)  # (bsz, ?)
        target_q = to_numpy(target_q_torch)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        end_flag = end_flag[indice]
        mean_target_q = np.mean(target_q, -1) if len(target_q.shape) > 1 else target_q
        _target_q = rew + gamma * mean_target_q * (1 - end_flag)
        target_q = np.repeat(_target_q[..., None], self.policy.model.num_branches, axis=-1)
        target_q = np.repeat(target_q[..., None], self.policy.model.action_per_branch, axis=-1)

        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return cast(BatchWithReturnsProtocol, batch)

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithReturnsProtocol:
        """Compute the 1-step return for BDQ targets."""
        return self._compute_return(batch, buffer, indices)

    def _update_with_batch(  # type: ignore[override]
        self,
        batch: BatchWithReturnsProtocol,
    ) -> SimpleLossTrainingStats:
        self._periodically_update_lagged_network_weights()
        weight = batch.pop("weight", 1.0)
        act = to_torch(batch.act, dtype=torch.long, device=batch.returns.device)
        q = self.policy(batch).logits
        act_mask = torch.zeros_like(q)
        act_mask = act_mask.scatter_(-1, act.unsqueeze(-1), 1)
        act_q = q * act_mask
        returns = batch.returns
        returns = returns * act_mask
        td_error = returns - act_q
        loss = (td_error.pow(2).sum(-1).mean(-1) * weight).mean()
        batch.weight = td_error.sum(-1).sum(-1)  # prio-buffer
        self.optim.step(loss)

        return SimpleLossTrainingStats(loss=loss.item())
