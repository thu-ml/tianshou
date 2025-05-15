import gymnasium as gym
import numpy as np
import torch

from tianshou.algorithm.modelfree.dqn import (
    DiscreteQLearningPolicy,
    QLearningOffPolicyAlgorithm,
)
from tianshou.algorithm.modelfree.pg import LossSequenceTrainingStats
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import Batch, ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol
from tianshou.utils.net.common import Net


class C51Policy(DiscreteQLearningPolicy):
    def __init__(
        self,
        model: torch.nn.Module | Net,
        action_space: gym.spaces.Space,
        observation_space: gym.Space | None = None,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        eps_training: float = 0.0,
        eps_inference: float = 0.0,
    ):
        """
        :param model: a model following the rules (s_B -> action_values_BA)
        :param num_atoms: the number of atoms in the support set of the
            value distribution. Default to 51.
        :param v_min: the value of the smallest atom in the support set.
            Default to -10.0.
        :param v_max: the value of the largest atom in the support set.
            Default to 10.0.
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
        super().__init__(
            model=model,
            action_space=action_space,
            observation_space=observation_space,
            eps_training=eps_training,
            eps_inference=eps_inference,
        )
        assert num_atoms > 1, f"num_atoms should be greater than 1 but got: {num_atoms}"
        assert v_min < v_max, f"v_max should be larger than v_min, but got {v_min=} and {v_max=}"
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.nn.Parameter(
            torch.linspace(self.v_min, self.v_max, self.num_atoms),
            requires_grad=False,
        )

    def compute_q_value(self, logits: torch.Tensor, mask: np.ndarray | None) -> torch.Tensor:
        return super().compute_q_value((logits * self.support).sum(2), mask)


class C51(QLearningOffPolicyAlgorithm[C51Policy]):
    """Implementation of Categorical Deep Q-Network. arXiv:1707.06887."""

    def __init__(
        self,
        *,
        policy: C51Policy,
        optim: OptimizerFactory,
        gamma: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
    ) -> None:
        """
        :param policy: a policy following the rules (s -> action_values_BA)
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
        """
        super().__init__(
            policy=policy,
            optim=optim,
            gamma=gamma,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
        )
        self.delta_z = (policy.v_max - policy.v_min) / (policy.num_atoms - 1)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        return self.policy.support.repeat(len(indices), 1)  # shape: [bsz, num_atoms]

    def _target_dist(self, batch: RolloutBatchProtocol) -> torch.Tensor:
        obs_next_batch = Batch(obs=batch.obs_next, info=[None] * len(batch))
        if self.use_target_network:
            act = self.policy(obs_next_batch).act
            next_dist = self.policy(obs_next_batch, model=self.model_old).logits
        else:
            next_batch = self.policy(obs_next_batch)
            act = next_batch.act
            next_dist = next_batch.logits
        next_dist = next_dist[np.arange(len(act)), act, :]
        target_support = batch.returns.clamp(self.policy.v_min, self.policy.v_max)
        # An amazing trick for calculating the projection gracefully.
        # ref: https://github.com/ShangtongZhang/DeepRL
        target_dist = (
            1
            - (target_support.unsqueeze(1) - self.policy.support.view(1, -1, 1)).abs()
            / self.delta_z
        ).clamp(0, 1) * next_dist.unsqueeze(1)
        return target_dist.sum(-1)

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> LossSequenceTrainingStats:
        self._periodically_update_lagged_network_weights()
        with torch.no_grad():
            target_dist = self._target_dist(batch)
        weight = batch.pop("weight", 1.0)
        curr_dist = self.policy(batch).logits
        act = batch.act
        curr_dist = curr_dist[np.arange(len(act)), act, :]
        cross_entropy = -(target_dist * torch.log(curr_dist + 1e-8)).sum(1)
        loss = (cross_entropy * weight).mean()
        # ref: https://github.com/Kaixhin/Rainbow/blob/master/agent.py L94-100
        batch.weight = cross_entropy.detach()  # prio-buffer
        self.optim.step(loss)

        return LossSequenceTrainingStats(loss=loss.item())
