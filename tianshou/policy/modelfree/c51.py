from dataclasses import dataclass
from typing import Generic, TypeVar

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy.modelfree.dqn import (
    DQNPolicy,
    DQNTrainingStats,
    QLearningOffPolicyAlgorithm,
)
from tianshou.policy.optim import OptimizerFactory
from tianshou.utils.net.common import Net


@dataclass(kw_only=True)
class C51TrainingStats(DQNTrainingStats):
    pass


TC51TrainingStats = TypeVar("TC51TrainingStats", bound=C51TrainingStats)


class C51Policy(DQNPolicy):
    def __init__(
        self,
        model: torch.nn.Module | Net,
        action_space: gym.spaces.Space,
        observation_space: gym.Space | None = None,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        """
        :param model: a model following the rules (s_B -> action_values_BA)
        :param num_atoms: the number of atoms in the support set of the
            value distribution. Default to 51.
        :param v_min: the value of the smallest atom in the support set.
            Default to -10.0.
        :param v_max: the value of the largest atom in the support set.
            Default to 10.0.
        """
        assert isinstance(action_space, gym.spaces.Discrete)
        super().__init__(
            model=model, action_space=action_space, observation_space=observation_space
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


class C51(QLearningOffPolicyAlgorithm[C51Policy, TC51TrainingStats], Generic[TC51TrainingStats]):
    """Implementation of Categorical Deep Q-Network. arXiv:1707.06887."""

    def __init__(
        self,
        *,
        policy: C51Policy,
        optim: OptimizerFactory,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
    ) -> None:
        """
        :param policy: a policy following the rules (s -> action_values_BA)
        :param optim: a torch.optim for optimizing the policy.
        :param discount_factor: in [0, 1].
        :param estimation_step: the number of steps to look ahead.
        :param target_update_freq: the target network update frequency (0 if
            you do not use the target network).
        :param reward_normalization: normalize the **returns** to Normal(0, 1).
            TODO: rename to return_normalization?
        """
        super().__init__(
            policy=policy,
            optim=optim,
            discount_factor=discount_factor,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
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
    ) -> TC51TrainingStats:
        self._periodically_update_lagged_network_weights()
        self.optim.zero_grad()
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
        loss.backward()
        self.optim.step()

        return C51TrainingStats(loss=loss.item())  # type: ignore[return-value]
