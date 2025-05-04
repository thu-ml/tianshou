from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, to_torch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    ModelOutputBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.policy import Algorithm
from tianshou.policy.base import (
    OfflineAlgorithm,
    OffPolicyAlgorithm,
    Policy,
    TrainingStats,
)
from tianshou.policy.optim import OptimizerFactory

# Dimension Naming Convention
# B - Batch Size
# A - Action
# D - Dist input (usually 2, loc and scale)
# H - Dimension of hidden, can be None


@dataclass(kw_only=True)
class ImitationTrainingStats(TrainingStats):
    loss: float = 0.0


TImitationTrainingStats = TypeVar("TImitationTrainingStats", bound=ImitationTrainingStats)


class ImitationPolicy(Policy):
    def __init__(
        self,
        *,
        actor: torch.nn.Module,
        action_space: gym.Space,
        observation_space: gym.Space | None = None,
        action_scaling: bool = False,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
    ):
        """
        :param actor: a model following the rules (s -> a)
        :param action_space: the environment's action_space.
        :param observation_space: the environment's observation space
        :param action_scaling: if True, scale the action from [-1, 1] to the range
            of action_space. Only used if the action_space is continuous.
        :param action_bound_method: method to bound action to range [-1, 1].
            Only used if the action_space is continuous.
        """
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
        )
        self.actor = actor

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ModelOutputBatchProtocol:
        # TODO - ALGO-REFACTORING: marked for refactoring when Algorithm abstraction is introduced
        if self.action_type == "discrete":
            # If it's discrete, the "actor" is usually a critic that maps obs to action_values
            # which then could be turned into logits or a Categorigal
            action_values_BA, hidden_BH = self.actor(batch.obs, state=state, info=batch.info)
            act_B = action_values_BA.argmax(dim=1)
            result = Batch(logits=action_values_BA, act=act_B, state=hidden_BH)
        elif self.action_type == "continuous":
            # If it's continuous, the actor would usually deliver something like loc, scale determining a
            # Gaussian dist
            dist_input_BD, hidden_BH = self.actor(batch.obs, state=state, info=batch.info)
            result = Batch(logits=dist_input_BD, act=dist_input_BD, state=hidden_BH)
        else:
            raise RuntimeError(f"Unknown {self.action_type=}, this shouldn't have happened!")
        return cast(ModelOutputBatchProtocol, result)


class ImitationLearningAlgorithmMixin:
    def _imitation_update(
        self,
        batch: RolloutBatchProtocol,
        policy: ImitationPolicy,
        optim: Algorithm.Optimizer,
    ) -> ImitationTrainingStats:
        if policy.action_type == "continuous":  # regression
            act = policy(batch).act
            act_target = to_torch(batch.act, dtype=torch.float32, device=act.device)
            loss = F.mse_loss(act, act_target)
        elif policy.action_type == "discrete":  # classification
            act = F.log_softmax(policy(batch).logits, dim=-1)
            act_target = to_torch(batch.act, dtype=torch.long, device=act.device)
            loss = F.nll_loss(act, act_target)
        else:
            raise ValueError(policy.action_type)
        optim.step(loss)

        return ImitationTrainingStats(loss=loss.item())


class OffPolicyImitationLearning(
    OffPolicyAlgorithm[ImitationPolicy, TImitationTrainingStats],
    ImitationLearningAlgorithmMixin,
    Generic[TImitationTrainingStats],
):
    """Implementation of off-policy vanilla imitation learning."""

    def __init__(
        self,
        *,
        policy: ImitationPolicy,
        optim: OptimizerFactory,
    ) -> None:
        """
        :param policy: the policy
        :param optim: the optimizer factory
        """
        super().__init__(
            policy=policy,
        )
        self.optim = self._create_optimizer(self.policy, optim)

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> TImitationTrainingStats:
        return self._imitation_update(batch, self.policy, self.optim)


class OfflineImitationLearning(
    OfflineAlgorithm[ImitationPolicy, TImitationTrainingStats],
    ImitationLearningAlgorithmMixin,
    Generic[TImitationTrainingStats],
):
    """Implementation of offline vanilla imitation learning."""

    def __init__(
        self,
        *,
        policy: ImitationPolicy,
        optim: OptimizerFactory,
    ) -> None:
        """
        :param policy: the policy
        :param optim: the optimizer factory
        """
        super().__init__(
            policy=policy,
        )
        self.optim = self._create_optimizer(self.policy, optim)

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> TImitationTrainingStats:
        return self._imitation_update(batch, self.policy, self.optim)
