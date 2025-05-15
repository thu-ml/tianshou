from dataclasses import dataclass
from typing import Any, Literal, cast

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
from tianshou.algorithm import Algorithm
from tianshou.algorithm.algorithm_base import (
    OfflineAlgorithm,
    OffPolicyAlgorithm,
    Policy,
    TrainingStats,
)
from tianshou.algorithm.optim import OptimizerFactory

# Dimension Naming Convention
# B - Batch Size
# A - Action
# D - Dist input (usually 2, loc and scale)
# H - Dimension of hidden, can be None


@dataclass(kw_only=True)
class ImitationTrainingStats(TrainingStats):
    loss: float = 0.0


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
        :param action_scaling: flag indicating whether, for continuous action spaces, actions
            should be scaled from the standard neural network output range [-1, 1] to the
            environment's action space range [action_space.low, action_space.high].
            This applies to continuous action spaces only (gym.spaces.Box) and has no effect
            for discrete spaces.
            When enabled, policy outputs are expected to be in the normalized range [-1, 1]
            (after bounding), and are then linearly transformed to the actual required range.
            This improves neural network training stability, allows the same algorithm to work
            across environments with different action ranges, and standardizes exploration
            strategies.
            Should be disabled if the actor model already produces outputs in the correct range.
        :param action_bound_method: the method used for bounding actions in continuous action spaces
            to the range [-1, 1] before scaling them to the environment's action space (provided
            that `action_scaling` is enabled).
            This applies to continuous action spaces only (`gym.spaces.Box`) and should be set to None
            for discrete spaces.
            When set to "clip", actions exceeding the [-1, 1] range are simply clipped to this
            range. When set to "tanh", a hyperbolic tangent function is applied, which smoothly
            constrains outputs to [-1, 1] while preserving gradients.
            The choice of bounding method affects both training dynamics and exploration behavior.
            Clipping provides hard boundaries but may create plateau regions in the gradient
            landscape, while tanh provides smoother transitions but can compress sensitivity
            near the boundaries.
            Should be set to None if the actor model inherently produces bounded outputs.
            Typically used together with `action_scaling=True`.
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
    OffPolicyAlgorithm[ImitationPolicy],
    ImitationLearningAlgorithmMixin,
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
    ) -> ImitationTrainingStats:
        return self._imitation_update(batch, self.policy, self.optim)


class OfflineImitationLearning(
    OfflineAlgorithm[ImitationPolicy],
    ImitationLearningAlgorithmMixin,
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
    ) -> ImitationTrainingStats:
        return self._imitation_update(batch, self.policy, self.optim)
