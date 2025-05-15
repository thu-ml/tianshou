import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from sensai.util.helper import mark_used

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    ActBatchProtocol,
    BatchWithReturnsProtocol,
    ModelOutputBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.policy import Algorithm
from tianshou.policy.base import (
    LaggedNetworkFullUpdateAlgorithmMixin,
    OffPolicyAlgorithm,
    Policy,
    TArrOrActBatch,
)
from tianshou.policy.modelfree.pg import (
    SimpleLossTrainingStats,
)
from tianshou.policy.optim import OptimizerFactory
from tianshou.utils.lagged_network import EvalModeModuleWrapper
from tianshou.utils.net.common import Net

mark_used(ActBatchProtocol)

TModel = TypeVar("TModel", bound=torch.nn.Module | Net)
log = logging.getLogger(__name__)


class DiscreteQLearningPolicy(Policy, Generic[TModel]):
    def __init__(
        self,
        *,
        model: TModel,
        action_space: gym.spaces.Space,
        observation_space: gym.Space | None = None,
        eps_training: float = 0.0,
        eps_inference: float = 0.0,
    ) -> None:
        """
        :param model: a model mapping (obs, state, info) to action_values_BA.
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
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=False,
            action_bound_method=None,
        )
        self.model = model
        self.max_action_num: int | None = None
        self.eps_training = eps_training
        self.eps_inference = eps_inference

    def set_eps_training(self, eps: float) -> None:
        """
        Sets the epsilon value for epsilon-greedy exploration during training.

        :param eps: the epsilon value for epsilon-greedy exploration during training.
            When collecting data for training, this is the probability of choosing a random action
            instead of the action chosen by the policy.
            A value of 0.0 means no exploration (fully greedy) and a value of 1.0 means full
            exploration (fully random).
        """
        self.eps_training = eps

    def set_eps_inference(self, eps: float) -> None:
        """
        Sets the epsilon value for epsilon-greedy exploration during inference.

        :param eps: the epsilon value for epsilon-greedy exploration during inference,
            i.e. non-training cases (such as evaluation during test steps).
            The epsilon value is the probability of choosing a random action instead of the action
            chosen by the policy.
            A value of 0.0 means no exploration (fully greedy) and a value of 1.0 means full
            exploration (fully random).
        """
        self.eps_inference = eps

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> ModelOutputBatchProtocol:
        """Compute action over the given batch data.

        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.
        """
        if model is None:
            model = self.model
        obs = batch.obs
        # TODO: this is convoluted! See also other places where this is done.
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        action_values_BA, hidden_BH = model(obs_next, state=state, info=batch.info)
        q = self.compute_q_value(action_values_BA, getattr(obs, "mask", None))
        if self.max_action_num is None:
            self.max_action_num = q.shape[1]
        act_B = to_numpy(q.argmax(dim=1))
        result = Batch(logits=action_values_BA, act=act_B, state=hidden_BH)
        return cast(ModelOutputBatchProtocol, result)

    def compute_q_value(self, logits: torch.Tensor, mask: np.ndarray | None) -> torch.Tensor:
        """Compute the q value based on the network's raw output and action mask."""
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    def add_exploration_noise(
        self,
        act: TArrOrActBatch,
        batch: ObsBatchProtocol,
    ) -> TArrOrActBatch:
        eps = self.eps_training if self.is_within_training_step else self.eps_inference
        # TODO: This looks problematic; the non-array case is silently ignored
        if isinstance(act, np.ndarray) and not np.isclose(eps, 0.0):
            batch_size = len(act)
            rand_mask = np.random.rand(batch_size) < eps
            assert (
                self.max_action_num is not None
            ), "Can't call this method before max_action_num was set in first forward"
            q = np.random.rand(batch_size, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act


TDQNPolicy = TypeVar("TDQNPolicy", bound=DiscreteQLearningPolicy)


class QLearningOffPolicyAlgorithm(
    OffPolicyAlgorithm[TDQNPolicy], LaggedNetworkFullUpdateAlgorithmMixin, ABC
):
    """
    Base class for Q-learning off-policy algorithms that use a Q-function to compute the
    n-step return.
    It optionally uses a lagged model, which is used as a target network and which is
    fully updated periodically.
    """

    def __init__(
        self,
        *,
        policy: TDQNPolicy,
        optim: OptimizerFactory,
        gamma: float = 0.99,
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
        self.optim = self._create_policy_optimizer(optim)
        LaggedNetworkFullUpdateAlgorithmMixin.__init__(self)
        assert 0.0 <= gamma <= 1.0, f"discount factor should be in [0, 1] but got: {gamma}"
        self.gamma = gamma
        assert (
            estimation_step > 0
        ), f"estimation_step should be greater than 0 but got: {estimation_step}"
        self.n_step = estimation_step
        self.target_update_freq = target_update_freq
        # TODO: 1 would be a more reasonable initialization given how it is incremented
        self._iter = 0
        self.model_old: EvalModeModuleWrapper | None = (
            self._add_lagged_network(self.policy.model) if self.use_target_network else None
        )

    def _create_policy_optimizer(self, optim: OptimizerFactory) -> Algorithm.Optimizer:
        return self._create_optimizer(self.policy, optim)

    @property
    def use_target_network(self) -> bool:
        return self.target_update_freq > 0

    @abstractmethod
    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        pass

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithReturnsProtocol:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        return self.compute_nstep_return(
            batch=batch,
            buffer=buffer,
            indices=indices,
            target_q_fn=self._target_q,
            gamma=self.gamma,
            n_step=self.n_step,
        )

    def _periodically_update_lagged_network_weights(self) -> None:
        """
        Periodically updates the parameters of the lagged target network (if any), i.e.
        every n-th call (where n=`target_update_freq`), the target network's parameters
        are fully updated with the model's parameters.
        """
        if self.use_target_network and self._iter % self.target_update_freq == 0:
            self._update_lagged_network_weights()
        self._iter += 1


class DQN(
    QLearningOffPolicyAlgorithm[TDQNPolicy],
    Generic[TDQNPolicy],
):
    """Implementation of Deep Q Network. arXiv:1312.5602.

    Implementation of Double Q-Learning. arXiv:1509.06461.

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here).
    """

    def __init__(
        self,
        *,
        policy: TDQNPolicy,
        optim: OptimizerFactory,
        gamma: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        is_double: bool = True,
        huber_loss_delta: float | None = None,
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
        :param is_double: flag indicating whether to use the Double DQN algorithm for target value computation.
            If True, the algorithm uses the online network to select actions and the target network to
            evaluate their Q-values. This approach helps reduce the overestimation bias in Q-learning
            by decoupling action selection from action evaluation.
            If False, the algorithm follows the vanilla DQN method that directly takes the maximum Q-value
            from the target network.
            Note: Double Q-learning will only be effective when a target network is used (target_update_freq > 0).
        :param huber_loss_delta: controls whether to use the Huber loss instead of the MSE loss for the TD error
            and the threshold for the Huber loss.
            If None, the MSE loss is used.
            If not None, uses the Huber loss as described in the Nature DQN paper (nature14236) with the given delta,
            which limits the influence of outliers.
            Unlike the MSE loss where the gradients grow linearly with the error magnitude, the Huber
            loss causes the gradients to plateau at a constant value for large errors, providing more stable training.
            NOTE: The magnitude of delta should depend on the scale of the returns obtained in the environment.
        """
        super().__init__(
            policy=policy,
            optim=optim,
            gamma=gamma,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
        )
        self.is_double = is_double
        self.huber_loss_delta = huber_loss_delta

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
            return target_q[np.arange(len(result.act)), result.act]
        # Nature DQN, over estimate
        return target_q.max(dim=1)[0]

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> SimpleLossTrainingStats:
        self._periodically_update_lagged_network_weights()
        weight = batch.pop("weight", 1.0)
        q = self.policy(batch).logits
        q = q[np.arange(len(q)), batch.act]
        returns = to_torch_as(batch.returns.flatten(), q)
        td_error = returns - q

        if self.huber_loss_delta is not None:
            y = q.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(
                y, t, delta=self.huber_loss_delta, reduction="mean"
            )
        else:
            loss = (td_error.pow(2) * weight).mean()

        batch.weight = td_error  # prio-buffer
        self.optim.step(loss)

        return SimpleLossTrainingStats(loss=loss.item())
