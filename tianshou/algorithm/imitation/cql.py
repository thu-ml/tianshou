import math
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from overrides import override

from tianshou.algorithm.algorithm_base import (
    LaggedNetworkPolyakUpdateAlgorithmMixin,
    OfflineAlgorithm,
)
from tianshou.algorithm.modelfree.sac import Alpha, SACPolicy, SACTrainingStats
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.data.buffer.buffer_base import TBuffer
from tianshou.data.types import RolloutBatchProtocol
from tianshou.utils.conversion import to_optional_float
from tianshou.utils.torch_utils import torch_device


def _sample_uniform_actions(
    act: torch.Tensor, num_repeat: int, low: float, high: float
) -> tuple[torch.Tensor, float]:
    """Sample in the critic's action space and return the joint log-density."""
    action_dim = act.shape[-1]
    samples = act.new_empty((len(act) * num_repeat, action_dim)).uniform_(low, high)
    return samples, -action_dim * math.log(high - low)


def _conservative_loss(
    data_q: torch.Tensor,
    sample_values: tuple[torch.Tensor, ...],
    temperature: float,
    calibration_returns: torch.Tensor | None,
) -> torch.Tensor:
    """Pool the importance-corrected values of all proposals for each state."""
    # Each state occupies N consecutive rows. Group (B * N, 3) into (B, 3 * N).
    values = torch.cat(sample_values, dim=1).reshape(len(data_q), -1)
    if calibration_returns is not None:
        values = torch.maximum(values, calibration_returns.reshape(-1, 1))
    # The reference CQL implementation uses the unnormalized sum for the Lagrange gap.
    return temperature * torch.logsumexp(values / temperature, dim=1).mean() - data_q.mean()


@dataclass(kw_only=True)
class CQLTrainingStats(SACTrainingStats):
    """A data structure for storing loss statistics of the CQL learn step."""

    cql_alpha: float | None = None
    cql_alpha_loss: float | None = None


# TODO: Perhaps SACPolicy should get a more generic name
class CQL(OfflineAlgorithm[SACPolicy], LaggedNetworkPolyakUpdateAlgorithmMixin):
    """Implementation of the conservative Q-learning (CQL) algorithm. arXiv:2006.04779."""

    def __init__(
        self,
        *,
        policy: SACPolicy,
        policy_optim: OptimizerFactory,
        critic: torch.nn.Module,
        critic_optim: OptimizerFactory,
        critic2: torch.nn.Module | None = None,
        critic2_optim: OptimizerFactory | None = None,
        cql_alpha_lr: float = 1e-4,
        cql_weight: float = 1.0,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float | Alpha = 0.2,
        temperature: float = 1.0,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        min_action: float = -1.0,
        max_action: float = 1.0,
        num_repeat_actions: int = 10,
        alpha_min: float = 0.0,
        alpha_max: float = 1e6,
        max_grad_norm: float = 1.0,
        calibrated: bool = True,
    ) -> None:
        """
        :param policy: the :class:`~tianshou.algorithm.modelfree.sac.SACPolicy` constructed
            with ``actor`` and ``action_space``.
        :param policy_optim: the optimizer factory for the policy/its actor network.
        :param critic: the first critic network.
        :param critic_optim: the optimizer factory for the first critic network.
        :param critic2: the second critic network. (s, a -> Q(s, a)).
            If None, use the same network as critic (via deepcopy).
        :param critic2_optim: the optimizer factory for the second critic network.
            If None, clone the first critic's optimizer factory.
        :param cql_alpha_lr: the learning rate for the Lagrange multiplier optimization.
            Controls how quickly the CQL regularization coefficient (alpha) adapts during training.
            Higher values allow faster adaptation but may cause instability in the training process.
            Lower values provide more stable but slower adaptation of the regularization strength.
            Only relevant when with_lagrange=True.
        :param cql_weight: the coefficient that scales the conservative regularization term in the Q-function loss.
            Controls the strength of the conservative Q-learning component relative to standard TD learning.
            Higher values enforce more conservative value estimates by penalizing overestimation more strongly.
            Lower values allow the algorithm to behave more like standard Q-learning.
            Increasing this weight typically improves performance in purely offline settings where
            overestimation bias can lead to poor policy extraction.
        :param tau: the soft update coefficient for target networks, controlling the rate at which
            target networks track the learned networks.
            When the parameters of the target network are updated with the current (source) network's
            parameters, a weighted average is used: target = tau * source + (1 - tau) * target.
            Smaller values (closer to 0) create more stable but slower learning as target networks
            change more gradually. Higher values (closer to 1) allow faster learning but may reduce
            stability.
            Typically set to a small value (0.001 to 0.01) for most reinforcement learning tasks.
        :param gamma: the discount factor in [0, 1] for future rewards.
            This determines how much future rewards are valued compared to immediate ones.
            Lower values (closer to 0) make the agent focus on immediate rewards, creating "myopic"
            behavior. Higher values (closer to 1) make the agent value long-term rewards more,
            potentially improving performance in tasks where delayed rewards are important but
            increasing training variance by incorporating more environmental stochasticity.
            Typically set between 0.9 and 0.99 for most reinforcement learning tasks
        :param alpha: the entropy regularization coefficient alpha or an object
            which can be used to automatically tune it (e.g. an instance of `AutoAlpha`).
        :param temperature: the temperature parameter used in the LogSumExp calculation of the CQL loss.
            Controls the sharpness of the softmax distribution when computing the expected Q-values.
            Lower values make the LogSumExp operation more selective, focusing on the highest Q-values.
            Higher values make the operation closer to an average, giving more weight to all Q-values.
            The temperature affects how conservatively the algorithm penalizes out-of-distribution actions.
        :param with_lagrange: a flag indicating whether to automatically tune the CQL regularization strength.
            If True, uses Lagrangian dual gradient descent to dynamically adjust the CQL alpha parameter.
            This formulation maintains the CQL regularization loss near the lagrange_threshold value.
            Adaptive tuning helps balance conservative learning against excessive pessimism.
            If False, the conservative loss is scaled by a fixed cql_weight throughout training.
            The original CQL paper recommends setting this to True for most offline RL tasks.
        :param lagrange_threshold: the target value for the CQL regularization loss when using Lagrangian optimization.
            When with_lagrange=True, the algorithm dynamically adjusts the CQL alpha parameter to maintain
            the regularization loss close to this threshold.
            Lower values result in more conservative behavior by enforcing stronger penalties on
            out-of-distribution actions.
            Higher values allow more optimistic Q-value estimates similar to standard Q-learning.
            This threshold effectively controls the level of conservatism in CQL's value estimation.
        :param min_action: lower bound for uniform sampling in each action dimension.
            Both bounds must use the same coordinates as the policy outputs and replay actions
            seen by the critic. SAC uses ``[-1, 1]`` before scaling actions for the environment.
        :param max_action: upper bound in the same critic action space as ``min_action``.
            The bounds must be finite and ``min_action < max_action``.
        :param num_repeat_actions: draws per state from each of the uniform, current-state
            policy and next-state policy distributions. All are evaluated at the current state.
            The unnormalized log-sum-exp makes the Lagrange gap depend on the number of draws.
        :param alpha_min: the minimum value allowed for the adaptive CQL regularization coefficient.
            When using Lagrangian optimization (with_lagrange=True), constrains the automatically tuned
            cql_alpha parameter to be at least this value.
            Prevents the regularization strength from becoming too small during training.
            Setting a positive value ensures the algorithm maintains at least some degree of conservatism.
            Only relevant when with_lagrange=True.
        :param alpha_max: the maximum value allowed for the adaptive CQL regularization coefficient.
            When using Lagrangian optimization (with_lagrange=True), constrains the automatically tuned
            cql_alpha parameter to be at most this value.
            Prevents the regularization strength from becoming too large during training.
            Setting an appropriate upper limit helps avoid overly conservative behavior that might hinder
            learning useful value functions.
            Only relevant when with_lagrange=True.
        :param max_grad_norm: the maximum L2 norm threshold for gradient clipping when updating critic networks.
            Gradients with norm exceeding this value will be rescaled to have norm equal to this value.
            Helps stabilize training by preventing excessively large parameter updates from outlier samples.
            Higher values allow larger updates but may lead to training instability.
            Lower values enforce more conservative updates but may slow down learning.
            Setting to a large value effectively disables gradient clipping.
        :param calibrated: a flag indicating whether to use the calibrated version of CQL (CalQL).
            If True, calibrates Q-values by taking the maximum of computed Q-values and Monte Carlo returns.
            This modification helps address the excessive pessimism problem in standard CQL.
            Particularly useful for offline pre-training followed by online fine-tuning scenarios.
            Experimental results suggest this approach often achieves better performance than vanilla CQL.
            Based on techniques from the CalQL paper (arXiv:2303.05479).
        """
        if not 0 < max_action - min_action < math.inf:
            raise ValueError("CQL requires finite action bounds with min_action < max_action")
        super().__init__(
            policy=policy,
        )
        LaggedNetworkPolyakUpdateAlgorithmMixin.__init__(self, tau=tau)

        device = torch_device(policy)

        self.policy_optim = self._create_optimizer(self.policy, policy_optim)
        self.critic = critic
        self.critic_optim = self._create_optimizer(
            self.critic, critic_optim, max_grad_norm=max_grad_norm
        )
        self.critic2 = critic2 or deepcopy(critic)
        self.critic2_optim = self._create_optimizer(
            self.critic2, critic2_optim or critic_optim, max_grad_norm=max_grad_norm
        )
        self.critic_old = self._add_lagged_network(self.critic)
        self.critic2_old = self._add_lagged_network(self.critic2)

        self.gamma = gamma
        self.alpha = Alpha.from_float_or_instance(alpha)

        self.temperature = temperature
        self.with_lagrange = with_lagrange
        self.lagrange_threshold = lagrange_threshold

        self.cql_weight = cql_weight

        self.cql_log_alpha = torch.tensor([0.0], requires_grad=True)
        # TODO: Use an OptimizerFactory?
        self.cql_alpha_optim = torch.optim.Adam([self.cql_log_alpha], lr=cql_alpha_lr)
        self.cql_log_alpha = self.cql_log_alpha.to(device)

        self.min_action = min_action
        self.max_action = max_action

        self.num_repeat_actions = num_repeat_actions

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.calibrated = calibrated

    def _policy_pred(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = Batch(obs=obs, info=[None] * len(obs))
        obs_result = self.policy(batch)
        return obs_result.act, obs_result.log_prob

    def _calc_policy_loss(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        act_pred, log_pi = self._policy_pred(obs)
        q1 = self.critic(obs, act_pred)
        q2 = self.critic2(obs, act_pred)
        min_Q = torch.min(q1, q2)
        # self.alpha: float | torch.Tensor
        actor_loss = (self.alpha.value * log_pi - min_Q).mean()
        # actor_loss.shape: (), log_pi.shape: (batch_size, 1)
        return actor_loss, log_pi

    def _calc_pi_values(
        self,
        obs_pi: torch.Tensor,
        obs_to_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        act_pred, log_pi = self._policy_pred(obs_pi)

        q1 = self.critic(obs_to_pred, act_pred)
        q2 = self.critic2(obs_to_pred, act_pred)

        return q1 - log_pi.detach(), q2 - log_pi.detach()

    @override
    def process_buffer(self, buffer: TBuffer) -> TBuffer:
        """If `self.calibrated = True`, adds `calibration_returns` to buffer._meta.

        :param buffer:
        :return:
        """
        if self.calibrated:
            # otherwise _meta hack cannot work
            assert isinstance(buffer, ReplayBuffer)
            batch, indices = buffer.sample(0)
            returns, _ = self.compute_episodic_return(
                batch=batch,
                buffer=buffer,
                indices=indices,
                gamma=self.gamma,
                gae_lambda=1.0,
            )
            # TODO: don't access _meta directly
            buffer._meta = cast(
                RolloutBatchProtocol,
                Batch(**buffer._meta.__dict__, calibration_returns=returns),
            )
        return buffer

    def _update_with_batch(self, batch: RolloutBatchProtocol) -> CQLTrainingStats:
        device = torch_device(self.policy)
        data: Batch = to_torch(batch, dtype=torch.float, device=device)
        obs, act, rew, obs_next = data.obs, data.act, data.rew, data.obs_next

        # compute actor loss and update actor
        actor_loss, log_pi = self._calc_policy_loss(obs)
        self.policy_optim.step(actor_loss)

        entropy = -log_pi.detach()
        alpha_loss = self.alpha.update(entropy)

        # compute target_Q
        with torch.no_grad():
            act_next, new_log_pi = self._policy_pred(obs_next)

            target_Q1 = self.critic_old(obs_next, act_next)
            target_Q2 = self.critic2_old(obs_next, act_next)

            target_Q = torch.min(target_Q1, target_Q2) - self.alpha.value * new_log_pi

            target_Q = rew + torch.logical_not(data.done) * self.gamma * target_Q.flatten()
            target_Q = target_Q.float()
            # shape: (batch_size)

        # compute critic loss
        current_Q1 = self.critic(obs, act).flatten()
        current_Q2 = self.critic2(obs, act).flatten()
        # shape: (batch_size)

        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        random_actions, random_log_prob = _sample_uniform_actions(
            act, self.num_repeat_actions, self.min_action, self.max_action
        )
        repeated_obs = obs.repeat_interleave(self.num_repeat_actions, dim=0)
        repeated_next_obs = obs_next.repeat_interleave(self.num_repeat_actions, dim=0)
        current_pi1, current_pi2 = self._calc_pi_values(repeated_obs, repeated_obs)
        next_pi1, next_pi2 = self._calc_pi_values(repeated_next_obs, repeated_obs)
        random_q1 = self.critic(repeated_obs, random_actions) - random_log_prob
        random_q2 = self.critic2(repeated_obs, random_actions) - random_log_prob

        returns = data.calibration_returns if self.calibrated else None
        cql1_scaled_loss = self.cql_weight * _conservative_loss(
            current_Q1, (random_q1, current_pi1, next_pi1), self.temperature, returns
        )
        cql2_scaled_loss = self.cql_weight * _conservative_loss(
            current_Q2, (random_q2, current_pi2, next_pi2), self.temperature, returns
        )

        cql_alpha_loss = None
        cql_alpha = None
        if self.with_lagrange:
            cql_alpha = torch.clamp(
                self.cql_log_alpha.exp(),
                self.alpha_min,
                self.alpha_max,
            )
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.lagrange_threshold)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.lagrange_threshold)

            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss = -(cql1_scaled_loss + cql2_scaled_loss) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()

        critic1_loss = critic1_loss + cql1_scaled_loss
        critic2_loss = critic2_loss + cql2_scaled_loss

        # update critics
        self.critic_optim.step(critic1_loss, retain_graph=True)
        self.critic2_optim.step(critic2_loss)

        self._update_lagged_network_weights()

        return CQLTrainingStats(
            actor_loss=to_optional_float(actor_loss),
            critic1_loss=to_optional_float(critic1_loss),
            critic2_loss=to_optional_float(critic2_loss),
            alpha=to_optional_float(self.alpha.value),
            alpha_loss=to_optional_float(alpha_loss),
            cql_alpha_loss=to_optional_float(cql_alpha_loss),
            cql_alpha=to_optional_float(cql_alpha),
        )
