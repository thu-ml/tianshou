from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import numpy as np
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
        :param actor: the actor network following the rules (s -> a)
        :param policy_optim: the optimizer factory for the policy/its actor network.
        :param critic: the first critic network.
        :param critic_optim: the optimizer factory for the first critic network.
        :param action_space: the environment's action space.
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
        :param min_action: the lower bound for each dimension of the action space.
            Used when sampling random actions for the CQL regularization term.
            Should match the environment's action space minimum values.
            These random actions help penalize Q-values for out-of-distribution actions.
            Typically set to -1.0 for normalized continuous action spaces.
        :param max_action: the upper bound for each dimension of the action space.
            Used when sampling random actions for the CQL regularization term.
            Should match the environment's action space maximum values.
            These random actions help penalize Q-values for out-of-distribution actions.
            Typically set to 1.0 for normalized continuous action spaces.
        :param num_repeat_actions: the number of action samples generated per state when computing
            the CQL regularization term.
            Controls how many random and policy actions are sampled for each state in the batch when
            estimating expected Q-values.
            Higher values provide more accurate approximation of the expected Q-values but increase
            computational cost.
            Lower values reduce computation but may provide less stable or less accurate regularization.
            The original CQL paper typically uses values around 10.
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

    def _calc_random_values(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        random_value1 = self.critic(obs, act)
        random_log_prob1 = np.log(0.5 ** act.shape[-1])

        random_value2 = self.critic2(obs, act)
        random_log_prob2 = np.log(0.5 ** act.shape[-1])

        return random_value1 - random_log_prob1, random_value2 - random_log_prob2

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
        batch: Batch = to_torch(batch, dtype=torch.float, device=device)
        obs, act, rew, obs_next = batch.obs, batch.act, batch.rew, batch.obs_next
        batch_size = obs.shape[0]

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

            target_Q = rew + torch.logical_not(batch.done) * self.gamma * target_Q.flatten()
            target_Q = target_Q.float()
            # shape: (batch_size)

        # compute critic loss
        current_Q1 = self.critic(obs, act).flatten()
        current_Q2 = self.critic2(obs, act).flatten()
        # shape: (batch_size)

        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        # CQL
        random_actions = (
            torch.FloatTensor(batch_size * self.num_repeat_actions, act.shape[-1])
            .uniform_(-self.min_action, self.max_action)
            .to(device)
        )

        obs_len = len(obs.shape)
        repeat_size = [1, self.num_repeat_actions] + [1] * (obs_len - 1)
        view_size = [batch_size * self.num_repeat_actions, *list(obs.shape[1:])]
        tmp_obs = obs.unsqueeze(1).repeat(*repeat_size).view(*view_size)
        tmp_obs_next = obs_next.unsqueeze(1).repeat(*repeat_size).view(*view_size)
        # tmp_obs & tmp_obs_next: (batch_size * num_repeat, state_dim)

        current_pi_value1, current_pi_value2 = self._calc_pi_values(tmp_obs, tmp_obs)
        next_pi_value1, next_pi_value2 = self._calc_pi_values(tmp_obs_next, tmp_obs)

        random_value1, random_value2 = self._calc_random_values(tmp_obs, random_actions)

        for value in [
            current_pi_value1,
            current_pi_value2,
            next_pi_value1,
            next_pi_value2,
            random_value1,
            random_value2,
        ]:
            value.reshape(batch_size, self.num_repeat_actions, 1)

        if self.calibrated:
            returns = (
                batch.calibration_returns.unsqueeze(1)
                .repeat(
                    (1, self.num_repeat_actions),
                )
                .view(-1, 1)
            )
            random_value1 = torch.max(random_value1, returns)
            random_value2 = torch.max(random_value2, returns)

            current_pi_value1 = torch.max(current_pi_value1, returns)
            current_pi_value2 = torch.max(current_pi_value2, returns)

            next_pi_value1 = torch.max(next_pi_value1, returns)
            next_pi_value2 = torch.max(next_pi_value2, returns)

        # cat q values
        cat_q1 = torch.cat([random_value1, current_pi_value1, next_pi_value1], 1)
        cat_q2 = torch.cat([random_value2, current_pi_value2, next_pi_value2], 1)
        # shape: (batch_size, 3 * num_repeat, 1)

        cql1_scaled_loss = (
            torch.logsumexp(cat_q1 / self.temperature, dim=1).mean()
            * self.cql_weight
            * self.temperature
            - current_Q1.mean() * self.cql_weight
        )
        cql2_scaled_loss = (
            torch.logsumexp(cat_q2 / self.temperature, dim=1).mean()
            * self.cql_weight
            * self.temperature
            - current_Q2.mean() * self.cql_weight
        )
        # shape: (1)

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
