from dataclasses import dataclass
from typing import Any, Literal, Self, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from overrides import override
from torch.nn.utils import clip_grad_norm_

from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.data.buffer.base import TBuffer
from tianshou.data.types import RolloutBatchProtocol
from tianshou.exploration import BaseNoise
from tianshou.policy import SACPolicy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.policy.modelfree.sac import SACTrainingStats
from tianshou.utils.conversion import to_optional_float
from tianshou.utils.net.continuous import ActorProb


@dataclass(kw_only=True)
class CQLTrainingStats(SACTrainingStats):
    """A data structure for storing loss statistics of the CQL learn step."""

    cql_alpha: float | None = None
    cql_alpha_loss: float | None = None


TCQLTrainingStats = TypeVar("TCQLTrainingStats", bound=CQLTrainingStats)


class CQLPolicy(SACPolicy[TCQLTrainingStats]):
    """Implementation of CQL algorithm. arXiv:2006.04779.

    :param actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> a)
    :param actor_optim: The optimizer for actor network.
    :param critic: The first critic network.
    :param critic_optim: The optimizer for the first critic network.
    :param action_space: Env's action space.
    :param critic2: the second critic network. (s, a -> Q(s, a)).
        If None, use the same network as critic (via deepcopy).
    :param critic2_optim: the optimizer for the second critic network.
        If None, clone critic_optim to use for critic2.parameters().
    :param cql_alpha_lr: The learning rate of cql_log_alpha.
    :param cql_weight:
    :param tau: Parameter for soft update of the target network.
    :param gamma: Discount factor, in [0, 1].
    :param alpha: Entropy regularization coefficient or a tuple
        (target_entropy, log_alpha, alpha_optim) for automatic tuning.
    :param temperature:
    :param with_lagrange: Whether to use Lagrange.
        TODO: extend documentation - what does this mean?
    :param lagrange_threshold: The value of tau in CQL(Lagrange).
    :param min_action: The minimum value of each dimension of action.
    :param max_action: The maximum value of each dimension of action.
    :param num_repeat_actions: The number of times the action is repeated when calculating log-sum-exp.
    :param alpha_min: Lower bound for clipping cql_alpha.
    :param alpha_max: Upper bound for clipping cql_alpha.
    :param clip_grad: Clip_grad for updating critic network.
    :param calibrated: calibrate Q-values as in CalQL paper `arXiv:2303.05479`.
        Useful for offline pre-training followed by online training,
        and also was observed to achieve better results than vanilla cql.
    :param device: Which device to create this model on.
    :param estimation_step: Estimation steps.
    :param exploration_noise: Type of exploration noise.
    :param deterministic_eval: Flag for deterministic evaluation.
    :param action_scaling: Flag for action scaling.
    :param action_bound_method: Method for action bounding. Only used if the
        action_space is continuous.
    :param observation_space: Env's Observation space.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update().

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        *,
        actor: ActorProb,
        actor_optim: torch.optim.Optimizer,
        critic: torch.nn.Module,
        critic_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Box,
        critic2: torch.nn.Module | None = None,
        critic2_optim: torch.optim.Optimizer | None = None,
        cql_alpha_lr: float = 1e-4,
        cql_weight: float = 1.0,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float | tuple[float, torch.Tensor, torch.optim.Optimizer] = 0.2,
        temperature: float = 1.0,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        min_action: float = -1.0,
        max_action: float = 1.0,
        num_repeat_actions: int = 10,
        alpha_min: float = 0.0,
        alpha_max: float = 1e6,
        clip_grad: float = 1.0,
        calibrated: bool = True,
        # TODO: why does this one have device? Almost no other policies have it
        device: str | torch.device = "cpu",
        estimation_step: int = 1,
        exploration_noise: BaseNoise | Literal["default"] | None = None,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: Literal["clip"] | None = "clip",
        observation_space: gym.Space | None = None,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            actor_optim=actor_optim,
            critic=critic,
            critic_optim=critic_optim,
            action_space=action_space,
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=tau,
            gamma=gamma,
            deterministic_eval=deterministic_eval,
            alpha=alpha,
            exploration_noise=exploration_noise,
            estimation_step=estimation_step,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=observation_space,
            lr_scheduler=lr_scheduler,
        )
        # There are _target_entropy, _log_alpha, _alpha_optim in SACPolicy.
        self.device = device
        self.temperature = temperature
        self.with_lagrange = with_lagrange
        self.lagrange_threshold = lagrange_threshold

        self.cql_weight = cql_weight

        self.cql_log_alpha = torch.tensor([0.0], requires_grad=True)
        self.cql_alpha_optim = torch.optim.Adam([self.cql_log_alpha], lr=cql_alpha_lr)
        self.cql_log_alpha = self.cql_log_alpha.to(device)

        self.min_action = min_action
        self.max_action = max_action

        self.num_repeat_actions = num_repeat_actions

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.clip_grad = clip_grad

        self.calibrated = calibrated

    def train(self, mode: bool = True) -> Self:
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        self.soft_update(self.critic_old, self.critic, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)

    def actor_pred(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = Batch(obs=obs, info=[None] * len(obs))
        obs_result = self(batch)
        return obs_result.act, obs_result.log_prob

    def calc_actor_loss(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        act_pred, log_pi = self.actor_pred(obs)
        q1 = self.critic(obs, act_pred)
        q2 = self.critic2(obs, act_pred)
        min_Q = torch.min(q1, q2)
        # self.alpha: float | torch.Tensor
        actor_loss = (self.alpha * log_pi - min_Q).mean()
        # actor_loss.shape: (), log_pi.shape: (batch_size, 1)
        return actor_loss, log_pi

    def calc_pi_values(
        self,
        obs_pi: torch.Tensor,
        obs_to_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        act_pred, log_pi = self.actor_pred(obs_pi)

        q1 = self.critic(obs_to_pred, act_pred)
        q2 = self.critic2(obs_to_pred, act_pred)

        return q1 - log_pi.detach(), q2 - log_pi.detach()

    def calc_random_values(
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

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        # TODO: mypy rightly complains here b/c the design violates
        #   Liskov Substitution Principle
        #   DDPGPolicy.process_fn() results in a batch with returns but
        #   CQLPolicy.process_fn() doesn't add the returns.
        #   Should probably be fixed!
        return batch

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TCQLTrainingStats:  # type: ignore
        batch: Batch = to_torch(batch, dtype=torch.float, device=self.device)
        obs, act, rew, obs_next = batch.obs, batch.act, batch.rew, batch.obs_next
        batch_size = obs.shape[0]

        # compute actor loss and update actor
        actor_loss, log_pi = self.calc_actor_loss(obs)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = None
        # compute alpha loss
        if self.is_auto_alpha:
            log_pi = log_pi + self.target_entropy
            alpha_loss = -(self.log_alpha * log_pi.detach()).mean()
            self.alpha_optim.zero_grad()
            # update log_alpha
            alpha_loss.backward()
            self.alpha_optim.step()
            # update alpha
            # TODO: it's probably a bad idea to track both alpha and log_alpha in different fields
            self.alpha = self.log_alpha.detach().exp()

        # compute target_Q
        with torch.no_grad():
            act_next, new_log_pi = self.actor_pred(obs_next)

            target_Q1 = self.critic_old(obs_next, act_next)
            target_Q2 = self.critic2_old(obs_next, act_next)

            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * new_log_pi

            target_Q = rew + self.gamma * (1 - batch.done) * target_Q.flatten()
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
            .to(self.device)
        )

        obs_len = len(obs.shape)
        repeat_size = [1, self.num_repeat_actions] + [1] * (obs_len - 1)
        view_size = [batch_size * self.num_repeat_actions, *list(obs.shape[1:])]
        tmp_obs = obs.unsqueeze(1).repeat(*repeat_size).view(*view_size)
        tmp_obs_next = obs_next.unsqueeze(1).repeat(*repeat_size).view(*view_size)
        # tmp_obs & tmp_obs_next: (batch_size * num_repeat, state_dim)

        current_pi_value1, current_pi_value2 = self.calc_pi_values(tmp_obs, tmp_obs)
        next_pi_value1, next_pi_value2 = self.calc_pi_values(tmp_obs_next, tmp_obs)

        random_value1, random_value2 = self.calc_random_values(tmp_obs, random_actions)

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

        # update critic
        self.critic_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        # clip grad, prevent the vanishing gradient problem
        # It doesn't seem necessary
        clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        self.critic_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad)
        self.critic2_optim.step()

        self.sync_weight()

        return CQLTrainingStats(  # type: ignore[return-value]
            actor_loss=to_optional_float(actor_loss),
            critic1_loss=to_optional_float(critic1_loss),
            critic2_loss=to_optional_float(critic2_loss),
            alpha=to_optional_float(self.alpha),
            alpha_loss=to_optional_float(alpha_loss),
            cql_alpha_loss=to_optional_float(cql_alpha_loss),
            cql_alpha=to_optional_float(cql_alpha),
        )
