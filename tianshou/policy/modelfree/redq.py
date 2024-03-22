from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Independent, Normal

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.types import ObsBatchProtocol, RolloutBatchProtocol
from tianshou.exploration import BaseNoise
from tianshou.policy import DDPGPolicy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.policy.modelfree.ddpg import DDPGTrainingStats


@dataclass
class REDQTrainingStats(DDPGTrainingStats):
    """A data structure for storing loss statistics of the REDQ learn step."""

    alpha: float | None = None
    alpha_loss: float | None = None


TREDQTrainingStats = TypeVar("TREDQTrainingStats", bound=REDQTrainingStats)


class REDQPolicy(DDPGPolicy[TREDQTrainingStats]):
    """Implementation of REDQ. arXiv:2101.05982.

    :param actor: The actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> model_output)
    :param actor_optim: The optimizer for actor network.
    :param critic: The critic network. (s, a -> Q(s, a))
    :param critic_optim: The optimizer for critic network.
    :param action_space: Env's action space.
    :param ensemble_size: Number of sub-networks in the critic ensemble.
    :param subset_size: Number of networks in the subset.
    :param tau: Param for soft update of the target network.
    :param gamma: Discount factor, in [0, 1].
    :param alpha: entropy regularization coefficient.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatically tuned.
    :param exploration_noise: The exploration noise, added to the action. Defaults
        to ``GaussianNoise(sigma=0.1)``.
    :param estimation_step: The number of steps to look ahead.
    :param actor_delay: Number of critic updates before an actor update.
    :param observation_space: Env's observation space.
    :param action_scaling: if True, scale the action from [-1, 1] to the range
        of action_space. Only used if the action_space is continuous.
    :param action_bound_method: method to bound action to range [-1, 1].
        Only used if the action_space is continuous.
    :param lr_scheduler: if not None, will be called in `policy.update()`.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        *,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic: torch.nn.Module,
        critic_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Box,
        ensemble_size: int = 10,
        subset_size: int = 2,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float | tuple[float, torch.Tensor, torch.optim.Optimizer] = 0.2,
        estimation_step: int = 1,
        actor_delay: int = 20,
        exploration_noise: BaseNoise | Literal["default"] | None = None,
        deterministic_eval: bool = True,
        target_mode: Literal["mean", "min"] = "min",
        action_scaling: bool = True,
        action_bound_method: Literal["clip"] | None = "clip",
        observation_space: gym.Space | None = None,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        if target_mode not in ("min", "mean"):
            raise ValueError(f"Unsupported target_mode: {target_mode}")
        if not 0 < subset_size <= ensemble_size:
            raise ValueError(
                f"Invalid choice of ensemble size or subset size. "
                f"Should be 0 < {subset_size=} <= {ensemble_size=}",
            )
        super().__init__(
            actor=actor,
            actor_optim=actor_optim,
            critic=critic,
            critic_optim=critic_optim,
            action_space=action_space,
            tau=tau,
            gamma=gamma,
            exploration_noise=exploration_noise,
            estimation_step=estimation_step,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=observation_space,
            lr_scheduler=lr_scheduler,
        )
        self.ensemble_size = ensemble_size
        self.subset_size = subset_size

        self.target_mode = target_mode
        self.critic_gradient_step = 0
        self.actor_delay = actor_delay
        self.deterministic_eval = deterministic_eval
        self.__eps = np.finfo(np.float32).eps.item()

        self._last_actor_loss = 0.0  # only for logging purposes

        # TODO: reduce duplication with SACPolicy
        self.alpha: float | torch.Tensor
        self._is_auto_alpha = not isinstance(alpha, float)
        if self._is_auto_alpha:
            # TODO: why doesn't mypy understand that this must be a tuple?
            alpha = cast(tuple[float, torch.Tensor, torch.optim.Optimizer], alpha)
            if alpha[1].shape != torch.Size([1]):
                raise ValueError(
                    f"Expected log_alpha to have shape torch.Size([1]), "
                    f"but got {alpha[1].shape} instead.",
                )
            if not alpha[1].requires_grad:
                raise ValueError("Expected log_alpha to require gradient, but it doesn't.")

            self.target_entropy, self.log_alpha, self.alpha_optim = alpha
            self.alpha = self.log_alpha.detach().exp()
        else:
            # TODO: make mypy undestand this, or switch to something like pyright...
            alpha = cast(float, alpha)
            self.alpha = alpha

    @property
    def is_auto_alpha(self) -> bool:
        return self._is_auto_alpha

    # TODO: why override from the base class?
    def sync_weight(self) -> None:
        for o, n in zip(self.critic_old.parameters(), self.critic.parameters(), strict=True):
            o.data.copy_(o.data * (1.0 - self.tau) + n.data * self.tau)

    def forward(  # type: ignore
        self,
        batch: ObsBatchProtocol,
        state: dict | Batch | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Batch:
        loc_scale, h = self.actor(batch.obs, state=state, info=batch.info)
        loc, scale = loc_scale
        dist = Independent(Normal(loc, scale), 1)
        if self.deterministic_eval and not self.training:
            act = dist.mode
        else:
            act = dist.rsample()
        log_prob = dist.log_prob(act).unsqueeze(-1)
        # apply correction for Tanh squashing when computing logprob from Gaussian
        # You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
        # in appendix C to get some understanding of this equation.
        squashed_action = torch.tanh(act)
        log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) + self.__eps).sum(
            -1,
            keepdim=True,
        )
        return Batch(logits=loc_scale, act=squashed_action, state=h, dist=dist, log_prob=log_prob)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # obs_next: s_{t+n}
        obs_next_result = self(obs_next_batch)
        a_ = obs_next_result.act
        sample_ensemble_idx = np.random.choice(self.ensemble_size, self.subset_size, replace=False)
        qs = self.critic_old(obs_next_batch.obs, a_)[sample_ensemble_idx, ...]
        if self.target_mode == "min":
            target_q, _ = torch.min(qs, dim=0)
        elif self.target_mode == "mean":
            target_q = torch.mean(qs, dim=0)

        target_q -= self.alpha * obs_next_result.log_prob

        return target_q

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TREDQTrainingStats:  # type: ignore
        # critic ensemble
        weight = getattr(batch, "weight", 1.0)
        current_qs = self.critic(batch.obs, batch.act).flatten(1)
        target_q = batch.returns.flatten()
        td = current_qs - target_q
        critic_loss = (td.pow(2) * weight).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        batch.weight = torch.mean(td, dim=0)  # prio-buffer
        self.critic_gradient_step += 1

        alpha_loss = None
        # actor
        if self.critic_gradient_step % self.actor_delay == 0:
            obs_result = self(batch)
            a = obs_result.act
            current_qa = self.critic(batch.obs, a).mean(dim=0).flatten()
            actor_loss = (self.alpha * obs_result.log_prob.flatten() - current_qa).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            if self.is_auto_alpha:
                log_prob = obs_result.log_prob.detach() + self._target_entropy
                alpha_loss = -(self._log_alpha * log_prob).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.detach().exp()

        self.sync_weight()

        if self.critic_gradient_step % self.actor_delay == 0:
            self._last_actor_loss = actor_loss.item()
        if self.is_auto_alpha:
            self.alpha = cast(torch.Tensor, self.alpha)

        return REDQTrainingStats(  # type: ignore[return-value]
            actor_loss=self._last_actor_loss,
            critic_loss=critic_loss.item(),
            alpha=self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
            alpha_loss=alpha_loss,
        )
