from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributions import Independent, Normal

from tianshou.data import Batch, ReplayBuffer
from tianshou.exploration import BaseNoise
from tianshou.policy import DDPGPolicy


class REDQPolicy(DDPGPolicy):
    """Implementation of REDQ. arXiv:2101.05982.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critics: critic ensemble networks.
    :param torch.optim.Optimizer critics_optim: the optimizer for the critic networks.
    :param int ensemble_size: Number of sub-networks in the critic ensemble.
        Default to 10.
    :param int subset_size: Number of networks in the subset. Default to 2.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient. Default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param int actor_delay: Number of critic updates before an actor update.
        Default to 20.
    :param BaseNoise exploration_noise: add a noise to action for exploration.
        Default to None. This is useful when solving hard-exploration problem.
    :param bool deterministic_eval: whether to use deterministic action (mean
        of Gaussian policy) instead of stochastic action sampled by the policy.
        Default to True.
    :param str target_mode: methods to integrate critic values in the subset,
        currently support minimum and average. Default to min.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critics: torch.nn.Module,
        critics_optim: torch.optim.Optimizer,
        ensemble_size: int = 10,
        subset_size: int = 2,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        actor_delay: int = 20,
        exploration_noise: Optional[BaseNoise] = None,
        deterministic_eval: bool = True,
        target_mode: str = "min",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            None, None, None, None, tau, gamma, exploration_noise,
            reward_normalization, estimation_step, **kwargs
        )
        self.actor, self.actor_optim = actor, actor_optim
        self.critics, self.critics_old = critics, deepcopy(critics)
        self.critics_old.eval()
        self.critics_optim = critics_optim
        assert 0 < subset_size <= ensemble_size, \
            "Invalid choice of ensemble size or subset size."
        self.ensemble_size = ensemble_size
        self.subset_size = subset_size

        self._is_auto_alpha = False
        self._alpha: Union[float, torch.Tensor]
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            assert alpha[1].shape == torch.Size([1]) and alpha[1].requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        if target_mode in ("min", "mean"):
            self.target_mode = target_mode
        else:
            raise ValueError("Unsupported mode of Q target computing.")

        self.critic_gradient_step = 0
        self.actor_delay = actor_delay
        self._deterministic_eval = deterministic_eval
        self.__eps = np.finfo(np.float32).eps.item()

    def train(self, mode: bool = True) -> "REDQPolicy":
        self.training = mode
        self.actor.train(mode)
        self.critics.train(mode)
        return self

    def sync_weight(self) -> None:
        for o, n in zip(self.critics_old.parameters(), self.critics.parameters()):
            o.data.copy_(o.data * (1.0 - self.tau) + n.data * self.tau)

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        obs = batch[input]
        logits, h = self.actor(obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)
        dist = Independent(Normal(*logits), 1)
        if self._deterministic_eval and not self.training:
            act = logits[0]
        else:
            act = dist.rsample()
        log_prob = dist.log_prob(act).unsqueeze(-1)
        # apply correction for Tanh squashing when computing logprob from Gaussian
        # You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
        # in appendix C to get some understanding of this equation.
        squashed_action = torch.tanh(act)
        log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) +
                                        self.__eps).sum(-1, keepdim=True)
        return Batch(
            logits=logits, act=squashed_action, state=h, dist=dist, log_prob=log_prob
        )

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs: s_{t+n}
        obs_next_result = self(batch, input="obs_next")
        a_ = obs_next_result.act
        sample_ensemble_idx = np.random.choice(
            self.ensemble_size, self.subset_size, replace=False
        )
        qs = self.critics_old(batch.obs_next, a_)[sample_ensemble_idx, ...]
        if self.target_mode == "min":
            target_q, _ = torch.min(qs, dim=0)
        elif self.target_mode == "mean":
            target_q = torch.mean(qs, dim=0)
        target_q -= self._alpha * obs_next_result.log_prob

        return target_q

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # critic ensemble
        weight = getattr(batch, "weight", 1.0)
        current_qs = self.critics(batch.obs, batch.act).flatten(1)
        target_q = batch.returns.flatten()
        td = current_qs - target_q
        critic_loss = (td.pow(2) * weight).mean()
        self.critics_optim.zero_grad()
        critic_loss.backward()
        self.critics_optim.step()
        batch.weight = torch.mean(td, dim=0)  # prio-buffer
        self.critic_gradient_step += 1

        # actor
        if self.critic_gradient_step % self.actor_delay == 0:
            obs_result = self(batch)
            a = obs_result.act
            current_qa = self.critics(batch.obs, a).mean(dim=0).flatten()
            actor_loss = (self._alpha * obs_result.log_prob.flatten() -
                          current_qa).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            if self._is_auto_alpha:
                log_prob = obs_result.log_prob.detach() + self._target_entropy
                alpha_loss = -(self._log_alpha * log_prob).mean()
                self._alpha_optim.zero_grad()
                alpha_loss.backward()
                self._alpha_optim.step()
                self._alpha = self._log_alpha.detach().exp()

        self.sync_weight()

        result = {"loss/critics": critic_loss.item()}
        if self.critic_gradient_step % self.actor_delay == 0:
            result["loss/actor"] = actor_loss.item(),
            if self._is_auto_alpha:
                result["loss/alpha"] = alpha_loss.item()
                result["alpha"] = self._alpha.item()  # type: ignore

        return result
