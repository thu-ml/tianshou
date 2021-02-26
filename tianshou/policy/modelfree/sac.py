import torch
import numpy as np
from copy import deepcopy
from torch.distributions import Independent, Normal
from typing import Any, Dict, Tuple, Union, Optional

from tianshou.policy import DDPGPolicy
from tianshou.exploration import BaseNoise
from tianshou.data import Batch, ReplayBuffer


class SACPolicy(DDPGPolicy):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param action_range: the action range (minimum, maximum).
    :type action_range: Tuple[float, float]
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient. Default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param BaseNoise exploration_noise: add a noise to action for exploration.
        Default to None. This is useful when solving hard-exploration problem.
    :param bool deterministic_eval: whether to use deterministic action (mean
        of Gaussian policy) instead of stochastic action sampled by the policy.
        Default to True.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        action_range: Tuple[float, float],
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        exploration_noise: Optional[BaseNoise] = None,
        deterministic_eval: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(None, None, None, None, action_range, tau, gamma,
                         exploration_noise, reward_normalization,
                         estimation_step, **kwargs)
        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim

        self._is_auto_alpha = False
        self._alpha: Union[float, torch.Tensor]
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            assert alpha[1].shape == torch.Size([1]) and alpha[1].requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self._deterministic_eval = deterministic_eval
        self.__eps = np.finfo(np.float32).eps.item()

    def train(self, mode: bool = True) -> "SACPolicy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

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
            x = logits[0]
        else:
            x = dist.rsample()
        y = torch.tanh(x)
        act = y * self._action_scale + self._action_bias
        y = self._action_scale * (1 - y.pow(2)) + self.__eps
        log_prob = dist.log_prob(x).unsqueeze(-1)
        log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)

        return Batch(logits=logits, act=act, state=h, dist=dist, log_prob=log_prob)

    def _target_q(
        self, buffer: ReplayBuffer, indice: np.ndarray
    ) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs: s_{t+n}
        obs_next_result = self(batch, input='obs_next')
        a_ = obs_next_result.act
        target_q = torch.min(
            self.critic1_old(batch.obs_next, a_),
            self.critic2_old(batch.obs_next, a_),
        ) - self._alpha * obs_next_result.log_prob
        return target_q

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        weight = batch.pop("weight", 1.0)

        # critic 1
        current_q1 = self.critic1(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td1 = current_q1 - target_q
        critic1_loss = (td1.pow(2) * weight).mean()
        # critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        # critic 2
        current_q2 = self.critic2(batch.obs, batch.act).flatten()
        td2 = current_q2 - target_q
        critic2_loss = (td2.pow(2) * weight).mean()
        # critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        obs_result = self(batch)
        a = obs_result.act
        current_q1a = self.critic1(batch.obs, a).flatten()
        current_q2a = self.critic2(batch.obs, a).flatten()
        actor_loss = (self._alpha * obs_result.log_prob.flatten()
                      - torch.min(current_q1a, current_q2a)).mean()
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

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore

        return result
