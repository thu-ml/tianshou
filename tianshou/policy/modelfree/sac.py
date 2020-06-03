import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Optional

from tianshou.policy import DDPGPolicy
from tianshou.policy.dist import DiagGaussian
from tianshou.data import Batch, to_torch_as, ReplayBuffer


class SACPolicy(DDPGPolicy):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param float exploration_noise: the noise intensity, add to the action,
        defaults to 0.1.
    :param float alpha: entropy regularization coefficient, default to 0.2.
    :param action_range: the action range (minimum, maximum).
    :type action_range: (float, float)
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to ``False``.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self,
                 actor: torch.nn.Module,
                 actor_optim: torch.optim.Optimizer,
                 critic1: torch.nn.Module,
                 critic1_optim: torch.optim.Optimizer,
                 critic2: torch.nn.Module,
                 critic2_optim: torch.optim.Optimizer,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 alpha: float = 0.2,
                 action_range: Optional[Tuple[float, float]] = None,
                 reward_normalization: bool = False,
                 ignore_done: bool = False,
                 estimation_step: int = 1,
                 **kwargs) -> None:
        super().__init__(None, None, None, None, tau, gamma, 0,
                         action_range, reward_normalization, ignore_done,
                         estimation_step, **kwargs)
        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self._alpha = alpha
        self.__eps = np.finfo(np.float32).eps.item()

    def train(self) -> None:
        self.training = True
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.training = False
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def sync_weight(self) -> None:
        for o, n in zip(
                self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                input: str = 'obs', **kwargs) -> Batch:
        obs = getattr(batch, input)
        logits, h = self.actor(obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)
        dist = DiagGaussian(*logits)
        x = dist.rsample()
        y = torch.tanh(x)
        act = y * self._action_scale + self._action_bias
        log_prob = dist.log_prob(x) - torch.log(
            self._action_scale * (1 - y.pow(2)) + self.__eps)
        act = act.clamp(self._range[0], self._range[1])
        return Batch(
            logits=logits, act=act, state=h, dist=dist, log_prob=log_prob)

    def _target_q(self, buffer: ReplayBuffer,
                  indice: np.ndarray) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs: s_{t+n}
        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next')
            a_ = obs_next_result.act
            batch.act = to_torch_as(batch.act, a_)
            target_q = torch.min(
                self.critic1_old(batch.obs_next, a_),
                self.critic2_old(batch.obs_next, a_),
            ) - self._alpha * obs_next_result.log_prob
        return target_q

    def learn(self, batch: Batch, **kwargs) -> Dict[str, float]:
        # critic 1
        current_q1 = self.critic1(batch.obs, batch.act)
        target_q = to_torch_as(batch.returns, current_q1)[:, None]
        critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        # critic 2
        current_q2 = self.critic2(batch.obs, batch.act)
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        # actor
        obs_result = self(batch)
        a = obs_result.act
        current_q1a = self.critic1(batch.obs, a)
        current_q2a = self.critic2(batch.obs, a)
        actor_loss = (self._alpha * obs_result.log_prob - torch.min(
            current_q1a, current_q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {
            'loss/actor': actor_loss.item(),
            'loss/critic1': critic1_loss.item(),
            'loss/critic2': critic2_loss.item(),
        }
