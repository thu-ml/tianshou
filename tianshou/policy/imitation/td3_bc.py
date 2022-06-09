from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from tianshou.data import Batch, to_torch_as
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.policy import TD3Policy


class TD3BCPolicy(TD3Policy):
    """Implementation of TD3+BC. arXiv:2106.06860.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param float exploration_noise: the exploration noise, add to the action.
        Default to ``GaussianNoise(sigma=0.1)``
    :param float policy_noise: the noise used in updating policy network.
        Default to 0.2.
    :param int update_actor_freq: the update frequency of actor network.
        Default to 2.
    :param float noise_clip: the clipping range used in updating policy network.
        Default to 0.5.
    :param float alpha: the value of alpha, which controls the weight for TD3 learning
        relative to behavior cloning.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

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
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        policy_noise: float = 0.2,
        update_actor_freq: int = 2,
        noise_clip: float = 0.5,
        alpha: float = 2.5,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim, tau,
            gamma, exploration_noise, policy_noise, update_actor_freq, noise_clip,
            reward_normalization, estimation_step, **kwargs
        )
        self._alpha = alpha

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # critic 1&2
        td1, critic1_loss = self._mse_optimizer(
            batch, self.critic1, self.critic1_optim
        )
        td2, critic2_loss = self._mse_optimizer(
            batch, self.critic2, self.critic2_optim
        )
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        if self._cnt % self._freq == 0:
            act = self(batch, eps=0.0).act
            q_value = self.critic1(batch.obs, act)
            lmbda = self._alpha / q_value.abs().mean().detach()
            actor_loss = -lmbda * q_value.mean() + F.mse_loss(
                act, to_torch_as(batch.act, act)
            )
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self._last = actor_loss.item()
            self.actor_optim.step()
            self.sync_weight()
        self._cnt += 1
        return {
            "loss/actor": self._last,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
