import torch
import numpy as np
from copy import deepcopy
from typing import Any, Dict, Tuple, Union, Optional

from tianshou.policy import BasePolicy
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.data import Batch, ReplayBuffer, to_torch_as


class DDPGPolicy(BasePolicy):
    """Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic: the critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic_optim: the optimizer for critic
        network.
    :param action_range: the action range (minimum, maximum).
    :type action_range: Tuple[float, float]
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param BaseNoise exploration_noise: the exploration noise,
        add to the action, defaults to ``GaussianNoise(sigma=0.1)``.
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to False.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to False.
    :param int estimation_step: greater than 1, the number of steps to look
        ahead.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: Optional[torch.nn.Module],
        actor_optim: Optional[torch.optim.Optimizer],
        critic: Optional[torch.nn.Module],
        critic_optim: Optional[torch.optim.Optimizer],
        action_range: Tuple[float, float],
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        reward_normalization: bool = False,
        ignore_done: bool = False,
        estimation_step: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if actor is not None and actor_optim is not None:
            self.actor: torch.nn.Module = actor
            self.actor_old = deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim: torch.optim.Optimizer = actor_optim
        if critic is not None and critic_optim is not None:
            self.critic: torch.nn.Module = critic
            self.critic_old = deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim: torch.optim.Optimizer = critic_optim
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self._tau = tau
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
        self._gamma = gamma
        self._noise = exploration_noise
        self._range = action_range
        self._action_bias = (action_range[0] + action_range[1]) / 2.0
        self._action_scale = (action_range[1] - action_range[0]) / 2.0
        # it is only a little difference to use GaussianNoise
        # self.noise = OUNoise()
        self._rm_done = ignore_done
        self._rew_norm = reward_normalization
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self._n_step = estimation_step

    def set_exp_noise(self, noise: Optional[BaseNoise]) -> None:
        """Set the exploration noise."""
        self._noise = noise

    def train(self, mode: bool = True) -> "DDPGPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(
            self.critic_old.parameters(), self.critic.parameters()
        ):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def _target_q(
        self, buffer: ReplayBuffer, indice: np.ndarray
    ) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs_next: s_{t+n}
        with torch.no_grad():
            target_q = self.critic_old(
                batch.obs_next,
                self(batch, model='actor_old', input='obs_next').act)
        return target_q

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        if self._rm_done:
            batch.done = batch.done * 0.0
        batch = self.compute_nstep_return(
            batch, buffer, indice, self._target_q,
            self._gamma, self._n_step, self._rew_norm)
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "actor",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        actions, h = model(obs, state=state, info=batch.info)
        actions += self._action_bias
        if self._noise and not self.updating:
            actions += to_torch_as(self._noise(actions.shape), actions)
        actions = actions.clamp(self._range[0], self._range[1])
        return Batch(act=actions, state=h)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        weight = batch.pop("weight", 1.0)
        current_q = self.critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        critic_loss = (td.pow(2) * weight).mean()
        batch.weight = td  # prio-buffer
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        action = self(batch).act
        actor_loss = -self.critic(batch.obs, action).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
        }
