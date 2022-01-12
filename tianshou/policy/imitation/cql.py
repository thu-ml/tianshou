import copy
from typing import Any, Dict, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tianshou.utils.net.continuous import ActorProb

from tianshou.data import Batch, to_torch
from tianshou.policy import SACPolicy


class CQLPolicy(SACPolicy):
    def __init__(
        self,
        actor: ActorProb,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        cql_log_alpha: torch.Tensor,
        cql_alpha_optim: torch.optim.Optimizer,
        cql_weight: float = 1.0,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        temperature: float = 1.0,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any
    ) -> None:
        super().__init__(
            actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
            tau, gamma, alpha,
            **kwargs
        )
        # There are _is_auto_alpha, _target_entropy, _log_alpha, _alpha_optim in SACPolicy.
        self.device = device
        self.temperature = temperature
        self.with_lagrange = with_lagrange
        self.lagrange_threshold = lagrange_threshold

        self.cql_weight = cql_weight
        self.cql_log_alpha = cql_log_alpha
        self.cql_alpha_optim = cql_alpha_optim

    def train(self, mode: bool = True) -> "CQLPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data."""
        # There is "obs" in the Batch
        obs: torch.Tensor = to_torch(  # type: ignore
            batch.obs, device=self.device
        )
        if self.training:
            (mu, sigma), h = self.actor.forward(obs)
            dist = torch.distributions.Normal(mu, sigma)
            e = dist.rsample().to(self.device)
            act = torch.tanh(e)
        else:
            # eval
            (mu, sigma), h = self.actor.forward(obs)
            act = torch.tanh(mu)
        act = np.array(act)
        return Batch(act=act, state=h)

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        for net, net_target in [
            [self.critic1, self.critic1_target], [self.critic2, self.critic2_target]
        ]:
            for param, target_param in zip(net.parameters(), net_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def calc_actor_loss(self, obs, alpha):
        pass

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # (batch_size, state_dim)
        batch: Batch = to_torch(  # type: ignore
            batch, dtype=torch.float, device=self.device
        )
        obs, act, rew, obs_next = batch.obs, batch.act, batch.rew, batch.obs_next
        batch_size = obs.shape[0]

        # calc actor loss
        # prevent alpha from being modified
        current_alpha = copy.deepcopy(self._alpha)
        actor_loss = self.calc_actor_loss(obs, current_alpha)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
