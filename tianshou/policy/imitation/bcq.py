import copy
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, to_torch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import BasePolicy
from tianshou.utils.net.continuous import VAE


class BCQPolicy(BasePolicy):
    """Implementation of BCQ algorithm. arXiv:1812.02900.

    :param Perturbation actor: the actor perturbation. (s, a -> perturbed a)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param VAE vae: the VAE network, generating actions similar
        to those in batch. (s, a -> generated a)
    :param torch.optim.Optimizer vae_optim: the optimizer for the VAE network.
    :param Union[str, torch.device] device: which device to create this model on.
        Default to "cpu".
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param float tau: param for soft update of the target network.
        Default to 0.005.
    :param float lmbda: param for Clipped Double Q-learning. Default to 0.75.
    :param int forward_sampled_times: the number of sampled actions in forward
        function. The policy samples many actions and takes the action with the
        max value. Default to 100.
    :param int num_sampled_action: the number of sampled actions in calculating
        target Q. The algorithm samples several actions using VAE, and perturbs
        each action to get the target Q. Default to 10.
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
        vae: VAE,
        vae_optim: torch.optim.Optimizer,
        device: Union[str, torch.device] = "cpu",
        gamma: float = 0.99,
        tau: float = 0.005,
        lmbda: float = 0.75,
        forward_sampled_times: int = 100,
        num_sampled_action: int = 10,
        **kwargs: Any,
    ) -> None:
        # actor is Perturbation!
        super().__init__(**kwargs)
        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = actor_optim

        self.critic1 = critic1
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optim = critic1_optim

        self.critic2 = critic2
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optim = critic2_optim

        self.vae = vae
        self.vae_optim = vae_optim

        self.gamma = gamma
        self.tau = tau
        self.lmbda = lmbda
        self.device = device
        self.forward_sampled_times = forward_sampled_times
        self.num_sampled_action = num_sampled_action

    def train(self, mode: bool = True) -> "BCQPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def forward(
        self,
        batch: RolloutBatchProtocol,
        state: Optional[Union[dict, BatchProtocol, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data."""
        # There is "obs" in the Batch
        # obs_group: several groups. Each group has a state.
        obs_group: torch.Tensor = to_torch(batch.obs, device=self.device)
        act_group = []
        for obs_orig in obs_group:
            # now obs is (state_dim)
            obs = (obs_orig.reshape(1, -1)).repeat(self.forward_sampled_times, 1)
            # now obs is (forward_sampled_times, state_dim)

            # decode(obs) generates action and actor perturbs it
            act = self.actor(obs, self.vae.decode(obs))
            # now action is (forward_sampled_times, action_dim)
            q1 = self.critic1(obs, act)
            # q1 is (forward_sampled_times, 1)
            max_indice = q1.argmax(0)
            act_group.append(act[max_indice].cpu().data.numpy().flatten())
        act_group = np.array(act_group)
        return Batch(act=act_group)

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        self.soft_update(self.critic1_target, self.critic1, self.tau)
        self.soft_update(self.critic2_target, self.critic2, self.tau)
        self.soft_update(self.actor_target, self.actor, self.tau)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> dict[str, float]:
        # batch: obs, act, rew, done, obs_next. (numpy array)
        # (batch_size, state_dim)
        batch: Batch = to_torch(batch, dtype=torch.float, device=self.device)
        obs, act = batch.obs, batch.act
        batch_size = obs.shape[0]

        # mean, std: (state.shape[0], latent_dim)
        recon, mean, std = self.vae(obs, act)
        recon_loss = F.mse_loss(act, recon)
        # (....) is D_KL( N(mu, sigma) || N(0,1) )
        KL_loss = (-torch.log(std) + (std.pow(2) + mean.pow(2) - 1) / 2).mean()
        vae_loss = recon_loss + KL_loss / 2

        self.vae_optim.zero_grad()
        vae_loss.backward()
        self.vae_optim.step()

        # critic training:
        with torch.no_grad():
            # repeat num_sampled_action times
            obs_next = batch.obs_next.repeat_interleave(self.num_sampled_action, dim=0)
            # now obs_next: (num_sampled_action * batch_size, state_dim)

            # perturbed action generated by VAE
            act_next = self.vae.decode(obs_next)
            # now obs_next: (num_sampled_action * batch_size, action_dim)
            target_Q1 = self.critic1_target(obs_next, act_next)
            target_Q2 = self.critic2_target(obs_next, act_next)

            # Clipped Double Q-learning
            target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1 - self.lmbda) * torch.max(
                target_Q1,
                target_Q2,
            )
            # now target_Q: (num_sampled_action * batch_size, 1)

            # the max value of Q
            target_Q = target_Q.reshape(batch_size, -1).max(dim=1)[0].reshape(-1, 1)
            # now target_Q: (batch_size, 1)

            target_Q = (
                batch.rew.reshape(-1, 1) + (1 - batch.done).reshape(-1, 1) * self.gamma * target_Q
            )

        current_Q1 = self.critic1(obs, act)
        current_Q2 = self.critic2(obs, act)

        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.step()

        sampled_act = self.vae.decode(obs)
        perturbed_act = self.actor(obs, sampled_act)

        # max
        actor_loss = -self.critic1(obs, perturbed_act).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update target network
        self.sync_weight()

        return {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/vae": vae_loss.item(),
        }
