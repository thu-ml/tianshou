import copy
from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, to_torch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.utils.net.continuous import VAE
from tianshou.utils.optim import clone_optimizer


@dataclass(kw_only=True)
class BCQTrainingStats(TrainingStats):
    actor_loss: float
    critic1_loss: float
    critic2_loss: float
    vae_loss: float


TBCQTrainingStats = TypeVar("TBCQTrainingStats", bound=BCQTrainingStats)


class BCQPolicy(BasePolicy[TBCQTrainingStats], Generic[TBCQTrainingStats]):
    """Implementation of BCQ algorithm. arXiv:1812.02900.

    :param actor_perturbation: the actor perturbation. `(s, a -> perturbed a)`
    :param actor_perturbation_optim: the optimizer for actor network.
    :param critic: the first critic network.
    :param critic_optim: the optimizer for the first critic network.
    :param critic2: the second critic network.
    :param critic2_optim: the optimizer for the second critic network.
    :param vae: the VAE network, generating actions similar to those in batch.
    :param vae_optim: the optimizer for the VAE network.
    :param device: which device to create this model on.
    :param gamma: discount factor, in [0, 1].
    :param tau: param for soft update of the target network.
    :param lmbda: param for Clipped Double Q-learning.
    :param forward_sampled_times: the number of sampled actions in forward function.
        The policy samples many actions and takes the action with the max value.
    :param num_sampled_action: the number of sampled actions in calculating target Q.
        The algorithm samples several actions using VAE, and perturbs each action to get the target Q.
    :param observation_space: Env's observation space.
    :param action_scaling: if True, scale the action from [-1, 1] to the range
        of action_space. Only used if the action_space is continuous.
    :param action_bound_method: method to bound action to range [-1, 1].
        Only used if the action_space is continuous.
    :param lr_scheduler: if not None, will be called in `policy.update()`.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed explanation.
    """

    def __init__(
        self,
        *,
        actor_perturbation: torch.nn.Module,
        actor_perturbation_optim: torch.optim.Optimizer,
        critic: torch.nn.Module,
        critic_optim: torch.optim.Optimizer,
        action_space: gym.Space,
        vae: VAE,
        vae_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module | None = None,
        critic2_optim: torch.optim.Optimizer | None = None,
        # TODO: remove? Many policies don't use this
        device: str | torch.device = "cpu",
        gamma: float = 0.99,
        tau: float = 0.005,
        lmbda: float = 0.75,
        forward_sampled_times: int = 100,
        num_sampled_action: int = 10,
        observation_space: gym.Space | None = None,
        action_scaling: bool = False,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        # actor is Perturbation!
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        self.actor_perturbation = actor_perturbation
        self.actor_perturbation_target = copy.deepcopy(self.actor_perturbation)
        self.actor_perturbation_optim = actor_perturbation_optim

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = critic_optim

        critic2 = critic2 or copy.deepcopy(critic)
        critic2_optim = critic2_optim or clone_optimizer(critic_optim, critic2.parameters())
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

    def train(self, mode: bool = True) -> Self:
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor_perturbation.train(mode)
        self.critic.train(mode)
        self.critic2.train(mode)
        return self

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:
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
            act = self.actor_perturbation(obs, self.vae.decode(obs))
            # now action is (forward_sampled_times, action_dim)
            q1 = self.critic(obs, act)
            # q1 is (forward_sampled_times, 1)
            max_indice = q1.argmax(0)
            act_group.append(act[max_indice].cpu().data.numpy().flatten())
        act_group = np.array(act_group)
        return cast(ActBatchProtocol, Batch(act=act_group))

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        self.soft_update(self.critic_target, self.critic, self.tau)
        self.soft_update(self.critic2_target, self.critic2, self.tau)
        self.soft_update(self.actor_perturbation_target, self.actor_perturbation, self.tau)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TBCQTrainingStats:
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
            target_Q1 = self.critic_target(obs_next, act_next)
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

        current_Q1 = self.critic(obs, act)
        current_Q2 = self.critic2(obs, act)

        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        self.critic_optim.zero_grad()
        self.critic2_optim.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic_optim.step()
        self.critic2_optim.step()

        sampled_act = self.vae.decode(obs)
        perturbed_act = self.actor_perturbation(obs, sampled_act)

        # max
        actor_loss = -self.critic(obs, perturbed_act).mean()

        self.actor_perturbation_optim.zero_grad()
        actor_loss.backward()
        self.actor_perturbation_optim.step()

        # update target network
        self.sync_weight()

        return BCQTrainingStats(  # type: ignore
            actor_loss=actor_loss.item(),
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
            vae_loss=vae_loss.item(),
        )
