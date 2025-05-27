import copy
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from tianshou.algorithm.algorithm_base import (
    LaggedNetworkPolyakUpdateAlgorithmMixin,
    OfflineAlgorithm,
    Policy,
    TrainingStats,
)
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import Batch, to_torch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.utils.net.continuous import VAE


@dataclass(kw_only=True)
class BCQTrainingStats(TrainingStats):
    actor_loss: float
    critic1_loss: float
    critic2_loss: float
    vae_loss: float


TBCQTrainingStats = TypeVar("TBCQTrainingStats", bound=BCQTrainingStats)


class BCQPolicy(Policy):
    def __init__(
        self,
        *,
        actor_perturbation: torch.nn.Module,
        action_space: gym.Space,
        critic: torch.nn.Module,
        vae: VAE,
        forward_sampled_times: int = 100,
        observation_space: gym.Space | None = None,
        action_scaling: bool = False,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
    ) -> None:
        """
        :param actor_perturbation: the actor perturbation. `(s, a -> perturbed a)`
        :param critic: the first critic network.
        :param vae: the VAE network, generating actions similar to those in batch.
        :param forward_sampled_times: the number of sampled actions in forward function.
            The policy samples many actions and takes the action with the max value.
        :param observation_space: the environment's observation space
        :param action_scaling: flag indicating whether, for continuous action spaces, actions
            should be scaled from the standard neural network output range [-1, 1] to the
            environment's action space range [action_space.low, action_space.high].
            This applies to continuous action spaces only (gym.spaces.Box) and has no effect
            for discrete spaces.
            When enabled, policy outputs are expected to be in the normalized range [-1, 1]
            (after bounding), and are then linearly transformed to the actual required range.
            This improves neural network training stability, allows the same algorithm to work
            across environments with different action ranges, and standardizes exploration
            strategies.
            Should be disabled if the actor model already produces outputs in the correct range.
        :param action_bound_method: the method used for bounding actions in continuous action spaces
            to the range [-1, 1] before scaling them to the environment's action space (provided
            that `action_scaling` is enabled).
            This applies to continuous action spaces only (`gym.spaces.Box`) and should be set to None
            for discrete spaces.
            When set to "clip", actions exceeding the [-1, 1] range are simply clipped to this
            range. When set to "tanh", a hyperbolic tangent function is applied, which smoothly
            constrains outputs to [-1, 1] while preserving gradients.
            The choice of bounding method affects both training dynamics and exploration behavior.
            Clipping provides hard boundaries but may create plateau regions in the gradient
            landscape, while tanh provides smoother transitions but can compress sensitivity
            near the boundaries.
            Should be set to None if the actor model inherently produces bounded outputs.
            Typically used together with `action_scaling=True`.
        """
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
        )
        self.actor_perturbation = actor_perturbation
        self.critic = critic
        self.vae = vae
        self.forward_sampled_times = forward_sampled_times

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:
        """Compute action over the given batch data."""
        # There is "obs" in the Batch
        # obs_group: several groups. Each group has a state.
        device = next(self.parameters()).device
        obs_group: torch.Tensor = to_torch(batch.obs, device=device)
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


class BCQ(
    OfflineAlgorithm[BCQPolicy],
    LaggedNetworkPolyakUpdateAlgorithmMixin,
):
    """Implementation of Batch-Constrained Deep Q-learning (BCQ) algorithm. arXiv:1812.02900."""

    def __init__(
        self,
        *,
        policy: BCQPolicy,
        actor_perturbation_optim: OptimizerFactory,
        critic_optim: OptimizerFactory,
        vae_optim: OptimizerFactory,
        critic2: torch.nn.Module | None = None,
        critic2_optim: OptimizerFactory | None = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        lmbda: float = 0.75,
        num_sampled_action: int = 10,
    ) -> None:
        """
        :param policy: the policy
        :param actor_perturbation_optim: the optimizer factory for the policy's actor perturbation network.
        :param critic_optim: the optimizer factory for the policy's critic network.
        :param critic2: the second critic network; if None, clone the critic from the policy
        :param critic2_optim: the optimizer factory for the second critic network; if None, use optimizer factory of first critic
        :param vae_optim: the optimizer factory for the VAE network.
        :param gamma: the discount factor in [0, 1] for future rewards.
            This determines how much future rewards are valued compared to immediate ones.
            Lower values (closer to 0) make the agent focus on immediate rewards, creating "myopic"
            behavior. Higher values (closer to 1) make the agent value long-term rewards more,
            potentially improving performance in tasks where delayed rewards are important but
            increasing training variance by incorporating more environmental stochasticity.
            Typically set between 0.9 and 0.99 for most reinforcement learning tasks
        :param tau: the soft update coefficient for target networks, controlling the rate at which
            target networks track the learned networks.
            When the parameters of the target network are updated with the current (source) network's
            parameters, a weighted average is used: target = tau * source + (1 - tau) * target.
            Smaller values (closer to 0) create more stable but slower learning as target networks
            change more gradually. Higher values (closer to 1) allow faster learning but may reduce
            stability.
            Typically set to a small value (0.001 to 0.01) for most reinforcement learning tasks.
        :param lmbda: param for Clipped Double Q-learning.
        :param num_sampled_action: the number of sampled actions in calculating target Q.
            The algorithm samples several actions using VAE, and perturbs each action to get the target Q.
        """
        # actor is Perturbation!
        super().__init__(
            policy=policy,
        )
        LaggedNetworkPolyakUpdateAlgorithmMixin.__init__(self, tau=tau)
        self.actor_perturbation_target = self._add_lagged_network(self.policy.actor_perturbation)
        self.actor_perturbation_optim = self._create_optimizer(
            self.policy.actor_perturbation, actor_perturbation_optim
        )

        self.critic_target = self._add_lagged_network(self.policy.critic)
        self.critic_optim = self._create_optimizer(self.policy.critic, critic_optim)

        self.critic2 = critic2 or copy.deepcopy(self.policy.critic)
        self.critic2_target = self._add_lagged_network(self.critic2)
        self.critic2_optim = self._create_optimizer(self.critic2, critic2_optim or critic_optim)

        self.vae_optim = self._create_optimizer(self.policy.vae, vae_optim)

        self.gamma = gamma
        self.lmbda = lmbda
        self.num_sampled_action = num_sampled_action

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> BCQTrainingStats:
        # batch: obs, act, rew, done, obs_next. (numpy array)
        # (batch_size, state_dim)
        # TODO: This does not use policy.forward but computes things directly, which seems odd

        device = next(self.parameters()).device
        batch: Batch = to_torch(batch, dtype=torch.float, device=device)
        obs, act = batch.obs, batch.act
        batch_size = obs.shape[0]

        # mean, std: (state.shape[0], latent_dim)
        recon, mean, std = self.policy.vae(obs, act)
        recon_loss = F.mse_loss(act, recon)
        # (....) is D_KL( N(mu, sigma) || N(0,1) )
        KL_loss = (-torch.log(std) + (std.pow(2) + mean.pow(2) - 1) / 2).mean()
        vae_loss = recon_loss + KL_loss / 2

        self.vae_optim.step(vae_loss)

        # critic training:
        with torch.no_grad():
            # repeat num_sampled_action times
            obs_next = batch.obs_next.repeat_interleave(self.num_sampled_action, dim=0)
            # now obs_next: (num_sampled_action * batch_size, state_dim)

            # perturbed action generated by VAE
            act_next = self.policy.vae.decode(obs_next)
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
                batch.rew.reshape(-1, 1)
                + torch.logical_not(batch.done).reshape(-1, 1) * self.gamma * target_Q
            )
            target_Q = target_Q.float()

        current_Q1 = self.policy.critic(obs, act)
        current_Q2 = self.critic2(obs, act)

        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)
        self.critic_optim.step(critic1_loss)
        self.critic2_optim.step(critic2_loss)

        sampled_act = self.policy.vae.decode(obs)
        perturbed_act = self.policy.actor_perturbation(obs, sampled_act)

        # max
        actor_loss = -self.policy.critic(obs, perturbed_act).mean()

        self.actor_perturbation_optim.step(actor_loss)

        # update target networks
        self._update_lagged_network_weights()

        return BCQTrainingStats(
            actor_loss=actor_loss.item(),
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
            vae_loss=vae_loss.item(),
        )
