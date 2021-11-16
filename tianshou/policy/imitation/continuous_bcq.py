from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.policy import BasePolicy
import copy


class Perturbation(nn.Module):
    def __init__(
            self,
            preprocess_net: nn.Module,
            max_action,
            device: Union[str, int, torch.device] = "cpu",
            phi: float = 0.05
    ):
        # preprocess_net: input_dim=state_dim+action_dim, output_dim=action_dim
        super(Perturbation, self).__init__()
        self.preprocess_net = preprocess_net
        self.device = device
        self.max_action = max_action
        self.phi = phi

    def forward(self, state, action):
        # preprocess_net
        logits = self.preprocess_net(torch.cat([state, action], 1))[0]
        a = self.phi * self.max_action * torch.tanh(logits)  # TODO
        # clip to [-max_action, max_action]
        return (a + action).clamp(-self.max_action, self.max_action)


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 hidden_dim, latent_dim, max_action, device):
        """
        encoder: input_dim=state_dim+action_dim, the last layer is ReLU
        decoder: input_dim=state_dim+action_dim, output_dim=action_dim
        for example:
        vae_encoder = MLP(
            input_dim=args.state_dim + args.action_dim,
            hidden_sizes=args.vae_hidden_sizes,
            device=args.device
        )
        args.latent_dim = args.action_dim * 2
        vae_decoder = MLP(
            input_dim=args.state_dim + args.latent_dim,
            output_dim=args.action_dim,
            hidden_sizes=args.vae_hidden_sizes,
            device=args.device
        )
        """
        # hidden_dim = the size of the last linear-layer in encoder
        # latent_dim = 2 * action_dim
        super(VAE, self).__init__()
        self.encoder = encoder

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.decoder = decoder

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        # [state, action] -> z , [state, z] -> action
        z = self.encoder(torch.cat([state, action], 1))
        # shape of z: (state.shape[0], hidden_dim=750)

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)  # in [1.8e-2, 3.3e6]
        # shape of mean, std: (state.shape[0], latent_dim)

        z = mean + std * torch.randn_like(std)  # (state.shape[0], latent_dim)

        u = self.decode(state, z)  # (state.shape[0], action_dim)
        return u, mean, std

    def decode(self, state, z=None):
        """
        decode(state) -> action
        """
        if z is None:
            # state.shape[0] may be batch_size
            # latent vector clipped to [-0.5, 0.5]
            z = torch.randn((state.shape[0], self.latent_dim))\
                .to(self.device).clamp(-0.5, 0.5)

        # decode z with state!
        return self.max_action * torch.tanh(self.decoder(torch.cat([state, z], 1)))


class ContinuousBCQPolicy(BasePolicy):  # TODO: 可能要改成Object
    """Implementation of continuous BCQ algorithm. arXiv:1812.02900.
        :param torch.nn.Module actor: the actor perturbation (s, a -> perturbed a)
        :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
        :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
        :param torch.optim.Optimizer critic1_optim: the optimizer for the first
            critic network.
        :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
        :param torch.optim.Optimizer critic2_optim: the optimizer for the second
            critic network.
        :param torch.nn.Module vae: the vae network, generating actions similar
            to those in batch. (s, a -> generated a)
        :param torch.optim.Optimizer vae_optim: the optimizer for the vae network.
        :param float gamma: discount factor, in [0, 1]. Default to 0.99.
        :param float tau: param for soft update of the target network.
            Default to 0.005.
        :param float lmbda: param for Clipped Double Q-learning. Default to 0.75.
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
        vae: torch.nn.Module,
        vae_optim: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        lmbda: float = 0.75,
        **kwargs: Any
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
        self.device = vae.device

        # assert target_update_freq > 0, "BCQ needs target network setting."

    def train(self, mode: bool = True) -> "ContinuousBCQPolicy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        # state: None, input: "obs"
        # There is "obs" in the Batch
        # obs: 10 groups. Each group has a state. shape: (10, state_dim)
        obs = batch["obs"]

        act = []
        with torch.no_grad():
            for state in obs:
                # now state is (state_dim)
                state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1)\
                    .to(self.device)
                # now state is (100, state_dim)

                # decode(state) generates action and actor perturbs it
                action = self.actor(state, self.vae.decode(state))
                # now action is (100, action_dim)
                q1 = self.critic1(state, action)
                # q1 is (100, 1)
                ind = q1.argmax(0)
                act.append(action[ind].cpu().data.numpy().flatten())
        act = np.array(act)
        return Batch(act=act)

    def sync_weight(self):
        for param, target_param in \
                zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in \
                zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in \
                zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # batch: obs, act, rew, done, obs_next. (numpy array)
        # (batch_size, state_dim)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        # (batch_size, action_dim)
        act = torch.FloatTensor(batch["act"]).to(self.device)
        # (batch_size)
        rew = torch.FloatTensor(batch["rew"]).to(self.device)
        # (batch_size)
        done = torch.IntTensor(batch["done"]).to(self.device)
        # (batch_size, state_dim)
        obs_next = torch.FloatTensor(batch["obs_next"]).to(self.device)

        batch_size = obs.shape[0]

        # mean, std: (state.shape[0], latent_dim)
        recon, mean, std = self.vae(obs, act)
        recon_loss = F.mse_loss(act, recon)
        # (....) is D_KL( N(mu, sigma) || N(0,1) )
        KL_loss = (- torch.log(std) + (std.pow(2) + mean.pow(2) - 1) / 2).mean()
        vae_loss = recon_loss + KL_loss / 2

        self.vae_optim.zero_grad()
        vae_loss.backward()
        self.vae_optim.step()

        # critic training:
        with torch.no_grad():
            obs_next = obs_next.repeat_interleave(10, dim=0)  # repeat 10 times
            # now obs_next: (10 * batch_size, state_dim)

            # perturbed action generated by VAE
            act_next = self.vae.decode(obs_next)
            # now obs_next: (10 * batch_size, action_dim)
            target_Q1 = self.critic1_target(obs_next, act_next)
            target_Q2 = self.critic2_target(obs_next, act_next)

            # Clipped Double Q-learning
            target_Q = \
                self.lmbda * torch.min(target_Q1, target_Q2) + \
                (1 - self.lmbda) * torch.max(target_Q1, target_Q2)
            # now target_Q: (10 * batch_size, 1)

            # max: [values, indeices]
            target_Q = target_Q.reshape(batch_size, -1).max(dim=1)[0].reshape(-1, 1)
            # now target_Q: (batch_size, 1)

            target_Q = \
                rew.reshape(-1, 1) + \
                (1 - done).reshape(-1, 1) * self.gamma * target_Q

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

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/vae": vae_loss.item(),
        }
        return result
