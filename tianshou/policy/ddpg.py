import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.policy import BasePolicy


# from tianshou.exploration import OUNoise


class DDPGPolicy(BasePolicy):
    """docstring for DDPGPolicy"""

    def __init__(self, actor, actor_optim, critic, critic_optim,
                 tau=0.005, gamma=0.99, exploration_noise=0.1,
                 action_range=None, reward_normalization=False,
                 ignore_done=False):
        super().__init__()
        if actor is not None:
            self.actor, self.actor_old = actor, deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim = actor_optim
        if critic is not None:
            self.critic, self.critic_old = critic, deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim = critic_optim
        assert 0 < tau <= 1, 'tau should in (0, 1]'
        self._tau = tau
        assert 0 < gamma <= 1, 'gamma should in (0, 1]'
        self._gamma = gamma
        assert 0 <= exploration_noise, 'noise should not be negative'
        self._eps = exploration_noise
        assert action_range is not None
        self._range = action_range
        self._action_bias = (action_range[0] + action_range[1]) / 2
        self._action_scale = (action_range[1] - action_range[0]) / 2
        # it is only a little difference to use rand_normal
        # self.noise = OUNoise()
        self._rm_done = ignore_done
        self._rew_norm = reward_normalization
        self.__eps = np.finfo(np.float32).eps.item()

    def set_eps(self, eps):
        self._eps = eps

    def train(self):
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic.eval()

    def sync_weight(self):
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic_old.parameters(), self.critic.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def process_fn(self, batch, buffer, indice):
        if self._rew_norm:
            bfr = buffer.rew[:len(buffer)]
            mean, std = bfr.mean(), bfr.std()
            if std > self.__eps:
                batch.rew = (batch.rew - mean) / std
        if self._rm_done:
            batch.done = batch.done * 0.
        return batch

    def __call__(self, batch, state=None,
                 model='actor', input='obs', eps=None):
        model = getattr(self, model)
        obs = getattr(batch, input)
        logits, h = model(obs, state=state, info=batch.info)
        logits += self._action_bias
        if eps is None:
            eps = self._eps
        if eps > 0:
            # noise = np.random.normal(0, eps, size=logits.shape)
            # logits += torch.tensor(noise, device=logits.device)
            # noise = self.noise(logits.shape, eps)
            logits += torch.randn(
                size=logits.shape, device=logits.device) * eps
        logits = logits.clamp(self._range[0], self._range[1])
        return Batch(act=logits, state=h)

    def learn(self, batch, batch_size=None, repeat=1):
        target_q = self.critic_old(batch.obs_next, self(
            batch, model='actor_old', input='obs_next', eps=0).act)
        dev = target_q.device
        rew = torch.tensor(batch.rew, dtype=torch.float, device=dev)[:, None]
        done = torch.tensor(batch.done, dtype=torch.float, device=dev)[:, None]
        target_q = (rew + (1. - done) * self._gamma * target_q).detach()
        current_q = self.critic(batch.obs, batch.act)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        actor_loss = -self.critic(batch.obs, self(batch, eps=0).act).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {
            'loss/actor': actor_loss.detach().cpu().numpy(),
            'loss/critic': critic_loss.detach().cpu().numpy(),
        }
