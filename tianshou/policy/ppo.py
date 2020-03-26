import torch
import numpy as np
from torch import nn
from copy import deepcopy
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.policy import PGPolicy


class PPOPolicy(PGPolicy):
    """docstring for PPOPolicy"""

    def __init__(self, actor, critic, optim, dist_fn,
                 discount_factor=0.99,
                 max_grad_norm=.5,
                 eps_clip=.2,
                 vf_coef=.5,
                 ent_coef=.0,
                 action_range=None):
        super().__init__(None, None, dist_fn, discount_factor)
        self._max_grad_norm = max_grad_norm
        self._eps_clip = eps_clip
        self._w_vf = vf_coef
        self._w_ent = ent_coef
        self._range = action_range
        self.actor, self.actor_old = actor, deepcopy(actor)
        self.actor_old.eval()
        self.critic, self.critic_old = critic, deepcopy(critic)
        self.critic_old.eval()
        self.optim = optim

    def train(self):
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic.eval()

    def __call__(self, batch, state=None, model='actor'):
        model = getattr(self, model)
        logits, h = model(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        act = dist.sample()
        if self._range:
            act = act.clamp(self._range[0], self._range[1])
        return Batch(logits=logits, act=act, state=h, dist=dist)

    def sync_weight(self):
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def learn(self, batch, batch_size=None, repeat=1):
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        r = batch.returns
        batch.returns = (r - r.mean()) / (r.std() + self._eps)
        batch.act = torch.tensor(batch.act)
        batch.returns = torch.tensor(batch.returns)[:, None]
        for _ in range(repeat):
            for b in batch.split(batch_size):
                vs_old, vs__old = self.critic_old(np.concatenate([
                    b.obs, b.obs_next])).split(b.obs.shape[0])
                dist = self(b).dist
                dist_old = self(b, model='actor_old').dist
                target_v = b.returns.to(vs__old.device) + self._gamma * vs__old
                adv = (target_v - vs_old).detach()
                a = b.act.to(adv.device)
                ratio = torch.exp(dist.log_prob(a) - dist_old.log_prob(a))
                surr1 = ratio * adv
                surr2 = ratio.clamp(
                    1. - self._eps_clip, 1. + self._eps_clip) * adv
                clip_loss = -torch.min(surr1, surr2).mean()
                clip_losses.append(clip_loss.detach().cpu().numpy())
                vf_loss = F.smooth_l1_loss(self.critic(b.obs), target_v)
                vf_losses.append(vf_loss.detach().cpu().numpy())
                e_loss = dist.entropy().mean()
                ent_losses.append(e_loss.detach().cpu().numpy())
                loss = clip_loss + self._w_vf * vf_loss - self._w_ent * e_loss
                losses.append(loss.detach().cpu().numpy())
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(
                    self.actor.parameters()) + list(self.critic.parameters()),
                    self._max_grad_norm)
                self.optim.step()
        self.sync_weight()
        return {
            'loss': losses,
            'loss/clip': clip_losses,
            'loss/vf': vf_losses,
            'loss/ent': ent_losses,
        }
