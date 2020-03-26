import torch
from torch import nn
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.policy import PGPolicy


class A2CPolicy(PGPolicy):
    """docstring for A2CPolicy"""

    def __init__(self, actor, critic, optim,
                 dist_fn=torch.distributions.Categorical,
                 discount_factor=0.99, vf_coef=.5, ent_coef=.01,
                 max_grad_norm=None):
        super().__init__(None, optim, dist_fn, discount_factor)
        self.actor = actor
        self.critic = critic
        self._w_vf = vf_coef
        self._w_ent = ent_coef
        self._grad_norm = max_grad_norm

    def __call__(self, batch, state=None):
        logits, h = self.actor(batch.obs, state=state, info=batch.info)
        dist = self.dist_fn(logits)
        act = dist.sample()
        return Batch(logits=logits, act=act, state=h, dist=dist)

    def learn(self, batch, batch_size=None, repeat=1):
        losses, actor_losses, vf_losses, ent_losses = [], [], [], []
        for _ in range(repeat):
            for b in batch.split(batch_size):
                self.optim.zero_grad()
                result = self(b)
                dist = result.dist
                v = self.critic(b.obs)
                a = torch.tensor(b.act, device=dist.logits.device)
                r = torch.tensor(b.returns, device=dist.logits.device)
                a_loss = -(dist.log_prob(a) * (r - v).detach()).mean()
                vf_loss = F.mse_loss(r[:, None], v)
                ent_loss = dist.entropy().mean()
                loss = a_loss + self._w_vf * vf_loss - self._w_ent * ent_loss
                loss.backward()
                if self._grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self._grad_norm)
                self.optim.step()
                actor_losses.append(a_loss.detach().cpu().numpy())
                vf_losses.append(vf_loss.detach().cpu().numpy())
                ent_losses.append(ent_loss.detach().cpu().numpy())
                losses.append(loss.detach().cpu().numpy())
        return {
            'loss': losses,
            'loss/actor': actor_losses,
            'loss/vf': vf_losses,
            'loss/ent': ent_losses,
        }
