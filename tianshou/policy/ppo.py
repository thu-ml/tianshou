import torch
from torch import nn
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.policy import PGPolicy


class PPOPolicy(PGPolicy):
    """docstring for PPOPolicy"""

    def __init__(self, actor, actor_optim,
                 critic, critic_optim,
                 dist_fn=torch.distributions.Categorical,
                 discount_factor=0.99, vf_coef=.5, entropy_coef=.01,
                 eps_clip=None):
        super().__init__(None, None, dist_fn, discount_factor)
        self._w_value = vf_coef
        self._w_entropy = entropy_coef
        self._eps_clip = eps_clip

    def __call__(self, batch, state=None):
        logits, value, h = self.model(batch.obs, state=state, info=batch.info)
        logits = F.softmax(logits, dim=1)
        dist = self.dist_fn(logits)
        act = dist.sample()
        return Batch(logits=logits, act=act, state=h, dist=dist, value=value)

    def learn(self, batch, batch_size=None):
        losses = []
        for b in batch.split(batch_size):
            self.optim.zero_grad()
            result = self(b)
            dist = result.dist
            v = result.value
            a = torch.tensor(b.act, device=dist.logits.device)
            r = torch.tensor(b.returns, device=dist.logits.device)
            actor_loss = -(dist.log_prob(a) * (r - v).detach()).mean()
            critic_loss = F.mse_loss(r[:, None], v)
            entropy_loss = dist.entropy().mean()
            loss = actor_loss \
                + self._w_value * critic_loss \
                - self._w_entropy * entropy_loss
            loss.backward()
            if self._grad_norm:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self._grad_norm)
            self.optim.step()
            losses.append(loss.detach().cpu().numpy())
        return losses
