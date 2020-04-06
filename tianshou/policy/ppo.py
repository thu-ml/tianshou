import torch
import numpy as np
from torch import nn
from copy import deepcopy
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.policy import PGPolicy


class PPOPolicy(PGPolicy):
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic
        network.
    :param torch.distributions.Distribution dist_fn: for computing the action.
    :param float discount_factor: in [0, 1], defaults to 0.99.
    :param float max_grad_norm: clipping gradients in back propagation,
        defaults to ``None``.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper, defaults to 0.2.
    :param float vf_coef: weight for value loss, defaults to 0.5.
    :param float ent_coef: weight for entropy loss, defaults to 0.01.
    :param action_range: the action range (minimum, maximum).
    :type action_range: [float, float]
    """

    def __init__(self, actor, critic, optim, dist_fn,
                 discount_factor=0.99,
                 max_grad_norm=.5,
                 eps_clip=.2,
                 vf_coef=.5,
                 ent_coef=.0,
                 action_range=None,
                 **kwargs):
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
        """Set the module in training mode, except for the target network."""
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        """Set the module in evaluation mode, except for the target network."""
        self.training = False
        self.actor.eval()
        self.critic.eval()

    def __call__(self, batch, state=None, model='actor', **kwargs):
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        More information can be found at
        :meth:`~tianshou.policy.BasePolicy.__call__`.
        """
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
        """Synchronize the weight for the target network."""
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def learn(self, batch, batch_size=None, repeat=1, **kwargs):
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
                clip_losses.append(clip_loss.item())
                vf_loss = F.smooth_l1_loss(self.critic(b.obs), target_v)
                vf_losses.append(vf_loss.item())
                e_loss = dist.entropy().mean()
                ent_losses.append(e_loss.item())
                loss = clip_loss + self._w_vf * vf_loss - self._w_ent * e_loss
                losses.append(loss.item())
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
