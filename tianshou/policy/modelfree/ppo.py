import torch
import numpy as np
from torch import nn

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
    :param float gae_lambda: in [0, 1], param for Generalized Advantage
        Estimation, defaults to 0.95.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound,
        defaults to 5.0 (set ``None`` if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553 Sec. 4.1,
        defaults to ``True``.
    :param bool reward_normalization: normalize the returns to Normal(0, 1),
        defaults to ``True``.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, actor, critic, optim, dist_fn,
                 discount_factor=0.99, max_grad_norm=.5, eps_clip=.2,
                 vf_coef=.5, ent_coef=.01, action_range=None, gae_lambda=0.95,
                 dual_clip=5., value_clip=True, reward_normalization=True,
                 **kwargs):
        super().__init__(None, None, dist_fn, discount_factor, **kwargs)
        self._max_grad_norm = max_grad_norm
        self._eps_clip = eps_clip
        self._w_vf = vf_coef
        self._w_ent = ent_coef
        self._range = action_range
        self.actor = actor
        self.critic = critic
        self.optim = optim
        self._batch = 64
        assert 0 <= gae_lambda <= 1, 'GAE lambda should be in [0, 1].'
        self._lambda = gae_lambda
        assert dual_clip is None or dual_clip > 1, \
            'Dual-clip PPO parameter should greater than 1.'
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        self._rew_norm = reward_normalization
        self.__eps = np.finfo(np.float32).eps.item()

    def process_fn(self, batch, buffer, indice):
        if self._rew_norm:
            mean, std = batch.rew.mean(), batch.rew.std()
            if std > self.__eps:
                batch.rew = (batch.rew - mean) / std
        if self._lambda in [0, 1]:
            return self.compute_episodic_return(
                batch, None, gamma=self._gamma, gae_lambda=self._lambda)
        v_ = []
        with torch.no_grad():
            for b in batch.split(self._batch, permute=False):
                v_.append(self.critic(b.obs_next))
        v_ = torch.cat(v_, dim=0).cpu().numpy()
        return self.compute_episodic_return(
            batch, v_, gamma=self._gamma, gae_lambda=self._lambda)

    def forward(self, batch, state=None, **kwargs):
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        logits, h = self.actor(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        act = dist.sample()
        if self._range:
            act = act.clamp(self._range[0], self._range[1])
        return Batch(logits=logits, act=act, state=h, dist=dist)

    def learn(self, batch, batch_size=None, repeat=1, **kwargs):
        self._batch = batch_size
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        v = []
        old_log_prob = []
        with torch.no_grad():
            for b in batch.split(batch_size, permute=False):
                v.append(self.critic(b.obs))
                old_log_prob.append(self(b).dist.log_prob(
                    torch.tensor(b.act, device=v[0].device)))
        batch.v = torch.cat(v, dim=0)  # old value
        dev = batch.v.device
        batch.act = torch.tensor(batch.act, dtype=torch.float, device=dev)
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        batch.returns = torch.tensor(
            batch.returns, dtype=torch.float, device=dev
        ).reshape(batch.v.shape)
        if self._rew_norm:
            mean, std = batch.returns.mean(), batch.returns.std()
            if std > self.__eps:
                batch.returns = (batch.returns - mean) / std
        batch.adv = batch.returns - batch.v
        if self._rew_norm:
            mean, std = batch.adv.mean(), batch.adv.std()
            if std > self.__eps:
                batch.adv = (batch.adv - mean) / std
        for _ in range(repeat):
            for b in batch.split(batch_size):
                dist = self(b).dist
                value = self.critic(b.obs)
                ratio = (dist.log_prob(b.act) - b.logp_old).exp().float()
                surr1 = ratio * b.adv
                surr2 = ratio.clamp(
                    1. - self._eps_clip, 1. + self._eps_clip) * b.adv
                if self._dual_clip:
                    clip_loss = -torch.max(torch.min(surr1, surr2),
                                           self._dual_clip * b.adv).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                clip_losses.append(clip_loss.item())
                if self._value_clip:
                    v_clip = b.v + (value - b.v).clamp(
                        -self._eps_clip, self._eps_clip)
                    vf1 = (b.returns - value).pow(2)
                    vf2 = (b.returns - v_clip).pow(2)
                    vf_loss = .5 * torch.max(vf1, vf2).mean()
                else:
                    vf_loss = .5 * (b.returns - value).pow(2).mean()
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
        return {
            'loss': losses,
            'loss/clip': clip_losses,
            'loss/vf': vf_losses,
            'loss/ent': ent_losses,
        }
