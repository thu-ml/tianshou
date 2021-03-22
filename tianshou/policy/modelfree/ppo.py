import torch
import numpy as np
from torch import nn
from typing import Any, Dict, List, Type, Union, Optional

from tianshou.policy import PGPolicy
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as


class PPOPolicy(PGPolicy):
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float max_grad_norm: clipping gradients in back propagation.
        Default to None.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper. Default to 0.2.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage
        Estimation. Default to 0.95.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound.
        Default to 5.0 (set None if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553 Sec. 4.1.
        Default to True.
    :param bool reward_normalization: normalize the returns to Normal(0, 1).
        Default to True.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the
        model; should be as large as possible within the memory constraint.
        Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        max_grad_norm: Optional[float] = None,
        eps_clip: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        gae_lambda: float = 0.95,
        dual_clip: Optional[float] = None,
        value_clip: bool = True,
        max_batchsize: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, optim, dist_fn, **kwargs)
        self._max_grad_norm = max_grad_norm
        self._eps_clip = eps_clip
        self._weight_vf = vf_coef
        self._weight_ent = ent_coef
        self.critic = critic
        self._batch = max_batchsize
        assert 0.0 <= gae_lambda <= 1.0, "GAE lambda should be in [0, 1]."
        self._lambda = gae_lambda
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        if self._rew_norm:
            mean, std = batch.rew.mean(), batch.rew.std()
            if not np.isclose(std, 0.0, 1e-2):
                batch.rew = (batch.rew - mean) / std
        v_s, v_s_, old_log_prob = [], [], []
        with torch.no_grad():
            for b in batch.split(self._batch, shuffle=False, merge_last=True):
                v_s_.append(self.critic(b.obs_next))
                v_s.append(self.critic(b.obs))
                old_log_prob.append(self(b).dist.log_prob(to_torch_as(b.act, v_s[0])))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s_ = to_numpy(torch.cat(v_s_, dim=0).flatten())
        v_s = to_numpy(batch.v_s)
        if self._rew_norm:
            # unnormalize v_s_ & v_s
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps) + self.ret_rms.mean
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps) + self.ret_rms.mean
        un_normalized_returns, advantages = self.compute_episodic_return(
            batch, buffer, indice, v_s_, v_s, gamma=self._gamma, gae_lambda=self._lambda)
        if self._rew_norm:
            batch.returns = (un_normalized_returns - self.ret_rms.mean) / \
                                        np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(un_normalized_returns)
        else:
            batch.returns = un_normalized_returns
        batch.act = to_torch_as(batch.act, batch.v_s[0])
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        batch.returns = to_torch_as(batch.returns, batch.v_s[0])
        batch.adv = to_torch_as(advantages, batch.v_s[0])
        if self._rew_norm:
            mean, std = batch.adv.mean(), batch.adv.std()
            if not np.isclose(std.item(), 0.0, 1e-2):
                batch.adv = (batch.adv - mean) / std
        return batch

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for _ in range(repeat):
            for b in batch.split(batch_size, merge_last=True):
                dist = self(b).dist
                value = self.critic(b.obs).flatten()
                ratio = (dist.log_prob(b.act) - b.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * b.adv
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * b.adv
                if self._dual_clip:
                    clip_loss = -torch.max(
                        torch.min(surr1, surr2), self._dual_clip * b.adv
                    ).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                clip_losses.append(clip_loss.item())
                if self._value_clip:
                    v_clip = b.v_s + (value - b.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (b.returns - value).pow(2)
                    vf2 = (b.returns - v_clip).pow(2)
                    vf_loss = 0.5 * torch.max(vf1, vf2).mean()
                else:
                    vf_loss = 0.5 * (b.returns - value).pow(2).mean()
                vf_losses.append(vf_loss.item())
                e_loss = dist.entropy().mean()
                ent_losses.append(e_loss.item())
                loss = clip_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * e_loss
                losses.append(loss.item())
                self.optim.zero_grad()
                loss.backward()
                if self._max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self._max_grad_norm)
                self.optim.step()
        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }
