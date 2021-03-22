import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, List, Type, Optional

from tianshou.policy import PGPolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy


class A2CPolicy(PGPolicy):
    """Implementation of Synchronous Advantage Actor-Critic. arXiv:1602.01783.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic
        network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation.
        Default to None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage
        Estimation. Default to 0.95.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
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
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: Optional[float] = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        **kwargs: Any
    ) -> None:
        super().__init__(actor, optim, dist_fn, **kwargs)
        self.critic = critic
        assert 0.0 <= gae_lambda <= 1.0, "GAE lambda should be in [0, 1]."
        self._lambda = gae_lambda
        self._weight_vf = vf_coef
        self._weight_ent = ent_coef
        self._grad_norm = max_grad_norm
        self._batch = max_batchsize

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        v_s_ = []
        with torch.no_grad():
            for b in batch.split(self._batch, shuffle=False, merge_last=True):
                v_s_.append(to_numpy(self.critic(b.obs_next)))
        v_s_ = np.concatenate(v_s_, axis=0)
        if self._rew_norm:
            # unnormalize v_s_
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps) + self.ret_rms.mean
        un_normalized_returns, _ = self.compute_episodic_return(
            batch, buffer, indice, v_s_, gamma=self._gamma, gae_lambda=self._lambda)
        if self._rew_norm:
            batch.returns = (un_normalized_returns - self.ret_rms.mean) / \
                                        np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(un_normalized_returns)
        else:
            batch.returns = un_normalized_returns
        return batch

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, actor_losses, vf_losses, ent_losses = [], [], [], []
        for _ in range(repeat):
            for b in batch.split(batch_size, merge_last=True):
                self.optim.zero_grad()
                dist = self(b).dist
                v = self.critic(b.obs).flatten()
                a = to_torch_as(b.act, v)
                r = to_torch_as(b.returns, v)
                log_prob = dist.log_prob(a).reshape(len(r), -1).transpose(0, 1)
                a_loss = -(log_prob * (r - v).detach()).mean()
                vf_loss = F.mse_loss(r, v)  # type: ignore
                ent_loss = dist.entropy().mean()
                loss = a_loss + self._weight_vf * vf_loss - self._weight_ent * ent_loss
                loss.backward()
                if self._grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        max_norm=self._grad_norm,
                    )
                self.optim.step()
                actor_losses.append(a_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())
        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "loss": losses,
            "loss/actor": actor_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }
