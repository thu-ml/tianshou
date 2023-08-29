from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import kl_divergence

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.types import BatchWithAdvantagesProtocol, RolloutBatchProtocol
from tianshou.policy import A2CPolicy
from tianshou.policy.modelfree.pg import TDistParams


class NPGPolicy(A2CPolicy):
    """Implementation of Natural Policy Gradient.

    https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param int optim_critic_iters: Number of times to optimize critic network per
        update. Default to 5.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close to
        1. Default to False.
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
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Callable[[TDistParams], torch.distributions.Distribution],
        advantage_normalization: bool = True,
        optim_critic_iters: int = 5,
        actor_step_size: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        del self._weight_vf, self._weight_ent, self._grad_norm
        self._norm_adv = advantage_normalization
        self._optim_critic_iters = optim_critic_iters
        self._step_size = actor_step_size
        # adjusts Hessian-vector product calculation for numerical stability
        self._damping = 0.1

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        batch = super().process_fn(batch, buffer, indices)
        old_log_prob = []
        with torch.no_grad():
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                old_log_prob.append(self(minibatch).dist.log_prob(minibatch.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        if self._norm_adv:
            batch.adv = (batch.adv - batch.adv.mean()) / batch.adv.std()
        return batch

    def learn(  # type: ignore
        self,
        batch: Batch,
        batch_size: int,
        repeat: int,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        actor_losses, vf_losses, kls = [], [], []
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                # optimize actor
                # direction: calculate villia gradient
                dist = self(minibatch).dist
                log_prob = dist.log_prob(minibatch.act)
                log_prob = log_prob.reshape(log_prob.size(0), -1).transpose(0, 1)
                actor_loss = -(log_prob * minibatch.adv).mean()
                flat_grads = self._get_flat_grad(actor_loss, self.actor, retain_graph=True).detach()

                # direction: calculate natural gradient
                with torch.no_grad():
                    old_dist = self(minibatch).dist

                kl = kl_divergence(old_dist, dist).mean()
                # calculate first order gradient of kl with respect to theta
                flat_kl_grad = self._get_flat_grad(kl, self.actor, create_graph=True)
                search_direction = -self._conjugate_gradients(flat_grads, flat_kl_grad, nsteps=10)

                # step
                with torch.no_grad():
                    flat_params = torch.cat(
                        [param.data.view(-1) for param in self.actor.parameters()],
                    )
                    new_flat_params = flat_params + self._step_size * search_direction
                    self._set_from_flat_params(self.actor, new_flat_params)
                    new_dist = self(minibatch).dist
                    kl = kl_divergence(old_dist, new_dist).mean()

                # optimize citirc
                for _ in range(self._optim_critic_iters):
                    value = self.critic(minibatch.obs).flatten()
                    vf_loss = F.mse_loss(minibatch.returns, value)
                    self.optim.zero_grad()
                    vf_loss.backward()
                    self.optim.step()

                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                kls.append(kl.item())

        return {"loss/actor": actor_losses, "loss/vf": vf_losses, "kl": kls}

    def _MVP(self, v: torch.Tensor, flat_kl_grad: torch.Tensor) -> torch.Tensor:
        """Matrix vector product."""
        # caculate second order gradient of kl with respect to theta
        kl_v = (flat_kl_grad * v).sum()
        flat_kl_grad_grad = self._get_flat_grad(kl_v, self.actor, retain_graph=True).detach()
        return flat_kl_grad_grad + v * self._damping

    def _conjugate_gradients(
        self,
        minibatch: torch.Tensor,
        flat_kl_grad: torch.Tensor,
        nsteps: int = 10,
        residual_tol: float = 1e-10,
    ) -> torch.Tensor:
        x = torch.zeros_like(minibatch)
        r, p = minibatch.clone(), minibatch.clone()
        # Note: should be 'r, p = minibatch - MVP(x)', but for x=0, MVP(x)=0.
        # Change if doing warm start.
        rdotr = r.dot(r)
        for _ in range(nsteps):
            z = self._MVP(p, flat_kl_grad)
            alpha = rdotr / p.dot(z)
            x += alpha * p
            r -= alpha * z
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            p = r + new_rdotr / rdotr * p
            rdotr = new_rdotr
        return x

    def _get_flat_grad(self, y: torch.Tensor, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        grads = torch.autograd.grad(y, model.parameters(), **kwargs)  # type: ignore
        return torch.cat([grad.reshape(-1) for grad in grads])

    def _set_from_flat_params(self, model: nn.Module, flat_params: torch.Tensor) -> nn.Module:
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_ind : prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size
        return model
