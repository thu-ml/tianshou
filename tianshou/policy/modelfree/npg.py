from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import kl_divergence

from tianshou.data import Batch, ReplayBuffer, SequenceSummaryStats
from tianshou.data.types import BatchWithAdvantagesProtocol, RolloutBatchProtocol
from tianshou.policy import A2CPolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.policy.modelfree.pg import TDistributionFunction


@dataclass(kw_only=True)
class NPGTrainingStats(TrainingStats):
    actor_loss: SequenceSummaryStats
    vf_loss: SequenceSummaryStats
    kl: SequenceSummaryStats


TNPGTrainingStats = TypeVar("TNPGTrainingStats", bound=NPGTrainingStats)


# TODO: the type ignore here is needed b/c the hierarchy is actually broken! Should reconsider the inheritance structure.
class NPGPolicy(A2CPolicy[TNPGTrainingStats], Generic[TNPGTrainingStats]):  # type: ignore[type-var]
    """Implementation of Natural Policy Gradient.

    https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf

    :param actor: the actor network following the rules in BasePolicy. (s -> logits)
    :param critic: the critic network. (s -> V(s))
    :param optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :param action_space: env's action space
    :param optim_critic_iters: Number of times to optimize critic network per update.
    :param actor_step_size: step size for actor update in natural gradient direction.
    :param advantage_normalization: whether to do per mini-batch advantage
        normalization.
    :param gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
    :param max_batchsize: the maximum size of the batch when computing GAE.
    :param discount_factor: in [0, 1].
    :param reward_normalization: normalize estimated values to have std close to 1.
    :param deterministic_eval: if True, use deterministic evaluation.
    :param observation_space: the space of the observation.
    :param action_scaling: if True, scale the action from [-1, 1] to the range of
        action_space. Only used if the action_space is continuous.
    :param action_bound_method: method to bound action to range [-1, 1].
    :param lr_scheduler: if not None, will be called in `policy.update()`.
    """

    def __init__(
        self,
        *,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: TDistributionFunction,
        action_space: gym.Space,
        optim_critic_iters: int = 5,
        actor_step_size: float = 0.5,
        advantage_normalization: bool = True,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        discount_factor: float = 0.99,
        # TODO: rename to return_normalization?
        reward_normalization: bool = False,
        deterministic_eval: bool = False,
        observation_space: gym.Space | None = None,
        action_scaling: bool = True,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist_fn,
            action_space=action_space,
            # TODO: violates Liskov substitution principle, see the del statement below
            vf_coef=None,  # type: ignore
            ent_coef=None,  # type: ignore
            max_grad_norm=None,
            gae_lambda=gae_lambda,
            max_batchsize=max_batchsize,
            discount_factor=discount_factor,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        # TODO: see above, it ain't pretty...
        del self.vf_coef, self.ent_coef, self.max_grad_norm
        self.norm_adv = advantage_normalization
        self.optim_critic_iters = optim_critic_iters
        self.actor_step_size = actor_step_size
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
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                old_log_prob.append(self(minibatch).dist.log_prob(minibatch.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        if self.norm_adv:
            batch.adv = (batch.adv - batch.adv.mean()) / batch.adv.std()
        return batch

    def learn(  # type: ignore
        self,
        batch: Batch,
        batch_size: int | None,
        repeat: int,
        **kwargs: Any,
    ) -> TNPGTrainingStats:
        actor_losses, vf_losses, kls = [], [], []
        split_batch_size = batch_size or -1
        for _ in range(repeat):
            for minibatch in batch.split(split_batch_size, merge_last=True):
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
                    new_flat_params = flat_params + self.actor_step_size * search_direction
                    self._set_from_flat_params(self.actor, new_flat_params)
                    new_dist = self(minibatch).dist
                    kl = kl_divergence(old_dist, new_dist).mean()

                # optimize critic
                for _ in range(self.optim_critic_iters):
                    value = self.critic(minibatch.obs).flatten()
                    vf_loss = F.mse_loss(minibatch.returns, value)
                    self.optim.zero_grad()
                    vf_loss.backward()
                    self.optim.step()

                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                kls.append(kl.item())

        actor_loss_summary_stat = SequenceSummaryStats.from_sequence(actor_losses)
        vf_loss_summary_stat = SequenceSummaryStats.from_sequence(vf_losses)
        kl_summary_stat = SequenceSummaryStats.from_sequence(kls)

        return NPGTrainingStats(  # type: ignore[return-value]
            actor_loss=actor_loss_summary_stat,
            vf_loss=vf_loss_summary_stat,
            kl=kl_summary_stat,
        )

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
