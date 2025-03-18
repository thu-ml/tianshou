from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import kl_divergence

from tianshou.data import Batch, ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.data.types import BatchWithAdvantagesProtocol, RolloutBatchProtocol
from tianshou.policy.base import TrainingStats
from tianshou.policy.modelfree.a2c import ActorCriticOnPolicyAlgorithm
from tianshou.policy.modelfree.pg import ActorPolicy
from tianshou.policy.optim import OptimizerFactory
from tianshou.utils.net.continuous import Critic
from tianshou.utils.net.discrete import Critic as DiscreteCritic


@dataclass(kw_only=True)
class NPGTrainingStats(TrainingStats):
    actor_loss: SequenceSummaryStats
    vf_loss: SequenceSummaryStats
    kl: SequenceSummaryStats


TNPGTrainingStats = TypeVar("TNPGTrainingStats", bound=NPGTrainingStats)


class NPG(ActorCriticOnPolicyAlgorithm[TNPGTrainingStats], Generic[TNPGTrainingStats]):
    """Implementation of Natural Policy Gradient.

    https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf
    """

    def __init__(
        self,
        *,
        policy: ActorPolicy,
        critic: torch.nn.Module | Critic | DiscreteCritic,
        optim: OptimizerFactory,
        optim_critic_iters: int = 5,
        actor_step_size: float = 0.5,
        advantage_normalization: bool = True,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        discount_factor: float = 0.99,
        # TODO: rename to return_normalization?
        reward_normalization: bool = False,
    ) -> None:
        """
        :param policy: the policy containing the actor network.
        :param critic: the critic network. (s -> V(s))
        :param optim: the optimizer factory for the critic network.
        :param optim_critic_iters: Number of times to optimize critic network per update.
        :param actor_step_size: step size for actor update in natural gradient direction.
        :param advantage_normalization: whether to do per mini-batch advantage
            normalization.
        :param gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        :param max_batchsize: the maximum size of the batch when computing GAE.
        :param discount_factor: in [0, 1].
        :param reward_normalization: normalize estimated values to have std close to 1.
        """
        super().__init__(
            policy=policy,
            critic=critic,
            optim=optim,
            optim_include_actor=False,
            gae_lambda=gae_lambda,
            max_batchsize=max_batchsize,
            discount_factor=discount_factor,
            reward_normalization=reward_normalization,
        )
        self.norm_adv = advantage_normalization
        self.optim_critic_iters = optim_critic_iters
        self.actor_step_size = actor_step_size
        # adjusts Hessian-vector product calculation for numerical stability
        self._damping = 0.1

    def preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        batch = self._add_returns_and_advantages(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        old_log_prob = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                old_log_prob.append(self.policy(minibatch).dist.log_prob(minibatch.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        if self.norm_adv:
            batch.adv = (batch.adv - batch.adv.mean()) / batch.adv.std()
        return batch

    def _update_with_batch(  # type: ignore
        self,
        batch: Batch,
        batch_size: int | None,
        repeat: int,
    ) -> TNPGTrainingStats:
        actor_losses, vf_losses, kls = [], [], []
        split_batch_size = batch_size or -1
        for _ in range(repeat):
            for minibatch in batch.split(split_batch_size, merge_last=True):
                # optimize actor
                # direction: calculate villia gradient
                dist = self.policy(minibatch).dist
                log_prob = dist.log_prob(minibatch.act)
                log_prob = log_prob.reshape(log_prob.size(0), -1).transpose(0, 1)
                actor_loss = -(log_prob * minibatch.adv).mean()
                flat_grads = self._get_flat_grad(
                    actor_loss, self.policy.actor, retain_graph=True
                ).detach()

                # direction: calculate natural gradient
                with torch.no_grad():
                    old_dist = self.policy(minibatch).dist

                kl = kl_divergence(old_dist, dist).mean()
                # calculate first order gradient of kl with respect to theta
                flat_kl_grad = self._get_flat_grad(kl, self.policy.actor, create_graph=True)
                search_direction = -self._conjugate_gradients(flat_grads, flat_kl_grad, nsteps=10)

                # step
                with torch.no_grad():
                    flat_params = torch.cat(
                        [param.data.view(-1) for param in self.policy.actor.parameters()],
                    )
                    new_flat_params = flat_params + self.actor_step_size * search_direction
                    self._set_from_flat_params(self.policy.actor, new_flat_params)
                    new_dist = self.policy(minibatch).dist
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
        flat_kl_grad_grad = self._get_flat_grad(kl_v, self.policy.actor, retain_graph=True).detach()
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
