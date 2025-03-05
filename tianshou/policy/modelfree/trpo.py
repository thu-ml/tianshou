import warnings
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

from tianshou.data import Batch, SequenceSummaryStats
from tianshou.policy import NPG
from tianshou.policy.base import TLearningRateScheduler
from tianshou.policy.modelfree.npg import NPGTrainingStats
from tianshou.policy.modelfree.pg import ActorPolicy
from tianshou.utils.net.continuous import Critic
from tianshou.utils.net.discrete import Critic as DiscreteCritic


@dataclass(kw_only=True)
class TRPOTrainingStats(NPGTrainingStats):
    step_size: SequenceSummaryStats


TTRPOTrainingStats = TypeVar("TTRPOTrainingStats", bound=TRPOTrainingStats)


class TRPO(NPG[TTRPOTrainingStats]):
    """Implementation of Trust Region Policy Optimization. arXiv:1502.05477."""

    def __init__(
        self,
        *,
        policy: ActorPolicy,
        critic: torch.nn.Module | Critic | DiscreteCritic,
        optim: torch.optim.Optimizer,
        max_kl: float = 0.01,
        backtrack_coeff: float = 0.8,
        max_backtracks: int = 10,
        optim_critic_iters: int = 5,
        actor_step_size: float = 0.5,
        advantage_normalization: bool = True,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        discount_factor: float = 0.99,
        # TODO: rename to return_normalization?
        reward_normalization: bool = False,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        """
        :param critic: the critic network. (s -> V(s))
        :param optim: the optimizer for actor and critic network.
        :param max_kl: max kl-divergence used to constrain each actor network update.
        :param backtrack_coeff: Coefficient to be multiplied by step size when
            constraints are not met.
        :param max_backtracks: Max number of backtracking times in linesearch.
        :param optim_critic_iters: Number of times to optimize critic network per update.
        :param actor_step_size: step size for actor update in natural gradient direction.
        :param advantage_normalization: whether to do per mini-batch advantage
            normalization.
        :param gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        :param max_batchsize: the maximum size of the batch when computing GAE.
        :param discount_factor: in [0, 1].
        :param reward_normalization: normalize estimated values to have std close to 1.
        :param lr_scheduler: if not None, will be called in `policy.update()`.
        """
        super().__init__(
            policy=policy,
            critic=critic,
            optim=optim,
            optim_critic_iters=optim_critic_iters,
            actor_step_size=actor_step_size,
            advantage_normalization=advantage_normalization,
            gae_lambda=gae_lambda,
            max_batchsize=max_batchsize,
            discount_factor=discount_factor,
            reward_normalization=reward_normalization,
            lr_scheduler=lr_scheduler,
        )
        self.max_backtracks = max_backtracks
        self.max_kl = max_kl
        self.backtrack_coeff = backtrack_coeff

    def _update_with_batch(  # type: ignore
        self,
        batch: Batch,
        batch_size: int | None,
        repeat: int,
        **kwargs: Any,
    ) -> TTRPOTrainingStats:
        actor_losses, vf_losses, step_sizes, kls = [], [], [], []
        split_batch_size = batch_size or -1
        for _ in range(repeat):
            for minibatch in batch.split(split_batch_size, merge_last=True):
                # optimize actor
                # direction: calculate villia gradient
                dist = self.policy(minibatch).dist  # TODO could come from batch
                ratio = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                actor_loss = -(ratio * minibatch.adv).mean()
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

                # stepsize: calculate max stepsize constrained by kl bound
                step_size = torch.sqrt(
                    2
                    * self.max_kl
                    / (search_direction * self._MVP(search_direction, flat_kl_grad)).sum(
                        0,
                        keepdim=True,
                    ),
                )

                # stepsize: linesearch stepsize
                with torch.no_grad():
                    flat_params = torch.cat(
                        [param.data.view(-1) for param in self.policy.actor.parameters()],
                    )
                    for i in range(self.max_backtracks):
                        new_flat_params = flat_params + step_size * search_direction
                        self._set_from_flat_params(self.policy.actor, new_flat_params)
                        # calculate kl and if in bound, loss actually down
                        new_dist = self.policy(minibatch).dist
                        new_dratio = (
                            (new_dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                        )
                        new_dratio = new_dratio.reshape(new_dratio.size(0), -1).transpose(0, 1)
                        new_actor_loss = -(new_dratio * minibatch.adv).mean()
                        kl = kl_divergence(old_dist, new_dist).mean()

                        if kl < self.max_kl and new_actor_loss < actor_loss:
                            if i > 0:
                                warnings.warn(f"Backtracking to step {i}.")
                            break
                        if i < self.max_backtracks - 1:
                            step_size = step_size * self.backtrack_coeff
                        else:
                            self._set_from_flat_params(self.policy.actor, new_flat_params)
                            step_size = torch.tensor([0.0])
                            warnings.warn(
                                "Line search failed! It seems hyperparamters"
                                " are poor and need to be changed.",
                            )

                # optimize critic
                # TODO: remove type-ignore once the top-level type-ignore is removed
                for _ in range(self.optim_critic_iters):  # type: ignore
                    value = self.critic(minibatch.obs).flatten()
                    vf_loss = F.mse_loss(minibatch.returns, value)
                    self.optim.zero_grad()
                    vf_loss.backward()
                    self.optim.step()

                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                step_sizes.append(step_size.item())
                kls.append(kl.item())

        actor_loss_summary_stat = SequenceSummaryStats.from_sequence(actor_losses)
        vf_loss_summary_stat = SequenceSummaryStats.from_sequence(vf_losses)
        kl_summary_stat = SequenceSummaryStats.from_sequence(kls)
        step_size_stat = SequenceSummaryStats.from_sequence(step_sizes)

        return TRPOTrainingStats(  # type: ignore[return-value]
            actor_loss=actor_loss_summary_stat,
            vf_loss=vf_loss_summary_stat,
            kl=kl_summary_stat,
            step_size=step_size_stat,
        )
