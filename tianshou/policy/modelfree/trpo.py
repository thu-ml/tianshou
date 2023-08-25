import warnings
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

from tianshou.data import Batch
from tianshou.policy import NPGPolicy
from tianshou.policy.modelfree.pg import TDistParams


class TRPOPolicy(NPGPolicy):
    """Implementation of Trust Region Policy Optimization. arXiv:1502.05477.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param int optim_critic_iters: Number of times to optimize critic network per
        update. Default to 5.
    :param int max_kl: max kl-divergence used to constrain each actor network update.
        Default to 0.01.
    :param float backtrack_coeff: Coefficient to be multiplied by step size when
        constraints are not met. Default to 0.8.
    :param int max_backtracks: Max number of backtracking times in linesearch. Default
        to 10.
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
        max_kl: float = 0.01,
        backtrack_coeff: float = 0.8,
        max_backtracks: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        self._max_backtracks = max_backtracks
        self._delta = max_kl
        self._backtrack_coeff = backtrack_coeff
        self._optim_critic_iters: int

    def learn(  # type: ignore
        self,
        batch: Batch,
        batch_size: int,
        repeat: int,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        actor_losses, vf_losses, step_sizes, kls = [], [], [], []
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                # optimize actor
                # direction: calculate villia gradient
                dist = self(minibatch).dist  # TODO could come from batch
                ratio = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                actor_loss = -(ratio * minibatch.adv).mean()
                flat_grads = self._get_flat_grad(actor_loss, self.actor, retain_graph=True).detach()

                # direction: calculate natural gradient
                with torch.no_grad():
                    old_dist = self(minibatch).dist

                kl = kl_divergence(old_dist, dist).mean()
                # calculate first order gradient of kl with respect to theta
                flat_kl_grad = self._get_flat_grad(kl, self.actor, create_graph=True)
                search_direction = -self._conjugate_gradients(flat_grads, flat_kl_grad, nsteps=10)

                # stepsize: calculate max stepsize constrained by kl bound
                step_size = torch.sqrt(
                    2
                    * self._delta
                    / (search_direction * self._MVP(search_direction, flat_kl_grad)).sum(
                        0,
                        keepdim=True,
                    ),
                )

                # stepsize: linesearch stepsize
                with torch.no_grad():
                    flat_params = torch.cat(
                        [param.data.view(-1) for param in self.actor.parameters()],
                    )
                    for i in range(self._max_backtracks):
                        new_flat_params = flat_params + step_size * search_direction
                        self._set_from_flat_params(self.actor, new_flat_params)
                        # calculate kl and if in bound, loss actually down
                        new_dist = self(minibatch).dist
                        new_dratio = (
                            (new_dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                        )
                        new_dratio = new_dratio.reshape(new_dratio.size(0), -1).transpose(0, 1)
                        new_actor_loss = -(new_dratio * minibatch.adv).mean()
                        kl = kl_divergence(old_dist, new_dist).mean()

                        if kl < self._delta and new_actor_loss < actor_loss:
                            if i > 0:
                                warnings.warn(f"Backtracking to step {i}.")
                            break
                        if i < self._max_backtracks - 1:
                            step_size = step_size * self._backtrack_coeff
                        else:
                            self._set_from_flat_params(self.actor, new_flat_params)
                            step_size = torch.tensor([0.0])
                            warnings.warn(
                                "Line search failed! It seems hyperparamters"
                                " are poor and need to be changed.",
                            )

                # optimize citirc
                for _ in range(self._optim_critic_iters):
                    value = self.critic(minibatch.obs).flatten()
                    vf_loss = F.mse_loss(minibatch.returns, value)
                    self.optim.zero_grad()
                    vf_loss.backward()
                    self.optim.step()

                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                step_sizes.append(step_size.item())
                kls.append(kl.item())

        return {
            "loss/actor": actor_losses,
            "loss/vf": vf_losses,
            "step_size": step_sizes,
            "kl": kls,
        }
