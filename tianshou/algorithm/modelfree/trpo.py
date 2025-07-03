import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

from tianshou.algorithm import NPG
from tianshou.algorithm.modelfree.npg import NPGTrainingStats
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import SequenceSummaryStats
from tianshou.data.types import BatchWithAdvantagesProtocol
from tianshou.utils.net.continuous import ContinuousCritic
from tianshou.utils.net.discrete import DiscreteCritic


@dataclass(kw_only=True)
class TRPOTrainingStats(NPGTrainingStats):
    step_size: SequenceSummaryStats


class TRPO(NPG):
    """Implementation of Trust Region Policy Optimization. arXiv:1502.05477."""

    def __init__(
        self,
        *,
        policy: ProbabilisticActorPolicy,
        critic: torch.nn.Module | ContinuousCritic | DiscreteCritic,
        optim: OptimizerFactory,
        max_kl: float = 0.01,
        backtrack_coeff: float = 0.8,
        max_backtracks: int = 10,
        optim_critic_iters: int = 5,
        trust_region_size: float = 0.5,
        advantage_normalization: bool = True,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        gamma: float = 0.99,
        return_scaling: bool = False,
    ) -> None:
        """
        :param policy: the policy
        :param critic: the critic network. (s -> V(s))
        :param optim: the optimizer factory for the critic network.
        :param max_kl: max kl-divergence used to constrain each actor network update.
        :param backtrack_coeff: Coefficient to be multiplied by step size when
            constraints are not met.
        :param max_backtracks: Max number of backtracking times in linesearch.
        :param optim_critic_iters: the number of optimization steps performed on the critic network
            for each policy (actor) update.
            Controls the learning rate balance between critic and actor.
            Higher values prioritize critic accuracy by training the value function more
            extensively before each policy update, which can improve stability but slow down
            training. Lower values maintain a more even learning pace between policy and value
            function but may lead to less reliable advantage estimates.
            Typically set between 1 and 10, depending on the complexity of the value function.
        :param trust_region_size: the parameter delta - a scalar multiplier for policy updates in the natural gradient direction.
            The mathematical meaning is the trust region size, which is the maximum KL divergence
            allowed between the old and new policy distributions.
            Controls how far the policy parameters move in the calculated direction
            during each update. Higher values allow for faster learning but may cause instability
            or policy deterioration; lower values provide more stable but slower learning. Unlike
            regular policy gradients, natural gradients already account for the local geometry of
            the parameter space, making this step size more robust to different parameterizations.
            Typically set between 0.1 and 1.0 for most reinforcement learning tasks.
        :param advantage_normalization: whether to do per mini-batch advantage
            normalization.
        :param gae_lambda: the lambda parameter in [0, 1] for generalized advantage estimation (GAE).
            Controls the bias-variance tradeoff in advantage estimates, acting as a
            weighting factor for combining different n-step advantage estimators. Higher values
            (closer to 1) reduce bias but increase variance by giving more weight to longer
            trajectories, while lower values (closer to 0) reduce variance but increase bias
            by relying more on the immediate TD error and value function estimates. At λ=0,
            GAE becomes equivalent to the one-step TD error (high bias, low variance); at λ=1,
            it becomes equivalent to Monte Carlo advantage estimation (low bias, high variance).
            Intermediate values create a weighted average of n-step returns, with exponentially
            decaying weights for longer-horizon returns. Typically set between 0.9 and 0.99 for
            most policy gradient methods.
        :param max_batchsize: the maximum number of samples to process at once when computing
            generalized advantage estimation (GAE) and value function predictions.
            Controls memory usage by breaking large batches into smaller chunks processed sequentially.
            Higher values may increase speed but require more GPU/CPU memory; lower values
            reduce memory requirements but may increase computation time. Should be adjusted
            based on available hardware resources and total batch size of your training data.
        :param gamma: the discount factor in [0, 1] for future rewards.
            This determines how much future rewards are valued compared to immediate ones.
            Lower values (closer to 0) make the agent focus on immediate rewards, creating "myopic"
            behavior. Higher values (closer to 1) make the agent value long-term rewards more,
            potentially improving performance in tasks where delayed rewards are important but
            increasing training variance by incorporating more environmental stochasticity.
            Typically set between 0.9 and 0.99 for most reinforcement learning tasks
        :param return_scaling: flag indicating whether to enable scaling of estimated returns by
            dividing them by their running standard deviation without centering the mean.
            This reduces the magnitude variation of advantages across different episodes while
            preserving their signs and relative ordering.
            The use of running statistics (rather than batch-specific scaling) means that early
            training experiences may be scaled differently than later ones as the statistics evolve.
            When enabled, this improves training stability in environments with highly variable
            reward scales and makes the algorithm less sensitive to learning rate settings.
            However, it may reduce the algorithm's ability to distinguish between episodes with
            different absolute return magnitudes.
            Best used in environments where the relative ordering of actions is more important
            than the absolute scale of returns.
        """
        super().__init__(
            policy=policy,
            critic=critic,
            optim=optim,
            optim_critic_iters=optim_critic_iters,
            trust_region_size=trust_region_size,
            advantage_normalization=advantage_normalization,
            gae_lambda=gae_lambda,
            max_batchsize=max_batchsize,
            gamma=gamma,
            return_scaling=return_scaling,
        )
        self.max_backtracks = max_backtracks
        self.max_kl = max_kl
        self.backtrack_coeff = backtrack_coeff

    def _update_with_batch(  # type: ignore[override]
        self,
        batch: BatchWithAdvantagesProtocol,
        batch_size: int | None,
        repeat: int,
    ) -> TRPOTrainingStats:
        actor_losses, vf_losses, step_sizes, kls = [], [], [], []
        split_batch_size = batch_size or -1
        for _ in range(repeat):
            for minibatch in batch.split(split_batch_size, merge_last=True):
                # optimize actor
                # direction: calculate villia gradient
                dist = self.policy(minibatch).dist
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
                for _ in range(self.optim_critic_iters):
                    value = self.critic(minibatch.obs).flatten()
                    vf_loss = F.mse_loss(minibatch.returns, value)
                    self.optim.step(vf_loss)

                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                step_sizes.append(step_size.item())
                kls.append(kl.item())

        actor_loss_summary_stat = SequenceSummaryStats.from_sequence(actor_losses)
        vf_loss_summary_stat = SequenceSummaryStats.from_sequence(vf_losses)
        kl_summary_stat = SequenceSummaryStats.from_sequence(kls)
        step_size_stat = SequenceSummaryStats.from_sequence(step_sizes)

        return TRPOTrainingStats(
            actor_loss=actor_loss_summary_stat,
            vf_loss=vf_loss_summary_stat,
            kl=kl_summary_stat,
            step_size=step_size_stat,
        )
