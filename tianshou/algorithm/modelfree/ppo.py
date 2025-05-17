from typing import cast

import numpy as np
import torch

from tianshou.algorithm import A2C
from tianshou.algorithm.modelfree.a2c import A2CTrainingStats
from tianshou.algorithm.modelfree.reinforce import ActorPolicyProbabilistic
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.data.types import LogpOldProtocol, RolloutBatchProtocol
from tianshou.utils.net.continuous import ContinuousCritic
from tianshou.utils.net.discrete import DiscreteCritic


class PPO(A2C):
    """Implementation of Proximal Policy Optimization. arXiv:1707.06347."""

    def __init__(
        self,
        *,
        policy: ActorPolicyProbabilistic,
        critic: torch.nn.Module | ContinuousCritic | DiscreteCritic,
        optim: OptimizerFactory,
        eps_clip: float = 0.2,
        dual_clip: float | None = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float | None = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        gamma: float = 0.99,
        return_scaling: bool = False,
    ) -> None:
        r"""
        :param policy: the policy containing the actor network.
        :param critic: the critic network. (s -> V(s))
        :param optim: the optimizer factory for the policy's actor network and the critic networks.
        :param eps_clip: determines the range of allowed change in the policy during a policy update:
            The ratio of action probabilities indicated by the new and old policy is
            constrained to stay in the interval [1 - eps_clip, 1 + eps_clip].
            Small values thus force the new policy to stay close to the old policy.
            Typical values range between 0.1 and 0.3, the value of 0.2 is recommended
            in the original PPO paper.
            The optimal value depends on the environment; more stochastic environments may
            need larger values.
        :param dual_clip: a clipping parameter (denoted as c in the literature) that prevents
            excessive pessimism in policy updates for negative-advantage actions.
            Excessive pessimism occurs when the policy update too strongly reduces the probability
            of selecting actions that led to negative advantages, potentially eliminating useful
            actions based on limited negative experiences.
            When enabled (c > 1), the objective for negative advantages becomes:
            max(min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A), c*A), where min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
            is the original single-clipping objective determined by `eps_clip`.
            This creates a floor on negative policy gradients, maintaining some probability
            of exploring actions despite initial negative outcomes.
            Larger values (e.g., 2.0 to 5.0) maintain more exploration, while values closer
            to 1.0 provide less protection against pessimistic updates.
            Set to None to disable dual clipping.
        :param value_clip: flag indicating whether to enable clipping for value function updates.
            When enabled, restricts how much the value function estimate can change from its
            previous prediction, using the same clipping range as the policy updates (eps_clip).
            This stabilizes training by preventing large fluctuations in value estimates,
            particularly useful in environments with high reward variance.
            The clipped value loss uses a pessimistic approach, taking the maximum of the
            original and clipped value errors:
            max((returns - value)², (returns - v_clipped)²)
            Setting to True often improves training stability but may slow convergence.
            Implementation follows the approach mentioned in arXiv:1811.02553v3 Sec. 4.1.
        :param advantage_normalization: whether to do per mini-batch advantage
            normalization.
        :param recompute_advantage: whether to recompute advantage every update
            repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
        :param vf_coef: coefficient that weights the value loss relative to the actor loss in
            the overall loss function.
            Higher values prioritize accurate value function estimation over policy improvement.
            Controls the trade-off between policy optimization and value function fitting.
            Typically set between 0.5 and 1.0 for most actor-critic implementations.
        :param ent_coef: coefficient that weights the entropy bonus relative to the actor loss.
            Controls the exploration-exploitation trade-off by encouraging policy entropy.
            Higher values promote more exploration by encouraging a more uniform action distribution.
            Lower values focus more on exploitation of the current policy's knowledge.
            Typically set between 0.01 and 0.05 for most actor-critic implementations.
        :param max_grad_norm: the maximum L2 norm threshold for gradient clipping.
            When not None, gradients will be rescaled using to ensure their L2 norm does not
            exceed this value. This prevents exploding gradients and stabilizes training by
            limiting the magnitude of parameter updates.
            Set to None to disable gradient clipping.
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
        :param max_batchsize: the maximum size of the batch when computing GAE.
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
        assert (
            dual_clip is None or dual_clip > 1.0
        ), f"Dual-clip PPO parameter should greater than 1.0 but got {dual_clip}"

        super().__init__(
            policy=policy,
            critic=critic,
            optim=optim,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            max_batchsize=max_batchsize,
            gamma=gamma,
            return_scaling=return_scaling,
        )
        self.eps_clip = eps_clip
        self.dual_clip = dual_clip
        self.value_clip = value_clip
        self.advantage_normalization = advantage_normalization
        self.recompute_adv = recompute_advantage

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> LogpOldProtocol:
        if self.recompute_adv:
            # buffer input `buffer` and `indices` to be used in `_update_with_batch()`.
            self._buffer, self._indices = buffer, indices
        batch = self._add_returns_and_advantages(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        logp_old = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                logp_old.append(self.policy(minibatch).dist.log_prob(minibatch.act))
            batch.logp_old = torch.cat(logp_old, dim=0).flatten()
        return cast(LogpOldProtocol, batch)

    def _update_with_batch(  # type: ignore[override]
        self,
        batch: LogpOldProtocol,
        batch_size: int | None,
        repeat: int,
    ) -> A2CTrainingStats:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        gradient_steps = 0
        split_batch_size = batch_size or -1
        for step in range(repeat):
            if self.recompute_adv and step > 0:
                batch = cast(
                    LogpOldProtocol,
                    self._add_returns_and_advantages(batch, self._buffer, self._indices),
                )
            for minibatch in batch.split(split_batch_size, merge_last=True):
                gradient_steps += 1
                # calculate loss for actor
                advantages = minibatch.adv
                dist = self.policy(minibatch).dist
                if self.advantage_normalization:
                    mean, std = advantages.mean(), advantages.std()
                    advantages = (advantages - mean) / (std + self._eps)  # per-batch norm
                ratios = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                ratios = ratios.reshape(ratios.size(0), -1).transpose(0, 1)
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
                if self.dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self.dual_clip * advantages)
                    clip_loss = -torch.where(advantages < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                # calculate loss for critic
                value = self.critic(minibatch.obs).flatten()
                if self.value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(
                        -self.eps_clip,
                        self.eps_clip,
                    )
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = clip_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss
                self.optim.step(loss)
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return A2CTrainingStats(
            loss=SequenceSummaryStats.from_sequence(losses),
            actor_loss=SequenceSummaryStats.from_sequence(clip_losses),
            vf_loss=SequenceSummaryStats.from_sequence(vf_losses),
            ent_loss=SequenceSummaryStats.from_sequence(ent_losses),
            gradient_steps=gradient_steps,
        )
