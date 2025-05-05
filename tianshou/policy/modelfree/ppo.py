from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Self, TypeVar, cast

import numpy as np
import torch

from tianshou.data import ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.data.types import LogpOldProtocol, RolloutBatchProtocol
from tianshou.policy import A2C
from tianshou.policy.base import TrainingStats
from tianshou.policy.modelfree.pg import ActorPolicy
from tianshou.policy.optim import OptimizerFactory
from tianshou.utils.net.continuous import ContinuousCritic
from tianshou.utils.net.discrete import DiscreteCritic


@dataclass(kw_only=True)
class PPOTrainingStats(TrainingStats):
    loss: SequenceSummaryStats
    clip_loss: SequenceSummaryStats
    vf_loss: SequenceSummaryStats
    ent_loss: SequenceSummaryStats
    gradient_steps: int = 0

    @classmethod
    def from_sequences(
        cls,
        *,
        losses: Sequence[float],
        clip_losses: Sequence[float],
        vf_losses: Sequence[float],
        ent_losses: Sequence[float],
        gradient_steps: int = 0,
    ) -> Self:
        return cls(
            loss=SequenceSummaryStats.from_sequence(losses),
            clip_loss=SequenceSummaryStats.from_sequence(clip_losses),
            vf_loss=SequenceSummaryStats.from_sequence(vf_losses),
            ent_loss=SequenceSummaryStats.from_sequence(ent_losses),
            gradient_steps=gradient_steps,
        )


TPPOTrainingStats = TypeVar("TPPOTrainingStats", bound=PPOTrainingStats)


# TODO: the type ignore here is needed b/c the hierarchy is actually broken! Should reconsider the inheritance structure.
class PPO(A2C[TPPOTrainingStats], Generic[TPPOTrainingStats]):  # type: ignore[type-var]
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        *,
        policy: ActorPolicy,
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
        # TODO: rename to return_normalization?
        reward_normalization: bool = False,
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
        :param dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
            where c > 1 is a constant indicating the lower bound. Set to None
            to disable dual-clip PPO.
        :param value_clip: a parameter mentioned in arXiv:1811.02553v3 Sec. 4.1.
        :param advantage_normalization: whether to do per mini-batch advantage
            normalization.
        :param recompute_advantage: whether to recompute advantage every update
            repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
        :param vf_coef: weight for value loss.
        :param ent_coef: weight for entropy loss.
        :param max_grad_norm: clipping gradients in back propagation.
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
        :param reward_normalization: normalize estimated values to have std close to 1.
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
            reward_normalization=reward_normalization,
        )
        self.eps_clip = eps_clip
        self.dual_clip = dual_clip
        self.value_clip = value_clip
        self.norm_adv = advantage_normalization
        self.recompute_adv = recompute_advantage

    def preprocess_batch(
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

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
    ) -> TPPOTrainingStats:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        gradient_steps = 0
        split_batch_size = batch_size or -1
        for step in range(repeat):
            if self.recompute_adv and step > 0:
                batch = self._add_returns_and_advantages(batch, self._buffer, self._indices)
            for minibatch in batch.split(split_batch_size, merge_last=True):
                gradient_steps += 1
                # calculate loss for actor
                advantages = minibatch.adv
                dist = self.policy(minibatch).dist
                if self.norm_adv:
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

        return PPOTrainingStats.from_sequences(  # type: ignore[return-value]
            losses=losses,
            clip_losses=clip_losses,
            vf_losses=vf_losses,
            ent_losses=ent_losses,
            gradient_steps=gradient_steps,
        )
