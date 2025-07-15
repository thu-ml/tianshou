from abc import ABC
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.algorithm.algorithm_base import (
    OnPolicyAlgorithm,
    TrainingStats,
)
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.data.types import BatchWithAdvantagesProtocol, RolloutBatchProtocol
from tianshou.utils import RunningMeanStd
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.continuous import ContinuousCritic
from tianshou.utils.net.discrete import DiscreteCritic


@dataclass(kw_only=True)
class A2CTrainingStats(TrainingStats):
    loss: SequenceSummaryStats
    actor_loss: SequenceSummaryStats
    vf_loss: SequenceSummaryStats
    ent_loss: SequenceSummaryStats
    gradient_steps: int


class ActorCriticOnPolicyAlgorithm(OnPolicyAlgorithm[ProbabilisticActorPolicy], ABC):
    """Abstract base class for actor-critic algorithms that use generalized advantage estimation (GAE)."""

    def __init__(
        self,
        *,
        policy: ProbabilisticActorPolicy,
        critic: torch.nn.Module | ContinuousCritic | DiscreteCritic,
        optim: OptimizerFactory,
        optim_include_actor: bool,
        max_grad_norm: float | None = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        gamma: float = 0.99,
        return_scaling: bool = False,
    ) -> None:
        """
        :param critic: the critic network. (s -> V(s))
        :param optim: the optimizer factory.
        :param optim_include_actor: whether the optimizer shall include the actor network's parameters.
            Pass False for algorithms that shall update only the critic via the optimizer.
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
        )
        self.critic = critic
        assert 0.0 <= gae_lambda <= 1.0, f"GAE lambda should be in [0, 1] but got: {gae_lambda}"
        self.gae_lambda = gae_lambda
        self.max_batchsize = max_batchsize
        if optim_include_actor:
            self.optim = self._create_optimizer(
                ActorCritic(self.policy.actor, self.critic), optim, max_grad_norm=max_grad_norm
            )
        else:
            self.optim = self._create_optimizer(self.critic, optim, max_grad_norm=max_grad_norm)
        self.gamma = gamma
        self.return_scaling = return_scaling
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8

    def _add_returns_and_advantages(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        """Adds the returns and advantages to the given batch."""
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                v_s.append(self.critic(minibatch.obs))
                v_s_.append(self.critic(minibatch.obs_next))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Empirical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        if self.return_scaling:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        if self.return_scaling:
            batch.returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return cast(BatchWithAdvantagesProtocol, batch)


class A2C(ActorCriticOnPolicyAlgorithm):
    """Implementation of (synchronous) Advantage Actor-Critic (A2C). arXiv:1602.01783."""

    def __init__(
        self,
        *,
        policy: ProbabilisticActorPolicy,
        critic: torch.nn.Module | ContinuousCritic | DiscreteCritic,
        optim: OptimizerFactory,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float | None = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        gamma: float = 0.99,
        return_scaling: bool = False,
    ) -> None:
        """
        :param policy: the policy containing the actor network.
        :param critic: the critic network. (s -> V(s))
        :param optim: the optimizer factory.
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
        super().__init__(
            policy=policy,
            critic=critic,
            optim=optim,
            optim_include_actor=True,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            max_batchsize=max_batchsize,
            gamma=gamma,
            return_scaling=return_scaling,
        )
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        batch = self._add_returns_and_advantages(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        return batch

    def _update_with_batch(  # type: ignore[override]
        self,
        batch: BatchWithAdvantagesProtocol,
        batch_size: int | None,
        repeat: int,
    ) -> A2CTrainingStats:
        losses, actor_losses, vf_losses, ent_losses = [], [], [], []
        split_batch_size = batch_size or -1
        gradient_steps = 0
        for _ in range(repeat):
            for minibatch in batch.split(split_batch_size, merge_last=True):
                gradient_steps += 1

                # calculate loss for actor
                dist = self.policy(minibatch).dist
                log_prob = dist.log_prob(minibatch.act)
                log_prob = log_prob.reshape(len(minibatch.adv), -1).transpose(0, 1)
                actor_loss = -(log_prob * minibatch.adv).mean()
                # calculate loss for critic
                value = self.critic(minibatch.obs).flatten()
                vf_loss = F.mse_loss(minibatch.returns, value)
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = actor_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss
                self.optim.step(loss)
                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        loss_summary_stat = SequenceSummaryStats.from_sequence(losses)
        actor_loss_summary_stat = SequenceSummaryStats.from_sequence(actor_losses)
        vf_loss_summary_stat = SequenceSummaryStats.from_sequence(vf_losses)
        ent_loss_summary_stat = SequenceSummaryStats.from_sequence(ent_losses)

        return A2CTrainingStats(
            loss=loss_summary_stat,
            actor_loss=actor_loss_summary_stat,
            vf_loss=vf_loss_summary_stat,
            ent_loss=ent_loss_summary_stat,
            gradient_steps=gradient_steps,
        )
