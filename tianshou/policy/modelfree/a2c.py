from abc import ABC
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.data.types import BatchWithAdvantagesProtocol, RolloutBatchProtocol
from tianshou.policy import Reinforce
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.policy.modelfree.pg import ActorPolicy, TPGTrainingStats
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.continuous import Critic
from tianshou.utils.net.discrete import Critic as DiscreteCritic


@dataclass(kw_only=True)
class A2CTrainingStats(TrainingStats):
    loss: SequenceSummaryStats
    actor_loss: SequenceSummaryStats
    vf_loss: SequenceSummaryStats
    ent_loss: SequenceSummaryStats


TA2CTrainingStats = TypeVar("TA2CTrainingStats", bound=A2CTrainingStats)


class AbstractActorCriticWithAdvantage(Reinforce[TPGTrainingStats], Generic[TPGTrainingStats], ABC):
    def __init__(
        self,
        *,
        policy: ActorPolicy,
        critic: torch.nn.Module | Critic | DiscreteCritic,
        optim: torch.optim.Optimizer,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            policy=policy,
            optim=optim,
            discount_factor=discount_factor,
            reward_normalization=reward_normalization,
            lr_scheduler=lr_scheduler,
        )
        self.critic = critic
        assert 0.0 <= gae_lambda <= 1.0, f"GAE lambda should be in [0, 1] but got: {gae_lambda}"
        self.gae_lambda = gae_lambda
        self.max_batchsize = max_batchsize
        self._actor_critic = ActorCritic(self.policy.actor, self.critic)

    def _compute_returns(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
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
        # TODO: see todo in PGPolicy.process_fn
        if self.rew_norm:  # unnormalize v_s & v_s_
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
        if self.rew_norm:
            batch.returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return cast(BatchWithAdvantagesProtocol, batch)


class A2C(AbstractActorCriticWithAdvantage[TA2CTrainingStats], Generic[TA2CTrainingStats]):
    """Implementation of Synchronous Advantage Actor-Critic. arXiv:1602.01783.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        *,
        policy: ActorPolicy,
        critic: torch.nn.Module | Critic | DiscreteCritic,
        optim: torch.optim.Optimizer,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float | None = None,
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
        :param vf_coef: weight for value loss.
        :param ent_coef: weight for entropy loss.
        :param max_grad_norm: clipping gradients in back propagation.
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
            gae_lambda=gae_lambda,
            max_batchsize=max_batchsize,
            discount_factor=discount_factor,
            reward_normalization=reward_normalization,
            lr_scheduler=lr_scheduler,
        )
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        return batch

    # TODO: mypy complains b/c signature is different from superclass, although
    #  it's compatible. Can this be fixed?
    def _update_with_batch(  # type: ignore
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> TA2CTrainingStats:
        losses, actor_losses, vf_losses, ent_losses = [], [], [], []
        split_batch_size = batch_size or -1
        for _ in range(repeat):
            for minibatch in batch.split(split_batch_size, merge_last=True):
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
                self.optim.zero_grad()
                loss.backward()
                if self.max_grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(),
                        max_norm=self.max_grad_norm,
                    )
                self.optim.step()
                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        loss_summary_stat = SequenceSummaryStats.from_sequence(losses)
        actor_loss_summary_stat = SequenceSummaryStats.from_sequence(actor_losses)
        vf_loss_summary_stat = SequenceSummaryStats.from_sequence(vf_losses)
        ent_loss_summary_stat = SequenceSummaryStats.from_sequence(ent_losses)

        return A2CTrainingStats(  # type: ignore[return-value]
            loss=loss_summary_stat,
            actor_loss=actor_loss_summary_stat,
            vf_loss=vf_loss_summary_stat,
            ent_loss=ent_loss_summary_stat,
        )
