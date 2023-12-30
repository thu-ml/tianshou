from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from tianshou.data import ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.data.types import LogpOldProtocol, RolloutBatchProtocol
from tianshou.policy import A2CPolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.policy.modelfree.pg import TDistributionFunction
from tianshou.utils.net.common import ActorCritic


@dataclass(kw_only=True)
class PPOTrainingStats(TrainingStats):
    loss: SequenceSummaryStats
    clip_loss: SequenceSummaryStats
    vf_loss: SequenceSummaryStats
    ent_loss: SequenceSummaryStats


TPPOTrainingStats = TypeVar("TPPOTrainingStats", bound=PPOTrainingStats)


# TODO: the type ignore here is needed b/c the hierarchy is actually broken! Should reconsider the inheritance structure.
class PPOPolicy(A2CPolicy[TPPOTrainingStats], Generic[TPPOTrainingStats]):  # type: ignore[type-var]
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    :param actor: the actor network following the rules in BasePolicy. (s -> logits)
    :param critic: the critic network. (s -> V(s))
    :param optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :param action_space: env's action space
    :param eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper.
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

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        *,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: TDistributionFunction,
        action_space: gym.Space,
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
        discount_factor: float = 0.99,
        # TODO: rename to return_normalization?
        reward_normalization: bool = False,
        deterministic_eval: bool = False,
        observation_space: gym.Space | None = None,
        action_scaling: bool = True,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        assert (
            dual_clip is None or dual_clip > 1.0
        ), f"Dual-clip PPO parameter should greater than 1.0 but got {dual_clip}"

        super().__init__(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist_fn,
            action_space=action_space,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
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
        self.eps_clip = eps_clip
        self.dual_clip = dual_clip
        self.value_clip = value_clip
        self.norm_adv = advantage_normalization
        self.recompute_adv = recompute_advantage
        self._actor_critic: ActorCritic

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> LogpOldProtocol:
        if self.recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        with torch.no_grad():
            batch.logp_old = self(batch).dist.log_prob(batch.act)
        batch: LogpOldProtocol
        return batch

    # TODO: why does mypy complain?
    def learn(  # type: ignore
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> TPPOTrainingStats:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        split_batch_size = batch_size or -1
        for step in range(repeat):
            if self.recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(split_batch_size, merge_last=True):
                # calculate loss for actor
                dist = self(minibatch).dist
                if self.norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv - mean) / (std + self._eps)  # per-batch norm
                ratio = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * minibatch.adv
                surr2 = ratio.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * minibatch.adv
                if self.dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self.dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
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
                self.optim.zero_grad()
                loss.backward()
                if self.max_grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(),
                        max_norm=self.max_grad_norm,
                    )
                self.optim.step()
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        losses_summary = SequenceSummaryStats.from_sequence(losses)
        clip_losses_summary = SequenceSummaryStats.from_sequence(clip_losses)
        vf_losses_summary = SequenceSummaryStats.from_sequence(vf_losses)
        ent_losses_summary = SequenceSummaryStats.from_sequence(ent_losses)

        return PPOTrainingStats(  # type: ignore[return-value]
            loss=losses_summary,
            clip_loss=clip_losses_summary,
            vf_loss=vf_losses_summary,
            ent_loss=ent_losses_summary,
        )
