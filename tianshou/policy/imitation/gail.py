from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import (
    ReplayBuffer,
    SequenceSummaryStats,
    to_numpy,
    to_torch,
)
from tianshou.data.types import LogpOldProtocol, RolloutBatchProtocol
from tianshou.policy import PPOPolicy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.policy.modelfree.pg import TDistributionFunction
from tianshou.policy.modelfree.ppo import PPOTrainingStats


@dataclass(kw_only=True)
class GailTrainingStats(PPOTrainingStats):
    disc_loss: SequenceSummaryStats
    acc_pi: SequenceSummaryStats
    acc_exp: SequenceSummaryStats


TGailTrainingStats = TypeVar("TGailTrainingStats", bound=GailTrainingStats)


class GAILPolicy(PPOPolicy[TGailTrainingStats]):
    r"""Implementation of Generative Adversarial Imitation Learning. arXiv:1606.03476.

    :param actor: the actor network following the rules in BasePolicy. (s -> logits)
    :param critic: the critic network. (s -> V(s))
    :param optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :param action_space: env's action space
    :param expert_buffer: the replay buffer containing expert experience.
    :param disc_net: the discriminator network with input dim equals
        state dim plus action dim and output dim equals 1.
    :param disc_optim: the optimizer for the discriminator network.
    :param disc_update_num: the number of discriminator grad steps per model grad step.
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

        Please refer to :class:`~tianshou.policy.PPOPolicy` for more detailed
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
        expert_buffer: ReplayBuffer,
        disc_net: torch.nn.Module,
        disc_optim: torch.optim.Optimizer,
        disc_update_num: int = 4,
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
        super().__init__(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist_fn,
            action_space=action_space,
            eps_clip=eps_clip,
            dual_clip=dual_clip,
            value_clip=value_clip,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
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
        self.disc_net = disc_net
        self.disc_optim = disc_optim
        self.disc_update_num = disc_update_num
        self.expert_buffer = expert_buffer
        self.action_dim = actor.output_dim

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> LogpOldProtocol:
        """Pre-process the data from the provided replay buffer.

        Used in :meth:`update`. Check out :ref:`process_fn` for more information.
        """
        # update reward
        with torch.no_grad():
            batch.rew = to_numpy(-F.logsigmoid(-self.disc(batch)).flatten())
        return super().process_fn(batch, buffer, indices)

    def disc(self, batch: RolloutBatchProtocol) -> torch.Tensor:
        obs = to_torch(batch.obs, device=self.disc_net.device)
        act = to_torch(batch.act, device=self.disc_net.device)
        return self.disc_net(torch.cat([obs, act], dim=1))

    def learn(  # type: ignore
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        **kwargs: Any,
    ) -> TGailTrainingStats:
        # update discriminator
        losses = []
        acc_pis = []
        acc_exps = []
        bsz = len(batch) // self.disc_update_num
        for b in batch.split(bsz, merge_last=True):
            logits_pi = self.disc(b)
            exp_b = self.expert_buffer.sample(bsz)[0]
            logits_exp = self.disc(exp_b)
            loss_pi = -F.logsigmoid(-logits_pi).mean()
            loss_exp = -F.logsigmoid(logits_exp).mean()
            loss_disc = loss_pi + loss_exp
            self.disc_optim.zero_grad()
            loss_disc.backward()
            self.disc_optim.step()
            losses.append(loss_disc.item())
            acc_pis.append((logits_pi < 0).float().mean().item())
            acc_exps.append((logits_exp > 0).float().mean().item())
        # update policy
        ppo_loss_stat = super().learn(batch, batch_size, repeat, **kwargs)

        disc_losses_summary = SequenceSummaryStats.from_sequence(losses)
        acc_pi_summary = SequenceSummaryStats.from_sequence(acc_pis)
        acc_exps_summary = SequenceSummaryStats.from_sequence(acc_exps)

        return GailTrainingStats(  # type: ignore[return-value]
            **ppo_loss_stat.__dict__,
            disc_loss=disc_losses_summary,
            acc_pi=acc_pi_summary,
            acc_exp=acc_exps_summary,
        )
