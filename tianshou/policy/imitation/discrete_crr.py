from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from tianshou.data import ReplayBuffer, to_torch, to_torch_as
from tianshou.data.types import BatchWithReturnsProtocol, RolloutBatchProtocol
from tianshou.policy.base import OfflineAlgorithm, TLearningRateScheduler
from tianshou.policy.modelfree.pg import (
    DiscountedReturnComputation,
    DiscreteActorPolicy,
    PGTrainingStats,
)
from tianshou.utils.net.discrete import Critic


@dataclass
class DiscreteCRRTrainingStats(PGTrainingStats):
    actor_loss: float
    critic_loss: float
    cql_loss: float


TDiscreteCRRTrainingStats = TypeVar("TDiscreteCRRTrainingStats", bound=DiscreteCRRTrainingStats)


class DiscreteCRR(OfflineAlgorithm[DiscreteActorPolicy, TDiscreteCRRTrainingStats]):
    r"""Implementation of discrete Critic Regularized Regression. arXiv:2006.15134."""

    def __init__(
        self,
        *,
        policy: DiscreteActorPolicy,
        critic: torch.nn.Module | Critic,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        policy_improvement_mode: Literal["exp", "binary", "all"] = "exp",
        ratio_upper_bound: float = 20.0,
        beta: float = 1.0,
        min_q_weight: float = 10.0,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        r"""
        :param policy: the policy
        :param critic: the action-value critic (i.e., Q function)
            network. (s -> Q(s, \*))
        :param optim: the optimizer for the policy's actor and the critic networks.
        :param discount_factor: in [0, 1].
        :param str policy_improvement_mode: type of the weight function f. Possible
            values: "binary"/"exp"/"all".
        :param ratio_upper_bound: when policy_improvement_mode is "exp", the value
            of the exp function is upper-bounded by this parameter.
        :param beta: when policy_improvement_mode is "exp", this is the denominator
            of the exp function.
        :param min_q_weight: weight for CQL loss/regularizer. Default to 10.
        :param target_update_freq: the target network update frequency (0 if
            you do not use the target network).
        :param reward_normalization: if True, will normalize the *returns*
            by subtracting the running mean and dividing by the running standard deviation.
            Can be detrimental to performance!
        :param lr_scheduler: if not None, will be called in `policy.update()`.
        """
        super().__init__(
            policy=policy,
            lr_scheduler=lr_scheduler,
        )
        self.optim = optim
        self.discounted_return_computation = DiscountedReturnComputation(
            discount_factor=discount_factor,
            reward_normalization=reward_normalization,
        )
        self.critic = critic
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._iter = 0
        if self._target:
            self.actor_old = deepcopy(self.policy.actor)
            self.actor_old.eval()
            self.critic_old = deepcopy(self.critic)
            self.critic_old.eval()
        else:
            self.actor_old = self.actor
            self.critic_old = self.critic
        self._policy_improvement_mode = policy_improvement_mode
        self._ratio_upper_bound = ratio_upper_bound
        self._beta = beta
        self._min_q_weight = min_q_weight

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithReturnsProtocol:
        return self.discounted_return_computation.add_discounted_returns(
            batch,
            buffer,
            indices,
        )

    def _update_lagged_network_weights(self) -> None:
        self.actor_old.load_state_dict(self.policy.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def _update_with_batch(  # type: ignore
        self,
        batch: RolloutBatchProtocol,
        *args: Any,
        **kwargs: Any,
    ) -> TDiscreteCRRTrainingStats:
        if self._target and self._iter % self._freq == 0:
            self._update_lagged_network_weights()
        self.optim.zero_grad()
        q_t = self.critic(batch.obs)
        act = to_torch(batch.act, dtype=torch.long, device=q_t.device)
        qa_t = q_t.gather(1, act.unsqueeze(1))
        # Critic loss
        with torch.no_grad():
            target_a_t, _ = self.actor_old(batch.obs_next)
            target_m = Categorical(logits=target_a_t)
            q_t_target = self.critic_old(batch.obs_next)
            rew = to_torch_as(batch.rew, q_t_target)
            expected_target_q = (q_t_target * target_m.probs).sum(-1, keepdim=True)
            expected_target_q[batch.done > 0] = 0.0
            target = rew.unsqueeze(1) + self.discounted_return_computation.gamma * expected_target_q
        critic_loss = 0.5 * F.mse_loss(qa_t, target)
        # Actor loss
        act_target, _ = self.policy.actor(batch.obs)
        dist = Categorical(logits=act_target)
        expected_policy_q = (q_t * dist.probs).sum(-1, keepdim=True)
        advantage = qa_t - expected_policy_q
        if self._policy_improvement_mode == "binary":
            actor_loss_coef = (advantage > 0).float()
        elif self._policy_improvement_mode == "exp":
            actor_loss_coef = (advantage / self._beta).exp().clamp(0, self._ratio_upper_bound)
        else:
            actor_loss_coef = 1.0  # effectively behavior cloning
        actor_loss = (-dist.log_prob(act) * actor_loss_coef).mean()
        # CQL loss/regularizer
        min_q_loss = (q_t.logsumexp(1) - qa_t).mean()
        loss = actor_loss + critic_loss + self._min_q_weight * min_q_loss
        loss.backward()
        self.optim.step()
        self._iter += 1

        return DiscreteCRRTrainingStats(  # type: ignore[return-value]
            # TODO: Type is wrong
            loss=loss.item(),
            actor_loss=actor_loss.item(),
            critic_loss=critic_loss.item(),
            cql_loss=min_q_loss.item(),
        )
