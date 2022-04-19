from copy import deepcopy
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from tianshou.data import Batch, to_torch, to_torch_as
from tianshou.policy.modelfree.pg import PGPolicy


class DiscreteCRRPolicy(PGPolicy):
    r"""Implementation of discrete Critic Regularized Regression. arXiv:2006.15134.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the action-value critic (i.e., Q function)
        network. (s -> Q(s, \*))
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param str policy_improvement_mode: type of the weight function f. Possible
        values: "binary"/"exp"/"all". Default to "exp".
    :param float ratio_upper_bound: when policy_improvement_mode is "exp", the value
        of the exp function is upper-bounded by this parameter. Default to 20.
    :param float beta: when policy_improvement_mode is "exp", this is the denominator
        of the exp function. Default to 1.
    :param float min_q_weight: weight for CQL loss/regularizer. Default to 10.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::
        Please refer to :class:`~tianshou.policy.PGPolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        policy_improvement_mode: str = "exp",
        ratio_upper_bound: float = 20.0,
        beta: float = 1.0,
        min_q_weight: float = 10.0,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor,
            optim,
            lambda x: Categorical(logits=x),  # type: ignore
            discount_factor,
            reward_normalization,
            **kwargs,
        )
        self.critic = critic
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._iter = 0
        if self._target:
            self.actor_old = deepcopy(self.actor)
            self.actor_old.eval()
            self.critic_old = deepcopy(self.critic)
            self.critic_old.eval()
        else:
            self.actor_old = self.actor
            self.critic_old = self.critic
        assert policy_improvement_mode in ["exp", "binary", "all"]
        self._policy_improvement_mode = policy_improvement_mode
        self._ratio_upper_bound = ratio_upper_bound
        self._beta = beta
        self._min_q_weight = min_q_weight

    def sync_weight(self) -> None:
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:  # type: ignore
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
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
            target = rew.unsqueeze(1) + self._gamma * expected_target_q
        critic_loss = 0.5 * F.mse_loss(qa_t, target)
        # Actor loss
        act_target, _ = self.actor(batch.obs)
        dist = Categorical(logits=act_target)
        expected_policy_q = (q_t * dist.probs).sum(-1, keepdim=True)
        advantage = qa_t - expected_policy_q
        if self._policy_improvement_mode == "binary":
            actor_loss_coef = (advantage > 0).float()
        elif self._policy_improvement_mode == "exp":
            actor_loss_coef = (
                (advantage / self._beta).exp().clamp(0, self._ratio_upper_bound)
            )
        else:
            actor_loss_coef = 1.0  # effectively behavior cloning
        actor_loss = (-dist.log_prob(act) * actor_loss_coef).mean()
        # CQL loss/regularizer
        min_q_loss = (q_t.logsumexp(1) - qa_t).mean()
        loss = actor_loss + critic_loss + self._min_q_weight * min_q_loss
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {
            "loss": loss.item(),
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
            "loss/cql": min_q_loss.item(),
        }
