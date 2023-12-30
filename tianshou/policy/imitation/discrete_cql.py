from dataclasses import dataclass
from typing import Any, TypeVar

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import to_torch
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import QRDQNPolicy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.policy.modelfree.qrdqn import QRDQNTrainingStats


@dataclass(kw_only=True)
class DiscreteCQLTrainingStats(QRDQNTrainingStats):
    cql_loss: float
    qr_loss: float


TDiscreteCQLTrainingStats = TypeVar("TDiscreteCQLTrainingStats", bound=DiscreteCQLTrainingStats)


class DiscreteCQLPolicy(QRDQNPolicy[TDiscreteCQLTrainingStats]):
    """Implementation of discrete Conservative Q-Learning algorithm. arXiv:2006.04779.

    :param model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param optim: a torch.optim for optimizing the model.
    :param action_space: Env's action space.
    :param min_q_weight: the weight for the cql loss.
    :param discount_factor: in [0, 1].
    :param num_quantiles: the number of quantile midpoints in the inverse
        cumulative distribution function of the value.
    :param estimation_step: the number of steps to look ahead.
    :param target_update_freq: the target network update frequency (0 if
        you do not use the target network).
    :param reward_normalization: normalize the **returns** to Normal(0, 1).
        TODO: rename to return_normalization?
    :param is_double: use double dqn.
    :param clip_loss_grad: clip the gradient of the loss in accordance
        with nature14236; this amounts to using the Huber loss instead of
        the MSE loss.
    :param observation_space: Env's observation space.
    :param lr_scheduler: if not None, will be called in `policy.update()`.

    .. seealso::
        Please refer to :class:`~tianshou.policy.QRDQNPolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        action_space: gym.spaces.Discrete,
        min_q_weight: float = 10.0,
        discount_factor: float = 0.99,
        num_quantiles: int = 200,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        observation_space: gym.Space | None = None,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            model=model,
            optim=optim,
            action_space=action_space,
            discount_factor=discount_factor,
            num_quantiles=num_quantiles,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
            is_double=is_double,
            clip_loss_grad=clip_loss_grad,
            observation_space=observation_space,
            lr_scheduler=lr_scheduler,
        )
        self.min_q_weight = min_q_weight

    def learn(
        self,
        batch: RolloutBatchProtocol,
        *args: Any,
        **kwargs: Any,
    ) -> TDiscreteCQLTrainingStats:
        if self._target and self._iter % self.freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        all_dist = self(batch).logits
        act = to_torch(batch.act, dtype=torch.long, device=all_dist.device)
        curr_dist = all_dist[np.arange(len(act)), act, :].unsqueeze(2)
        target_dist = batch.returns.unsqueeze(1)
        # calculate each element's difference between curr_dist and target_dist
        dist_diff = F.smooth_l1_loss(target_dist, curr_dist, reduction="none")
        huber_loss = (
            (dist_diff * (self.tau_hat - (target_dist - curr_dist).detach().le(0.0).float()).abs())
            .sum(-1)
            .mean(1)
        )
        qr_loss = (huber_loss * weight).mean()
        # ref: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/
        # blob/master/fqf_iqn_qrdqn/agent/qrdqn_agent.py L130
        batch.weight = dist_diff.detach().abs().sum(-1).mean(1)  # prio-buffer
        # add CQL loss
        q = self.compute_q_value(all_dist, None)
        dataset_expec = q.gather(1, act.unsqueeze(1)).mean()
        negative_sampling = q.logsumexp(1).mean()
        min_q_loss = negative_sampling - dataset_expec
        loss = qr_loss + min_q_loss * self.min_q_weight
        loss.backward()
        self.optim.step()
        self._iter += 1

        return DiscreteCQLTrainingStats(  # type: ignore[return-value]
            loss=loss.item(),
            qr_loss=qr_loss.item(),
            cql_loss=min_q_loss.item(),
        )
