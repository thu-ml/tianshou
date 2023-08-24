from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import to_torch
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import QRDQNPolicy


class DiscreteCQLPolicy(QRDQNPolicy):
    """Implementation of discrete Conservative Q-Learning algorithm. arXiv:2006.04779.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int num_quantiles: the number of quantile midpoints in the inverse
        cumulative distribution function of the value. Default to 200.
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network).
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param float min_q_weight: the weight for the cql loss.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::
        Please refer to :class:`~tianshou.policy.QRDQNPolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        num_quantiles: int = 200,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        min_q_weight: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            optim,
            discount_factor,
            num_quantiles,
            estimation_step,
            target_update_freq,
            reward_normalization,
            **kwargs,
        )
        self._min_q_weight = min_q_weight

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> dict[str, float]:
        if self._target and self._iter % self._freq == 0:
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
        loss = qr_loss + min_q_loss * self._min_q_weight
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {
            "loss": loss.item(),
            "loss/qr": qr_loss.item(),
            "loss/cql": min_q_loss.item(),
        }
