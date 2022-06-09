from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, to_numpy
from tianshou.policy import QRDQNPolicy


class IQNPolicy(QRDQNPolicy):
    """Implementation of Implicit Quantile Network. arXiv:1806.06923.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int sample_size: the number of samples for policy evaluation.
        Default to 32.
    :param int online_sample_size: the number of samples for online model
        in training. Default to 8.
    :param int target_sample_size: the number of samples for target model
        in training. Default to 8.
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network).
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
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
        sample_size: int = 32,
        online_sample_size: int = 8,
        target_sample_size: int = 8,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model, optim, discount_factor, sample_size, estimation_step,
            target_update_freq, reward_normalization, **kwargs
        )
        assert sample_size > 1, "sample_size should be greater than 1"
        assert online_sample_size > 1, "online_sample_size should be greater than 1"
        assert target_sample_size > 1, "target_sample_size should be greater than 1"
        self._sample_size = sample_size  # for policy eval
        self._online_sample_size = online_sample_size
        self._target_sample_size = target_sample_size

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        if model == "model_old":
            sample_size = self._target_sample_size
        elif self.training:
            sample_size = self._online_sample_size
        else:
            sample_size = self._sample_size
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        (logits, taus), hidden = model(
            obs_next, sample_size=sample_size, state=state, info=batch.info
        )
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        return Batch(logits=logits, act=act, state=hidden, taus=taus)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        action_batch = self(batch)
        curr_dist, taus = action_batch.logits, action_batch.taus
        act = batch.act
        curr_dist = curr_dist[np.arange(len(act)), act, :].unsqueeze(2)
        target_dist = batch.returns.unsqueeze(1)
        # calculate each element's difference between curr_dist and target_dist
        dist_diff = F.smooth_l1_loss(target_dist, curr_dist, reduction="none")
        huber_loss = (
            dist_diff *
            (taus.unsqueeze(2) -
             (target_dist - curr_dist).detach().le(0.).float()).abs()
        ).sum(-1).mean(1)
        loss = (huber_loss * weight).mean()
        # ref: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/
        # blob/master/fqf_iqn_qrdqn/agent/qrdqn_agent.py L130
        batch.weight = dist_diff.detach().abs().sum(-1).mean(1)  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}
