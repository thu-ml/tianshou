from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, to_numpy
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    ObsBatchProtocol,
    QuantileRegressionBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.policy import QRDQNPolicy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.policy.modelfree.qrdqn import QRDQNTrainingStats


@dataclass(kw_only=True)
class IQNTrainingStats(QRDQNTrainingStats):
    pass


TIQNTrainingStats = TypeVar("TIQNTrainingStats", bound=IQNTrainingStats)


class IQNPolicy(QRDQNPolicy[TIQNTrainingStats]):
    """Implementation of Implicit Quantile Network. arXiv:1806.06923.

    :param model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param optim: a torch.optim for optimizing the model.
    :param discount_factor: in [0, 1].
    :param sample_size: the number of samples for policy evaluation.
    :param online_sample_size: the number of samples for online model
        in training.
    :param target_sample_size: the number of samples for target model
        in training.
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

        Please refer to :class:`~tianshou.policy.QRDQNPolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        action_space: gym.spaces.Discrete,
        discount_factor: float = 0.99,
        sample_size: int = 32,
        online_sample_size: int = 8,
        target_sample_size: int = 8,
        num_quantiles: int = 200,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        observation_space: gym.Space | None = None,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        assert sample_size > 1, f"sample_size should be greater than 1 but got: {sample_size}"
        assert (
            online_sample_size > 1
        ), f"online_sample_size should be greater than 1 but got: {online_sample_size}"
        assert (
            target_sample_size > 1
        ), f"target_sample_size should be greater than 1 but got: {target_sample_size}"
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
        self.sample_size = sample_size  # for policy eval
        self.online_sample_size = online_sample_size
        self.target_sample_size = target_sample_size

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        model: Literal["model", "model_old"] = "model",
        **kwargs: Any,
    ) -> QuantileRegressionBatchProtocol:
        if model == "model_old":
            sample_size = self.target_sample_size
        elif self.training:
            sample_size = self.online_sample_size
        else:
            sample_size = self.sample_size
        model = getattr(self, model)
        obs = batch.obs
        # TODO: this seems very contrived!
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        (logits, taus), hidden = model(
            obs_next,
            sample_size=sample_size,
            state=state,
            info=batch.info,
        )
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if self.max_action_num is None:  # type: ignore
            # TODO: see same thing in DQNPolicy!
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        result = Batch(logits=logits, act=act, state=hidden, taus=taus)
        return cast(QuantileRegressionBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TIQNTrainingStats:
        if self._target and self._iter % self.freq == 0:
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
            (
                dist_diff
                * (taus.unsqueeze(2) - (target_dist - curr_dist).detach().le(0.0).float()).abs()
            )
            .sum(-1)
            .mean(1)
        )
        loss = (huber_loss * weight).mean()
        # ref: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/
        # blob/master/fqf_iqn_qrdqn/agent/qrdqn_agent.py L130
        batch.weight = dist_diff.detach().abs().sum(-1).mean(1)  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1

        return IQNTrainingStats(loss=loss.item())  # type: ignore[return-value]
