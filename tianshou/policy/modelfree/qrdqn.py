import warnings
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import DQNPolicy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.policy.modelfree.dqn import DQNTrainingStats


@dataclass(kw_only=True)
class QRDQNTrainingStats(DQNTrainingStats):
    pass


TQRDQNTrainingStats = TypeVar("TQRDQNTrainingStats", bound=QRDQNTrainingStats)


class QRDQNPolicy(DQNPolicy[TQRDQNTrainingStats], Generic[TQRDQNTrainingStats]):
    """Implementation of Quantile Regression Deep Q-Network. arXiv:1710.10044.

    :param model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param optim: a torch.optim for optimizing the model.
    :param action_space: Env's action space.
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

        Please refer to :class:`~tianshou.policy.DQNPolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        action_space: gym.spaces.Discrete,
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
        assert num_quantiles > 1, f"num_quantiles should be greater than 1 but got: {num_quantiles}"
        super().__init__(
            model=model,
            optim=optim,
            action_space=action_space,
            discount_factor=discount_factor,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
            is_double=is_double,
            clip_loss_grad=clip_loss_grad,
            observation_space=observation_space,
            lr_scheduler=lr_scheduler,
        )
        self.num_quantiles = num_quantiles
        tau = torch.linspace(0, 1, self.num_quantiles + 1)
        self.tau_hat = torch.nn.Parameter(
            ((tau[:-1] + tau[1:]) / 2).view(1, -1, 1),
            requires_grad=False,
        )
        warnings.filterwarnings("ignore", message="Using a target size")

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # obs_next: s_{t+n}
        if self._target:
            act = self(obs_next_batch).act
            next_dist = self(obs_next_batch, model="model_old").logits
        else:
            next_batch = self(obs_next_batch)
            act = next_batch.act
            next_dist = next_batch.logits
        return next_dist[np.arange(len(act)), act, :]

    def compute_q_value(self, logits: torch.Tensor, mask: np.ndarray | None) -> torch.Tensor:
        return super().compute_q_value(logits.mean(2), mask)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TQRDQNTrainingStats:
        if self._target and self._iter % self.freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        curr_dist = self(batch).logits
        act = batch.act
        curr_dist = curr_dist[np.arange(len(act)), act, :].unsqueeze(2)
        target_dist = batch.returns.unsqueeze(1)
        # calculate each element's difference between curr_dist and target_dist
        dist_diff = F.smooth_l1_loss(target_dist, curr_dist, reduction="none")
        huber_loss = (
            (dist_diff * (self.tau_hat - (target_dist - curr_dist).detach().le(0.0).float()).abs())
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

        return QRDQNTrainingStats(loss=loss.item())  # type: ignore[return-value]
