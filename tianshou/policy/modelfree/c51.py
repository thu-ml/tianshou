from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import DQNPolicy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.policy.modelfree.dqn import DQNTrainingStats


@dataclass(kw_only=True)
class C51TrainingStats(DQNTrainingStats):
    pass


TC51TrainingStats = TypeVar("TC51TrainingStats", bound=C51TrainingStats)


class C51Policy(DQNPolicy[TC51TrainingStats], Generic[TC51TrainingStats]):
    """Implementation of Categorical Deep Q-Network. arXiv:1707.06887.

    :param model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param optim: a torch.optim for optimizing the model.
    :param discount_factor: in [0, 1].
    :param num_atoms: the number of atoms in the support set of the
        value distribution. Default to 51.
    :param v_min: the value of the smallest atom in the support set.
        Default to -10.0.
    :param v_max: the value of the largest atom in the support set.
        Default to 10.0.
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
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        observation_space: gym.Space | None = None,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        assert num_atoms > 1, f"num_atoms should be greater than 1 but got: {num_atoms}"
        assert v_min < v_max, f"v_max should be larger than v_min, but got {v_min=} and {v_max=}"
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
        self._num_atoms = num_atoms
        self._v_min = v_min
        self._v_max = v_max
        self.support = torch.nn.Parameter(
            torch.linspace(self._v_min, self._v_max, self._num_atoms),
            requires_grad=False,
        )
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        return self.support.repeat(len(indices), 1)  # shape: [bsz, num_atoms]

    def compute_q_value(self, logits: torch.Tensor, mask: np.ndarray | None) -> torch.Tensor:
        return super().compute_q_value((logits * self.support).sum(2), mask)

    def _target_dist(self, batch: RolloutBatchProtocol) -> torch.Tensor:
        obs_next_batch = Batch(obs=batch.obs_next, info=[None] * len(batch))
        if self._target:
            act = self(obs_next_batch).act
            next_dist = self(obs_next_batch, model="model_old").logits
        else:
            next_batch = self(obs_next_batch)
            act = next_batch.act
            next_dist = next_batch.logits
        next_dist = next_dist[np.arange(len(act)), act, :]
        target_support = batch.returns.clamp(self._v_min, self._v_max)
        # An amazing trick for calculating the projection gracefully.
        # ref: https://github.com/ShangtongZhang/DeepRL
        target_dist = (
            1 - (target_support.unsqueeze(1) - self.support.view(1, -1, 1)).abs() / self.delta_z
        ).clamp(0, 1) * next_dist.unsqueeze(1)
        return target_dist.sum(-1)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TC51TrainingStats:
        if self._target and self._iter % self.freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        with torch.no_grad():
            target_dist = self._target_dist(batch)
        weight = batch.pop("weight", 1.0)
        curr_dist = self(batch).logits
        act = batch.act
        curr_dist = curr_dist[np.arange(len(act)), act, :]
        cross_entropy = -(target_dist * torch.log(curr_dist + 1e-8)).sum(1)
        loss = (cross_entropy * weight).mean()
        # ref: https://github.com/Kaixhin/Rainbow/blob/master/agent.py L94-100
        batch.weight = cross_entropy.detach()  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1

        return C51TrainingStats(loss=loss.item())  # type: ignore[return-value]
