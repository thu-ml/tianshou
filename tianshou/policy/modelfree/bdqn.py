from dataclasses import dataclass
from typing import Any, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from sensai.util.helper import mark_used

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch, to_torch_as
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    ActBatchProtocol,
    BatchWithReturnsProtocol,
    ModelOutputBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.policy import DeepQLearning
from tianshou.policy.base import TArrOrActBatch, TLearningRateScheduler
from tianshou.policy.modelfree.dqn import DQNPolicy, DQNTrainingStats
from tianshou.utils.net.common import BranchingNet

mark_used(ActBatchProtocol)


@dataclass(kw_only=True)
class BDQNTrainingStats(DQNTrainingStats):
    pass


TBDQNTrainingStats = TypeVar("TBDQNTrainingStats", bound=BDQNTrainingStats)


class BranchingDuelingQNetworkPolicy(DQNPolicy[BranchingNet]):
    def __init__(
        self,
        *,
        model: BranchingNet,
        action_space: gym.spaces.Discrete,
        observation_space: gym.Space | None = None,
    ):
        """
        :param model: BranchingNet mapping (obs, state, info) -> action_values_BA.
        :param action_space: the environment's action space
        :param observation_space: the environment's observation space.
        """
        super().__init__(
            model=model,
            action_space=action_space,
            observation_space=observation_space,
        )

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> ModelOutputBatchProtocol:
        if model is None:
            model = self.model
        obs = batch.obs
        # TODO: this is very contrived, see also iqn.py
        obs_next_BO = obs.obs if hasattr(obs, "obs") else obs
        action_values_BA, hidden_BH = model(obs_next_BO, state=state, info=batch.info)
        act_B = to_numpy(action_values_BA.argmax(dim=-1))
        result = Batch(logits=action_values_BA, act=act_B, state=hidden_BH)
        return cast(ModelOutputBatchProtocol, result)

    def add_exploration_noise(
        self,
        act: TArrOrActBatch,
        batch: ObsBatchProtocol,
    ) -> TArrOrActBatch:
        # TODO: This looks problematic; the non-array case is silently ignored
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            rand_act = np.random.randint(
                low=0,
                high=self.model.action_per_branch,
                size=(bsz, act.shape[-1]),
            )
            if hasattr(batch.obs, "mask"):
                rand_act += batch.obs.mask
            act[rand_mask] = rand_act[rand_mask]
        return act


class BranchingDuelingQNetwork(DeepQLearning[BranchingDuelingQNetworkPolicy, TBDQNTrainingStats]):
    """Implementation of the Branching Dueling Q-Network algorithm arXiv:1711.08946."""

    def __init__(
        self,
        *,
        policy: BranchingDuelingQNetworkPolicy,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        """
        :param policy: policy
        :param optim: the optimizer for the policy
        :param discount_factor: in [0, 1].
        :param estimation_step: the number of steps to look ahead.
        :param target_update_freq: the target network update frequency (0 if
            you do not use the target network).
        :param reward_normalization: normalize the **returns** to Normal(0, 1).
            TODO: rename to return_normalization?
        :param is_double: use double dqn.
        :param clip_loss_grad: clip the gradient of the loss in accordance
            with nature14236; this amounts to using the Huber loss instead of
            the MSE loss.
        :param lr_scheduler: if not None, will be called in `policy.update()`.
        """
        assert (
            estimation_step == 1
        ), f"N-step bigger than one is not supported by BDQ but got: {estimation_step}"
        super().__init__(
            policy=policy,
            optim=optim,
            discount_factor=discount_factor,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
            is_double=is_double,
            clip_loss_grad=clip_loss_grad,
            lr_scheduler=lr_scheduler,
        )

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # obs_next: s_{t+n}
        result = self.policy(obs_next_batch)
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self.policy(obs_next_batch, model=self.model_old).logits
        else:
            target_q = result.logits
        if self.is_double:
            act = np.expand_dims(self.policy(obs_next_batch).act, -1)
            act = to_torch(act, dtype=torch.long, device=target_q.device)
        else:
            act = target_q.max(-1).indices.unsqueeze(-1)
        return torch.gather(target_q, -1, act).squeeze()

    def _compute_return(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        gamma: float = 0.99,
    ) -> BatchWithReturnsProtocol:
        rew = batch.rew
        with torch.no_grad():
            target_q_torch = self._target_q(buffer, indice)  # (bsz, ?)
        target_q = to_numpy(target_q_torch)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        end_flag = end_flag[indice]
        mean_target_q = np.mean(target_q, -1) if len(target_q.shape) > 1 else target_q
        _target_q = rew + gamma * mean_target_q * (1 - end_flag)
        target_q = np.repeat(_target_q[..., None], self.policy.model.num_branches, axis=-1)
        target_q = np.repeat(target_q[..., None], self.policy.model.action_per_branch, axis=-1)

        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return cast(BatchWithReturnsProtocol, batch)

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithReturnsProtocol:
        """Compute the 1-step return for BDQ targets."""
        return self._compute_return(batch, buffer, indices)

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
        *args: Any,
        **kwargs: Any,
    ) -> TBDQNTrainingStats:
        if self._target and self._iter % self.freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        act = to_torch(batch.act, dtype=torch.long, device=batch.returns.device)
        q = self.policy(batch).logits
        act_mask = torch.zeros_like(q)
        act_mask = act_mask.scatter_(-1, act.unsqueeze(-1), 1)
        act_q = q * act_mask
        returns = batch.returns
        returns = returns * act_mask
        td_error = returns - act_q
        loss = (td_error.pow(2).sum(-1).mean(-1) * weight).mean()
        batch.weight = td_error.sum(-1).sum(-1)  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1

        return BDQNTrainingStats(loss=loss.item())  # type: ignore[return-value]
