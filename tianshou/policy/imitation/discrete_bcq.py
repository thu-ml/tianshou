import math
from typing import Any, Self, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.data.types import ImitationBatchProtocol, RolloutBatchProtocol
from tianshou.policy import DQNPolicy
from tianshou.policy.base import TLearningRateScheduler


class DiscreteBCQPolicy(DQNPolicy):
    """Implementation of discrete BCQ algorithm. arXiv:1910.01708.

    :param model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> q_value)
    :param imitator: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> imitation_logits)
    :param optim: a torch.optim for optimizing the model.
    :param discount_factor: in [0, 1].
    :param estimation_step: the number of steps to look ahead
    :param target_update_freq: the target network update frequency.
    :param eval_eps: the epsilon-greedy noise added in evaluation.
    :param unlikely_action_threshold: the threshold (tau) for unlikely
        actions, as shown in Equ. (17) in the paper.
    :param imitation_logits_penalty: regularization weight for imitation
        logits.
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

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        imitator: torch.nn.Module,
        optim: torch.optim.Optimizer,
        action_space: gym.spaces.Discrete,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 8000,
        eval_eps: float = 1e-3,
        unlikely_action_threshold: float = 0.3,
        imitation_logits_penalty: float = 1e-2,
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
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
            is_double=is_double,
            clip_loss_grad=clip_loss_grad,
            observation_space=observation_space,
            lr_scheduler=lr_scheduler,
        )
        assert (
            target_update_freq > 0
        ), f"BCQ needs target_update_freq>0 but got: {target_update_freq}."
        self.imitator = imitator
        assert (
            0.0 <= unlikely_action_threshold < 1.0
        ), f"unlikely_action_threshold should be in [0, 1) but got: {unlikely_action_threshold}"
        if unlikely_action_threshold > 0:
            self._log_tau = math.log(unlikely_action_threshold)
        else:
            self._log_tau = -np.inf
        assert 0.0 <= eval_eps < 1.0
        self.eps = eval_eps
        self._weight_reg = imitation_logits_penalty

    def train(self, mode: bool = True) -> Self:
        self.training = mode
        self.model.train(mode)
        self.imitator.train(mode)
        return self

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
        act = self(batch, input="obs_next").act
        target_q, _ = self.model_old(batch.obs_next)
        return target_q[np.arange(len(act)), act]

    def forward(  # type: ignore
        self,
        batch: RolloutBatchProtocol,
        state: dict | Batch | np.ndarray | None = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> ImitationBatchProtocol:
        # TODO: Liskov substitution principle is violated here, the superclass
        #  produces a batch with the field logits, but this one doesn't.
        #  Should be fixed in the future!
        obs = batch[input]
        q_value, state = self.model(obs, state=state, info=batch.info)
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q_value.shape[1]
        imitation_logits, _ = self.imitator(obs, state=state, info=batch.info)

        # mask actions for argmax
        ratio = imitation_logits - imitation_logits.max(dim=-1, keepdim=True).values
        mask = (ratio < self._log_tau).float()
        act = (q_value - np.inf * mask).argmax(dim=-1)

        result = Batch(act=act, state=state, q_value=q_value, imitation_logits=imitation_logits)
        return cast(ImitationBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> dict[str, float]:
        if self._iter % self.freq == 0:
            self.sync_weight()
        self._iter += 1

        target_q = batch.returns.flatten()
        result = self(batch)
        imitation_logits = result.imitation_logits
        current_q = result.q_value[np.arange(len(target_q)), batch.act]
        act = to_torch(batch.act, dtype=torch.long, device=target_q.device)
        q_loss = F.smooth_l1_loss(current_q, target_q)
        i_loss = F.nll_loss(F.log_softmax(imitation_logits, dim=-1), act)
        reg_loss = imitation_logits.pow(2).mean()
        loss = q_loss + i_loss + self._weight_reg * reg_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return {
            "loss": loss.item(),
            "loss/q": q_loss.item(),
            "loss/i": i_loss.item(),
            "loss/reg": reg_loss.item(),
        }
