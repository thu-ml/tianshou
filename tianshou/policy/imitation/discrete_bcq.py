import math
import torch
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Union, Optional

from tianshou.policy import DQNPolicy
from tianshou.data import Batch, ReplayBuffer, to_torch


class DiscreteBCQPolicy(DQNPolicy):
    """Implementation of discrete BCQ algorithm. arXiv:1910.01708.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> q_value)
    :param torch.nn.Module imitator: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> imtation_logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency.
    :param float eval_eps: the epsilon-greedy noise added in evaluation.
    :param float unlikely_action_threshold: the threshold (tau) for unlikely
        actions, as shown in Equ. (17) in the paper. Default to 0.3.
    :param float imitation_logits_penalty: reguralization weight for imitation
        logits. Default to 1e-2.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        imitator: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 8000,
        eval_eps: float = 1e-3,
        unlikely_action_threshold: float = 0.3,
        imitation_logits_penalty: float = 1e-2,
        reward_normalization: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, optim, discount_factor, estimation_step,
                         target_update_freq, reward_normalization, **kwargs)
        assert target_update_freq > 0, "BCQ needs target network setting."
        self.imitator = imitator
        assert 0.0 <= unlikely_action_threshold < 1.0, \
            "unlikely_action_threshold should be in [0, 1)"
        if unlikely_action_threshold > 0:
            self._log_tau = math.log(unlikely_action_threshold)
        else:
            self._log_tau = -np.inf
        assert 0.0 <= eval_eps < 1.0
        self._eps = eval_eps
        self._weight_reg = imitation_logits_penalty

    def train(self, mode: bool = True) -> "DiscreteBCQPolicy":
        self.training = mode
        self.model.train(mode)
        self.imitator.train(mode)
        return self

    def _target_q(self, buffer: ReplayBuffer, indice: np.ndarray) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs_next: s_{t+n}
        # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
        act = self(batch, input="obs_next").act
        target_q, _ = self.model_old(batch.obs_next)
        target_q = target_q[np.arange(len(act)), act]
        return target_q

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        obs = batch[input]
        q_value, state = self.model(obs, state=state, info=batch.info)
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q_value.shape[1]
        imitation_logits, _ = self.imitator(obs, state=state, info=batch.info)

        # mask actions for argmax
        ratio = imitation_logits - imitation_logits.max(dim=-1, keepdim=True).values
        mask = (ratio < self._log_tau).float()
        action = (q_value - np.inf * mask).argmax(dim=-1)

        return Batch(act=action, state=state, q_value=q_value,
                     imitation_logits=imitation_logits)

    def exploration_noise(self, act: np.ndarray, batch: Batch) -> np.ndarray:
        # add eps to act
        if not np.isclose(self._eps, 0.0):
            bsz = len(act)
            mask = np.random.rand(bsz) < self._eps
            act_rand = np.random.randint(self.max_action_num, size=[bsz])
            act[mask] = act_rand[mask]
        return act

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._iter % self._freq == 0:
            self.sync_weight()
        self._iter += 1

        target_q = batch.returns.flatten()
        result = self(batch)
        imitation_logits = result.imitation_logits
        current_q = result.q_value[np.arange(len(target_q)), batch.act]
        act = to_torch(batch.act, dtype=torch.long, device=target_q.device)
        q_loss = F.smooth_l1_loss(current_q, target_q)
        i_loss = F.nll_loss(
            F.log_softmax(imitation_logits, dim=-1), act)  # type: ignore
        reg_loss = imitation_logits.pow(2).mean()
        loss = q_loss + i_loss + self._weight_reg * reg_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return {
            "loss": loss.item(),
            "q_loss": q_loss.item(),
            "i_loss": i_loss.item(),
            "reg_loss": reg_loss.item(),
        }
