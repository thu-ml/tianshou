import torch
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Union, Optional

from tianshou.policy import DQNPolicy
from tianshou.data import Batch, ReplayBuffer, to_torch


class DiscreteBCQPolicy(DQNPolicy):
    """Implementation of discrete BCQ algorithm. arXiv:1812.02900."""

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 8000,
        eval_eps: float = 1e-3,
        unlikely_action_threshold: float = 0.3,
        imitation_logits_penalty: float = 1e-2,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, optim, discount_factor, estimation_step,
                         target_update_freq, **kwargs)
        self._iter = 0
        assert (
            0.0 <= unlikely_action_threshold < 1.0
        ), "unlikely_action_threshold should be in [0, 1)"
        self._thres = unlikely_action_threshold
        self._eps = eval_eps
        self._w_imitation = imitation_logits_penalty

    def _target_q(
        self, buffer: ReplayBuffer, indice: np.ndarray
    ) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs_next: s_{t+n}
        # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
        with torch.no_grad():
            act = self(batch, input="obs_next", eps=0.0).act
            target_q = self(
                batch, model="model_old", input="obs_next", eps=0.0
            ).logits
            target_q = target_q[np.arange(len(act)), act]
        return target_q

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        eps: Optional[float] = None,
        **kwargs: Any,
    ) -> Batch:
        if eps is None:
            eps = self._eps
        obs = batch[input]
        (q, imt, i), state = self.model(obs, state=state, info=batch.info)
        imt = imt.exp()
        imt = (imt / imt.max(1, keepdim=True)[0] > self._thres).float()
        # mask actions for argmax
        action = (imt * q + (1.0 - imt) * -np.inf).argmax(-1)

        # add eps to act
        if not np.isclose(eps, 0.0) and np.random.rand() < eps:
            bsz, action_num = q.shape
            action = np.random.randint(action_num, size=bsz)

        return Batch(logits=q, act=action, state=state)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._iter % self._freq == 0:
            self.sync_weight()
        self._iter += 1

        target_q = batch.returns.flatten()
        (current_q, imt, i), _ = self.model(batch.obs)
        current_q = current_q[np.arange(len(target_q)), batch.act]

        act = to_torch(batch.act, dtype=torch.long, device=target_q.device)
        q_loss = F.smooth_l1_loss(current_q, target_q)
        i_loss = F.nll_loss(imt, act)  # type: ignore
        loss = q_loss + i_loss + self._w_imitation * i.pow(2).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return {"loss": loss.item()}
