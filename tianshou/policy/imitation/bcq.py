import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from typing import Any, Dict, Union, Optional

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as


class BCQPolicy(BasePolicy):
    """Implementation for discrete BCQ algorithm."""

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
        super().__init__(**kwargs)
        # init model
        self.model = model
        self.optim = optim
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self._iter = 0
        # init hparam
        assert (
            0.0 <= discount_factor <= 1.0
        ), "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        assert (
            0.0 <= unlikely_action_threshold < 1.0
        ), "unlikely_action_threshold should be in [0, 1)"
        self._thres = unlikely_action_threshold
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self._n_step = estimation_step
        self._eps = eval_eps
        self._freq = target_update_freq
        self._w_imitation = imitation_logits_penalty

    def train(self, mode: bool = True) -> "BCQPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

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

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        batch = self.compute_nstep_return(
            batch, buffer, indice, self._target_q,
            self._gamma, self._n_step, False)
        return batch

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
        # Use large negative number to mask actions from argmax
        action = (imt * q + (1.0 - imt) * -np.inf).argmax(-1)
        assert len(action.shape) == 1

        # add eps to act
        if not np.isclose(eps, 0.0) and np.random.rand() < eps:
            bsz, action_num = q.shape
            action = np.random.randint(action_num, size=bsz)

        return Batch(logits=q, act=action, state=state)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._iter % self._freq == 0:
            self.sync_weight()

        target_q = batch.returns.flatten()

        (current_q, imt, i), _ = self.model(batch.obs)
        current_q = current_q[np.arange(len(target_q)), batch.act]

        act = to_torch_as(batch.act, target_q)
        q_loss = F.smooth_l1_loss(current_q, target_q)
        i_loss = F.nll_loss(imt, act)  # type: ignore

        loss = q_loss + i_loss + self._w_imitation * i.pow(2).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self._iter += 1
        return {"loss": loss.item()}
