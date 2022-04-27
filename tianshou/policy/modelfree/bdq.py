from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import BasePolicy


class BDQPolicy(BasePolicy):
    """Implementation of the branching dueling network arXiv:1711.08946

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool is_double: use double network. Default to True.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.eps = 0.0
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._iter = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self._rew_norm = reward_normalization
        self._is_double = is_double

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode: bool = True) -> "BDQPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def _target_q(self, batch: Batch) -> torch.Tensor:
        result = self(batch, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(batch, model="model_old", input="obs_next").logits
        else:
            target_q = result.logits
        if self._is_double:
            act = self(batch, input="obs_next").act
            return np.squeeze(
                np.take_along_axis(target_q, np.expand_dims(act, -1), -1)
            )
        else:  # Nature DQN, over estimate
            return NotImplementedError

    def _compute_return(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        gamma: float = 0.99,
    ) -> Batch:
        rew = batch.rew
        with torch.no_grad():
            target_q_torch = self._target_q(batch)  # (bsz, ?)
        target_q = to_numpy(target_q_torch)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        end_flag = end_flag[indice]
        _target_q = rew + gamma * np.mean(target_q, -1) * (1 - end_flag)
        target_q = np.repeat(_target_q[..., None], target_q.shape[-1], axis=-1)
        target_q = np.repeat(target_q[..., None], self.max_action_num, axis=-1)

        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Compute the return for BDQ targets.

        """
        batch = self._compute_return(batch, buffer, indices)
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :param float eps: in [0, 1], for epsilon-greedy exploration method.

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        logits, hidden = model(obs_next, state=state, info=batch.info)
        if not hasattr(self, "max_action_num"):
            self.max_action_num = logits.shape[-1]
        act = to_numpy(logits.max(dim=-1)[1].squeeze())
        if len(act.shape) == 1:
            act = np.expand_dims(act, 0)
        return Batch(logits=logits, act=act, state=hidden)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        act = torch.tensor(batch.act).to(batch.returns.get_device())
        q = self(batch).logits
        act_mask = torch.zeros_like(q)
        act_mask = act_mask.scatter_(-1, act.unsqueeze(-1), 1)
        act_q = q * act_mask
        returns = batch.returns
        returns = returns * act_mask
        td_error = (returns - act_q)
        loss = (td_error.pow(2).sum(-1).mean(-1) * weight).mean()
        batch.weight = td_error.sum(-1).sum(-1)  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            if len(act.shape) == 1:
                act = np.expand_dims(act, 0)
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            rand_act = np.random.randint(
                0, self.max_action_num, (bsz, act.shape[-1])
            )  # [0, 1]
            if hasattr(batch.obs, "mask"):
                rand_act += batch.obs.mask
            act[rand_mask] = rand_act[rand_mask]
        return act

    def map_action(self, act: Union[Batch, np.ndarray]) -> Union[Batch, np.ndarray]:
        return act
