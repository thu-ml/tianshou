import torch
import numpy as np
from numba import njit
from typing import Any, Dict, Union, Optional, Tuple

from tianshou.policy import DQNPolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy


class C51Policy(DQNPolicy):
    """Implementation of Categorical Deep Q-network. arXiv:1707.06887.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int num_atoms: the number of atoms in the support set of the
        value distribution, defaults to 51.
    :param float v_min: the value of the smallest atom in the support set,
        defaults to -10.0.
    :param float v_max: the value of the largest atom in the support set,
        defaults to -10.0.
    :param int estimation_step: greater than 1, the number of steps to look
        ahead.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network).
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.DQNPolicy` for more detailed
         explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, optim, discount_factor,
                         estimation_step, target_update_freq,
                         reward_normalization, **kwargs)
        self._num_atoms = num_atoms
        self._v_min = v_min
        self._v_max = v_max
        self.device = model.device
        self.support = torch.linspace(self._v_min, self._v_max, 
                                      self._num_atoms, device=self.device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

    @staticmethod
    def prepare_n_step(
        batch: Batch,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> Batch:
        """ Modify the obs_next, done and rew in batch for computing n-step
        return in Distributional Q-learning based algorithms.

        :param batch: a data batch, which is equal to buffer[indice].
        :type batch: :class:`~tianshou.data.Batch`
        :param buffer: a data buffer which contains several full-episode data
            chronologically.
        :type buffer: :class:`~tianshou.data.ReplayBuffer`
        :param indice: sampled timestep.
        :type indice: numpy.ndarray
        :param float gamma: the discount factor, should be in [0, 1], defaults
            to 0.99.
        :param int n_step: the number of estimation step, should be an int
            greater than 0, defaults to 1.
        :param bool rew_norm: normalize the reward to Normal(0, 1), defaults
            to False.

        :return: a Batch with modified obs_next, done and rew.
        """
        buf_len = len(buffer)
        if rew_norm:
            bfr = buffer.rew[: min(buf_len, 1000)]  # avoid large buffer
            mean, std = bfr.mean(), bfr.std()
            if np.isclose(std, 0, 1e-2):
                mean, std = 0.0, 1.0
        else:
            mean, std = 0.0, 1.0
        buffer_n = buffer[(indice + n_step - 1) % buf_len]
        batch.obs_next = buffer_n.obs_next
        rew_n, done_n = _nstep_batch(buffer.rew, buffer.done,
                                     indice, gamma, n_step, buf_len, mean, std)
        batch.rew = rew_n
        batch.done = done_n
        return batch

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Prepare the batch for calculating the n-step return.

        More details can be found at
        :meth:`~tianshou.policy.C51Policy.prepare_n_step`.
        """
        batch = self.prepare_n_step(
            batch, buffer, indice,
            self._gamma, self._n_step, self._rew_norm)
        return batch

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.DQNPolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        dist, h = model(obs_, state=state, info=batch.info)
        q = (dist * self.support).sum(2)
        act: np.ndarray = to_numpy(q.max(dim=1)[1])
        if hasattr(obs, "mask"):
            # some of actions are masked, they cannot be selected
            q_: np.ndarray = to_numpy(q)
            q_[~obs.mask] = -np.inf
            act = q_.argmax(axis=1)
        # add eps to act in training or testing phase
        if not self.updating and not np.isclose(self.eps, 0.0):
            for i in range(len(q)):
                if np.random.rand() < self.eps:
                    q_ = np.random.rand(*q[i].shape)
                    if hasattr(obs, "mask"):
                        q_[~obs.mask[i]] = -np.inf
                    act[i] = q_.argmax()
        return Batch(logits=dist, act=act, state=h)

    def _target_dist(
            self, batch: Batch
    ) -> torch.Tensor:
        if self._target:
            a = self(batch, input="obs_next").act
            next_dist = self(
                batch, model="model_old", input="obs_next"
            ).logits
        else:
            next_b = self(batch, input="obs_next")
            a = next_b.act
            next_dist = next_b.logits
        batch_size = len(a)
        next_dist = next_dist[np.arange(batch_size), a, :]

        reward = to_torch_as(batch.rew, next_dist).unsqueeze(1)
        done = to_torch_as(batch.done, next_dist).unsqueeze(1)

        # Compute the projection of bellman update Tz onto the support z.
        target_support = reward + (self._gamma ** self._n_step
                                  ) * (1.0 - done) * self.support.unsqueeze(0)
        target_support = target_support.clamp(self._v_min, self._v_max)

        # An amazing trick for calculating the projection gracefully.
        # ref: https://github.com/ShangtongZhang/DeepRL
        target_dist = (1 - (target_support.unsqueeze(1) - 
                            self.support.view(1, -1, 1)).abs() / self.delta_z
                       ).clamp(0, 1) * next_dist.unsqueeze(1)
        target_dist = target_dist.sum(-1)
        return target_dist

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._cnt % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        with torch.no_grad():
            target_dist = self._target_dist(batch)
        curr_dist = self(batch).logits
        act = batch.act
        curr_dist = curr_dist[np.arange(len(act)), act, :]
        cross_entropy = - (target_dist * torch.log(curr_dist + 1e-8)).sum(1)
        loss = (cross_entropy * weight).mean()
        batch.weight = cross_entropy.detach()  # prio-buffer
        loss.backward()
        self.optim.step()
        self._cnt += 1
        return {"loss": loss.item()}


@njit
def _nstep_batch(
    rew: np.ndarray,
    done: np.ndarray,
    indice: np.ndarray,
    gamma: float,
    n_step: int,
    buf_len: int,
    mean: float,
    std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rew_n = np.zeros(indice.shape)
    done_n = done[indice]
    for n in range(n_step - 1, -1, -1):
        now = (indice + n) % buf_len
        done_t = done[now]
        done_n = np.bitwise_or(done_n, done_t)
        rew_n = (rew[now] - mean) / std + (1.0 - done_t) * gamma * rew_n
    return rew_n, done_n
