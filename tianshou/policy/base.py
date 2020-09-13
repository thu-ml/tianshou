import gym
import torch
import numpy as np
from torch import nn
from numba import njit
from abc import ABC, abstractmethod
from typing import Any, List, Union, Mapping, Optional, Callable

from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy


class BasePolicy(ABC, nn.Module):
    """The base class for any RL policy.

    Tianshou aims to modularizing RL algorithms. It comes into several classes
    of policies in Tianshou. All of the policy classes must inherit
    :class:`~tianshou.policy.BasePolicy`.

    A policy class typically has four parts:

    * :meth:`~tianshou.policy.BasePolicy.__init__`: initialize the policy, \
        including coping the target network and so on;
    * :meth:`~tianshou.policy.BasePolicy.forward`: compute action with given \
        observation;
    * :meth:`~tianshou.policy.BasePolicy.process_fn`: pre-process data from \
        the replay buffer (this function can interact with replay buffer);
    * :meth:`~tianshou.policy.BasePolicy.learn`: update policy with a given \
        batch of data.

    Most of the policy needs a neural network to predict the action and an
    optimizer to optimize the policy. The rules of self-defined networks are:

    1. Input: observation "obs" (may be a ``numpy.ndarray``, a \
    ``torch.Tensor``, a dict or any others), hidden state "state" (for RNN \
    usage), and other information "info" provided by the environment.
    2. Output: some "logits", the next hidden state "state", and the \
    intermediate result during policy forwarding procedure "policy". The \
    "logits" could be a tuple instead of a ``torch.Tensor``. It depends on how\
    the policy process the network output. For example, in PPO, the return of \
    the network might be ``(mu, sigma), state`` for Gaussian policy. The \
    "policy" can be a Batch of torch.Tensor or other things, which will be \
    stored in the replay buffer, and can be accessed in the policy update \
    process (e.g. in "policy.learn()", the "batch.policy" is what you need).

    Since :class:`~tianshou.policy.BasePolicy` inherits ``torch.nn.Module``,
    you can use :class:`~tianshou.policy.BasePolicy` almost the same as
    ``torch.nn.Module``, for instance, loading and saving the model:
    ::

        torch.save(policy.state_dict(), "policy.pth")
        policy.load_state_dict(torch.load("policy.pth"))
    """

    def __init__(
        self,
        observation_space: gym.Space = None,
        action_space: gym.Space = None
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.agent_id = 0
        self._compile()

    def set_agent_id(self, agent_id: int) -> None:
        """Set self.agent_id = agent_id, for MARL."""
        self.agent_id = agent_id

    @abstractmethod
    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which MUST have the following\
        keys:

            * ``act`` an numpy.ndarray or a torch.Tensor, the action over \
                given batch data.
            * ``state`` a dict, an numpy.ndarray or a torch.Tensor, the \
                internal state of the policy, ``None`` as default.

        Other keys are user-defined. It depends on the algorithm. For example,
        ::

            # some code
            return Batch(logits=..., act=..., state=None, dist=...)

        The keyword ``policy`` is reserved and the corresponding data will be
        stored into the replay buffer. For instance,
        ::

            # some code
            return Batch(..., policy=Batch(log_prob=dist.log_prob(act)))
            # and in the sampled data batch, you can directly use
            # batch.policy.log_prob to get your data.
        """
        pass

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Pre-process the data from the provided replay buffer.

        Used in :meth:`update`. Check out :ref:`process_fn` for more
        information.
        """
        return batch

    @abstractmethod
    def learn(
        self, batch: Batch, **kwargs: Any
    ) -> Mapping[str, Union[float, List[float]]]:
        """Update policy with a given batch of data.

        :return: A dict which includes loss and its corresponding label.

        .. warning::

            If you use ``torch.distributions.Normal`` and
            ``torch.distributions.Categorical`` to calculate the log_prob,
            please be careful about the shape: Categorical distribution gives
            "[batch_size]" shape while Normal distribution gives "[batch_size,
            1]" shape. The auto-broadcasting of numerical operation with torch
            tensors will amplify this error.
        """
        pass

    def post_process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> None:
        """Post-process the data from the provided replay buffer.

        Typical usage is to update the sampling weight in prioritized
        experience replay. Used in :meth:`update`.
        """
        if hasattr(buffer, "update_weight") and hasattr(batch, "weight"):
            buffer.update_weight(indice, batch.weight)

    def update(
        self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any
    ) -> Mapping[str, Union[float, List[float]]]:
        """Update the policy network and replay buffer.

        It includes 3 function steps: process_fn, learn, and post_process_fn.

        :param int sample_size: 0 means it will extract all the data from the
            buffer, otherwise it will sample a batch with given sample_size.
        :param ReplayBuffer buffer: the corresponding replay buffer.
        """
        if buffer is None:
            return {}
        batch, indice = buffer.sample(sample_size)
        batch = self.process_fn(batch, buffer, indice)
        result = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indice)
        return result

    @staticmethod
    def compute_episodic_return(
        batch: Batch,
        v_s_: Optional[Union[np.ndarray, torch.Tensor]] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        rew_norm: bool = False,
    ) -> Batch:
        """Compute returns over given full-length episodes.

        Implementation of Generalized Advantage Estimator (arXiv:1506.02438).

        :param batch: a data batch which contains several full-episode data
            chronologically.
        :type batch: :class:`~tianshou.data.Batch`
        :param v_s_: the value function of all next states :math:`V(s')`.
        :type v_s_: numpy.ndarray
        :param float gamma: the discount factor, should be in [0, 1], defaults
            to 0.99.
        :param float gae_lambda: the parameter for Generalized Advantage
            Estimation, should be in [0, 1], defaults to 0.95.
        :param bool rew_norm: normalize the reward to Normal(0, 1), defaults
            to False.

        :return: a Batch. The result will be stored in batch.returns as a numpy
            array with shape (bsz, ).
        """
        rew = batch.rew
        v_s_ = np.zeros_like(rew) if v_s_ is None else to_numpy(v_s_.flatten())
        returns = _episodic_return(v_s_, rew, batch.done, gamma, gae_lambda)
        if rew_norm and not np.isclose(returns.std(), 0.0, 1e-2):
            returns = (returns - returns.mean()) / returns.std()
        batch.returns = returns
        return batch

    @staticmethod
    def compute_nstep_return(
        batch: Batch,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> Batch:
        r"""Compute n-step return for Q-learning targets.

        .. math::
            G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +
            \gamma^n (1 - d_{t + n}) Q_{\mathrm{target}}(s_{t + n})

        where :math:`\gamma` is the discount factor,
        :math:`\gamma \in [0, 1]`, :math:`d_t` is the done flag of step
        :math:`t`.

        :param batch: a data batch, which is equal to buffer[indice].
        :type batch: :class:`~tianshou.data.Batch`
        :param buffer: a data buffer which contains several full-episode data
            chronologically.
        :type buffer: :class:`~tianshou.data.ReplayBuffer`
        :param indice: sampled timestep.
        :type indice: numpy.ndarray
        :param function target_q_fn: a function receives :math:`t+n-1` step's
            data and compute target Q value.
        :param float gamma: the discount factor, should be in [0, 1], defaults
            to 0.99.
        :param int n_step: the number of estimation step, should be an int
            greater than 0, defaults to 1.
        :param bool rew_norm: normalize the reward to Normal(0, 1), defaults
            to False.

        :return: a Batch. The result will be stored in batch.returns as a
            torch.Tensor with shape (bsz, ).
        """
        rew = buffer.rew
        if rew_norm:
            bfr = rew[:min(len(buffer), 1000)]  # avoid large buffer
            mean, std = bfr.mean(), bfr.std()
            if np.isclose(std, 0, 1e-2):
                mean, std = 0.0, 1.0
        else:
            mean, std = 0.0, 1.0
        buf_len = len(buffer)
        terminal = (indice + n_step - 1) % buf_len
        target_q_torch = target_q_fn(buffer, terminal).flatten()  # (bsz, )
        target_q = to_numpy(target_q_torch)

        target_q = _nstep_return(rew, buffer.done, target_q, indice,
                                 gamma, n_step, len(buffer), mean, std)

        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch

    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        b = np.array([False, True], dtype=np.bool_)
        i64 = np.array([0, 1], dtype=np.int64)
        _episodic_return(f64, f64, b, 0.1, 0.1)
        _episodic_return(f32, f64, b, 0.1, 0.1)
        _nstep_return(f64, b, f32, i64, 0.1, 1, 4, 1.0, 0.0)


@njit
def _episodic_return(
    v_s_: np.ndarray,
    rew: np.ndarray,
    done: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    """Numba speedup: 4.1s -> 0.057s."""
    returns = np.roll(v_s_, 1)
    m = (1.0 - done) * gamma
    delta = rew + v_s_ * m - returns
    m *= gae_lambda
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + m[i] * gae
        returns[i] += gae
    return returns


@njit
def _nstep_return(
    rew: np.ndarray,
    done: np.ndarray,
    target_q: np.ndarray,
    indice: np.ndarray,
    gamma: float,
    n_step: int,
    buf_len: int,
    mean: float,
    std: float,
) -> np.ndarray:
    """Numba speedup: 0.3s -> 0.15s."""
    returns = np.zeros(indice.shape)
    gammas = np.full(indice.shape, n_step)
    for n in range(n_step - 1, -1, -1):
        now = (indice + n) % buf_len
        gammas[done[now] > 0] = n
        returns[done[now] > 0] = 0.0
        returns = (rew[now] - mean) / std + gamma * returns
    target_q[gammas != n_step] = 0.0
    target_q = target_q * (gamma ** gammas) + returns
    return target_q
