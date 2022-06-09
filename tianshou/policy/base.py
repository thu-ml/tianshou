from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from numba import njit
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.utils import MultipleLRSchedulers


class BasePolicy(ABC, nn.Module):
    """The base class for any RL policy.

    Tianshou aims to modularize RL algorithms. It comes into several classes of
    policies in Tianshou. All of the policy classes must inherit
    :class:`~tianshou.policy.BasePolicy`.

    A policy class typically has the following parts:

    * :meth:`~tianshou.policy.BasePolicy.__init__`: initialize the policy, including \
        coping the target network and so on;
    * :meth:`~tianshou.policy.BasePolicy.forward`: compute action with given \
        observation;
    * :meth:`~tianshou.policy.BasePolicy.process_fn`: pre-process data from the \
        replay buffer (this function can interact with replay buffer);
    * :meth:`~tianshou.policy.BasePolicy.learn`: update policy with a given batch of \
        data.
    * :meth:`~tianshou.policy.BasePolicy.post_process_fn`: update the replay buffer \
        from the learning process (e.g., prioritized replay buffer needs to update \
        the weight);
    * :meth:`~tianshou.policy.BasePolicy.update`: the main interface for training, \
        i.e., `process_fn -> learn -> post_process_fn`.

    Most of the policy needs a neural network to predict the action and an
    optimizer to optimize the policy. The rules of self-defined networks are:

    1. Input: observation "obs" (may be a ``numpy.ndarray``, a ``torch.Tensor``, a \
    dict or any others), hidden state "state" (for RNN usage), and other information \
    "info" provided by the environment.
    2. Output: some "logits", the next hidden state "state", and the intermediate \
    result during policy forwarding procedure "policy". The "logits" could be a tuple \
    instead of a ``torch.Tensor``. It depends on how the policy process the network \
    output. For example, in PPO, the return of the network might be \
    ``(mu, sigma), state`` for Gaussian policy. The "policy" can be a Batch of \
    torch.Tensor or other things, which will be stored in the replay buffer, and can \
    be accessed in the policy update process (e.g. in "policy.learn()", the \
    "batch.policy" is what you need).

    Since :class:`~tianshou.policy.BasePolicy` inherits ``torch.nn.Module``, you can
    use :class:`~tianshou.policy.BasePolicy` almost the same as ``torch.nn.Module``,
    for instance, loading and saving the model:
    ::

        torch.save(policy.state_dict(), "policy.pth")
        policy.load_state_dict(torch.load("policy.pth"))
    """

    def __init__(
        self,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        action_scaling: bool = False,
        action_bound_method: str = "",
        lr_scheduler: Optional[Union[torch.optim.lr_scheduler.LambdaLR,
                                     MultipleLRSchedulers]] = None,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_type = ""
        if isinstance(action_space, (Discrete, MultiDiscrete, MultiBinary)):
            self.action_type = "discrete"
        elif isinstance(action_space, Box):
            self.action_type = "continuous"
        self.agent_id = 0
        self.updating = False
        self.action_scaling = action_scaling
        # can be one of ("clip", "tanh", ""), empty string means no bounding
        assert action_bound_method in ("", "clip", "tanh")
        self.action_bound_method = action_bound_method
        self.lr_scheduler = lr_scheduler
        self._compile()

    def set_agent_id(self, agent_id: int) -> None:
        """Set self.agent_id = agent_id, for MARL."""
        self.agent_id = agent_id

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        """Modify the action from policy.forward with exploration noise.

        :param act: a data batch or numpy.ndarray which is the action taken by
            policy.forward.
        :param batch: the input batch for policy.forward, kept for advanced usage.

        :return: action in the same form of input "act" but with added exploration
            noise.
        """
        return act

    def soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """Softly update the parameters of target module towards the parameters \
        of source module."""
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    @abstractmethod
    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which MUST have the following keys:

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

        .. note::

            In continuous action space, you should do another step "map_action" to get
            the real action:
            ::

                act = policy(batch).act  # doesn't map to the target action range
                act = policy.map_action(act, batch)
        """
        pass

    def map_action(self, act: Union[Batch, np.ndarray]) -> Union[Batch, np.ndarray]:
        """Map raw network output to action range in gym's env.action_space.

        This function is called in :meth:`~tianshou.data.Collector.collect` and only
        affects action sending to env. Remapped action will not be stored in buffer
        and thus can be viewed as a part of env (a black box action transformation).

        Action mapping includes 2 standard procedures: bounding and scaling. Bounding
        procedure expects original action range is (-inf, inf) and maps it to [-1, 1],
        while scaling procedure expects original action range is (-1, 1) and maps it
        to [action_space.low, action_space.high]. Bounding procedure is applied first.

        :param act: a data batch or numpy.ndarray which is the action taken by
            policy.forward.

        :return: action in the same form of input "act" but remap to the target action
            space.
        """
        if isinstance(self.action_space, gym.spaces.Box) and \
                isinstance(act, np.ndarray):
            # currently this action mapping only supports np.ndarray action
            if self.action_bound_method == "clip":
                act = np.clip(act, -1.0, 1.0)
            elif self.action_bound_method == "tanh":
                act = np.tanh(act)
            if self.action_scaling:
                assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                    "action scaling only accepts raw action range = [-1, 1]"
                low, high = self.action_space.low, self.action_space.high
                act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
        return act

    def map_action_inverse(
        self, act: Union[Batch, List, np.ndarray]
    ) -> Union[Batch, List, np.ndarray]:
        """Inverse operation to :meth:`~tianshou.policy.BasePolicy.map_action`.

        This function is called in :meth:`~tianshou.data.Collector.collect` for
        random initial steps. It scales [action_space.low, action_space.high] to
        the value ranges of policy.forward.

        :param act: a data batch, list or numpy.ndarray which is the action taken
            by gym.spaces.Box.sample().

        :return: action remapped.
        """
        if isinstance(self.action_space, gym.spaces.Box):
            act = to_numpy(act)
            if isinstance(act, np.ndarray):
                if self.action_scaling:
                    low, high = self.action_space.low, self.action_space.high
                    scale = high - low
                    eps = np.finfo(np.float32).eps.item()
                    scale[scale < eps] += eps
                    act = (act - low) * 2.0 / scale - 1.0
                if self.action_bound_method == "tanh":
                    act = (np.log(1.0 + act) - np.log(1.0 - act)) / 2.0  # type: ignore
        return act

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Pre-process the data from the provided replay buffer.

        Used in :meth:`update`. Check out :ref:`process_fn` for more information.
        """
        return batch

    @abstractmethod
    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        """Update policy with a given batch of data.

        :return: A dict, including the data needed to be logged (e.g., loss).

        .. note::

            In order to distinguish the collecting state, updating state and
            testing state, you can check the policy state by ``self.training``
            and ``self.updating``. Please refer to :ref:`policy_state` for more
            detailed explanation.

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
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> None:
        """Post-process the data from the provided replay buffer.

        Typical usage is to update the sampling weight in prioritized
        experience replay. Used in :meth:`update`.
        """
        if hasattr(buffer, "update_weight") and hasattr(batch, "weight"):
            buffer.update_weight(indices, batch.weight)

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        """Update the policy network and replay buffer.

        It includes 3 function steps: process_fn, learn, and post_process_fn. In
        addition, this function will change the value of ``self.updating``: it will be
        False before this function and will be True when executing :meth:`update`.
        Please refer to :ref:`policy_state` for more detailed explanation.

        :param int sample_size: 0 means it will extract all the data from the buffer,
            otherwise it will sample a batch with given sample_size.
        :param ReplayBuffer buffer: the corresponding replay buffer.

        :return: A dict, including the data needed to be logged (e.g., loss) from
            ``policy.learn()``.
        """
        if buffer is None:
            return {}
        batch, indices = buffer.sample(sample_size)
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)
        result = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False
        return result

    @staticmethod
    def value_mask(buffer: ReplayBuffer, indices: np.ndarray) -> np.ndarray:
        """Value mask determines whether the obs_next of buffer[indices] is valid.

        For instance, usually "obs_next" after "done" flag is considered to be invalid,
        and its q/advantage value can provide meaningless (even misleading)
        information, and should be set to 0 by hand. But if "done" flag is generated
        because timelimit of game length (info["TimeLimit.truncated"] is set to True in
        gym's settings), "obs_next" will instead be valid. Value mask is typically used
        for assisting in calculating the correct q/advantage value.

        :param ReplayBuffer buffer: the corresponding replay buffer.
        :param numpy.ndarray indices: indices of replay buffer whose "obs_next" will be
            judged.

        :return: A bool type numpy.ndarray in the same shape with indices. "True" means
            "obs_next" of that buffer[indices] is valid.
        """
        mask = ~buffer.done[indices]
        # info["TimeLimit.truncated"] will be True if "done" flag is generated by
        # timelimit of environments. Checkout gym.wrappers.TimeLimit.
        if hasattr(buffer, 'info') and 'TimeLimit.truncated' in buffer.info:
            mask = mask | buffer.info['TimeLimit.truncated'][indices]
        return mask

    @staticmethod
    def compute_episodic_return(
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray,
        v_s_: Optional[Union[np.ndarray, torch.Tensor]] = None,
        v_s: Optional[Union[np.ndarray, torch.Tensor]] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute returns over given batch.

        Use Implementation of Generalized Advantage Estimator (arXiv:1506.02438)
        to calculate q/advantage value of given batch.

        :param Batch batch: a data batch which contains several episodes of data in
            sequential order. Mind that the end of each finished episode of batch
            should be marked by done flag, unfinished (or collecting) episodes will be
            recognized by buffer.unfinished_index().
        :param numpy.ndarray indices: tell batch's location in buffer, batch is equal
            to buffer[indices].
        :param np.ndarray v_s_: the value function of all next states :math:`V(s')`.
        :param float gamma: the discount factor, should be in [0, 1]. Default to 0.99.
        :param float gae_lambda: the parameter for Generalized Advantage Estimation,
            should be in [0, 1]. Default to 0.95.

        :return: two numpy arrays (returns, advantage) with each shape (bsz, ).
        """
        rew = batch.rew
        if v_s_ is None:
            assert np.isclose(gae_lambda, 1.0)
            v_s_ = np.zeros_like(rew)
        else:
            v_s_ = to_numpy(v_s_.flatten())
            v_s_ = v_s_ * BasePolicy.value_mask(buffer, indices)
        v_s = np.roll(v_s_, 1) if v_s is None else to_numpy(v_s.flatten())

        end_flag = batch.done.copy()
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        advantage = _gae_return(v_s, v_s_, rew, end_flag, gamma, gae_lambda)
        returns = advantage + v_s
        # normalization varies from each policy, so we don't do it here
        return returns, advantage

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

        where :math:`\gamma` is the discount factor, :math:`\gamma \in [0, 1]`,
        :math:`d_t` is the done flag of step :math:`t`.

        :param Batch batch: a data batch, which is equal to buffer[indice].
        :param ReplayBuffer buffer: the data buffer.
        :param function target_q_fn: a function which compute target Q value
            of "obs_next" given data buffer and wanted indices.
        :param float gamma: the discount factor, should be in [0, 1]. Default to 0.99.
        :param int n_step: the number of estimation step, should be an int greater
            than 0. Default to 1.
        :param bool rew_norm: normalize the reward to Normal(0, 1), Default to False.

        :return: a Batch. The result will be stored in batch.returns as a
            torch.Tensor with the same shape as target_q_fn's return tensor.
        """
        assert not rew_norm, \
            "Reward normalization in computing n-step returns is unsupported now."
        rew = buffer.rew
        bsz = len(indice)
        indices = [indice]
        for _ in range(n_step - 1):
            indices.append(buffer.next(indices[-1]))
        indices = np.stack(indices)
        # terminal indicates buffer indexes nstep after 'indice',
        # and are truncated at the end of each episode
        terminal = indices[-1]
        with torch.no_grad():
            target_q_torch = target_q_fn(buffer, terminal)  # (bsz, ?)
        target_q = to_numpy(target_q_torch.reshape(bsz, -1))
        target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)

        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch

    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        b = np.array([False, True], dtype=np.bool_)
        i64 = np.array([[0, 1]], dtype=np.int64)
        _gae_return(f64, f64, f64, b, 0.1, 0.1)
        _gae_return(f32, f32, f64, b, 0.1, 0.1)
        _nstep_return(f64, b, f32.reshape(-1, 1), i64, 0.1, 1)


@njit
def _gae_return(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    end_flag: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    returns = np.zeros(rew.shape)
    delta = rew + v_s_ * gamma - v_s
    discount = (1.0 - end_flag) * (gamma * gae_lambda)
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + discount[i] * gae
        returns[i] = gae
    return returns


@njit
def _nstep_return(
    rew: np.ndarray,
    end_flag: np.ndarray,
    target_q: np.ndarray,
    indices: np.ndarray,
    gamma: float,
    n_step: int,
) -> np.ndarray:
    gamma_buffer = np.ones(n_step + 1)
    for i in range(1, n_step + 1):
        gamma_buffer[i] = gamma_buffer[i - 1] * gamma
    target_shape = target_q.shape
    bsz = target_shape[0]
    # change target_q to 2d array
    target_q = target_q.reshape(bsz, -1)
    returns = np.zeros(target_q.shape)
    gammas = np.full(indices[0].shape, n_step)
    for n in range(n_step - 1, -1, -1):
        now = indices[n]
        gammas[end_flag[now] > 0] = n + 1
        returns[end_flag[now] > 0] = 0.0
        returns = rew[now].reshape(bsz, 1) + gamma * returns
    target_q = target_q * gamma_buffer[gammas].reshape(bsz, 1) + returns
    return target_q.reshape(target_shape)
