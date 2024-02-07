import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from numba import njit
from overrides import override
from torch import nn

from tianshou.data import ReplayBuffer, SequenceSummaryStats, to_numpy, to_torch_as
from tianshou.data.batch import Batch, BatchProtocol, arr_type
from tianshou.data.buffer.base import TBuffer
from tianshou.data.types import (
    ActBatchProtocol,
    BatchWithReturnsProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.utils import MultipleLRSchedulers
from tianshou.utils.print import DataclassPPrintMixin

logger = logging.getLogger(__name__)

TLearningRateScheduler: TypeAlias = torch.optim.lr_scheduler.LRScheduler | MultipleLRSchedulers


@dataclass(kw_only=True)
class TrainingStats(DataclassPPrintMixin):
    _non_loss_fields = ("train_time", "smoothed_loss")

    train_time: float = 0.0
    """The time for learning models."""

    # TODO: modified in the trainer but not used anywhere else. Should be refactored.
    smoothed_loss: dict = field(default_factory=dict)
    """The smoothed loss statistics of the policy learn step."""

    # Mainly so that we can override this in the TrainingStatsWrapper
    def _get_self_dict(self) -> dict[str, Any]:
        return self.__dict__

    def get_loss_stats_dict(self) -> dict[str, float]:
        """Return loss statistics as a dict for logging.

        Returns a dict with all fields except train_time and smoothed_loss. Moreover, fields with value None excluded,
        and instances of SequenceSummaryStats are replaced by their mean.
        """
        result = {}
        for k, v in self._get_self_dict().items():
            if k.startswith("_"):
                logger.debug(f"Skipping {k=} as it starts with an underscore.")
                continue
            if k in self._non_loss_fields or v is None:
                continue
            if isinstance(v, SequenceSummaryStats):
                result[k] = v.mean
            else:
                result[k] = v

        return result


class TrainingStatsWrapper(TrainingStats):
    _setattr_frozen = False
    _training_stats_public_fields = TrainingStats.__dataclass_fields__.keys()

    def __init__(self, wrapped_stats: TrainingStats) -> None:
        """In this particular case, super().__init__() should be called LAST in the subclass init."""
        self._wrapped_stats = wrapped_stats

        # HACK: special sauce for the existing attributes of the base TrainingStats class
        # for some reason, delattr doesn't work here, so we need to delegate their handling
        # to the wrapped stats object by always keeping the value there and in self in sync
        # see also __setattr__
        for k in self._training_stats_public_fields:
            super().__setattr__(k, getattr(self._wrapped_stats, k))

        self._setattr_frozen = True

    @override
    def _get_self_dict(self) -> dict[str, Any]:
        return {**self._wrapped_stats._get_self_dict(), **self.__dict__}

    @property
    def wrapped_stats(self) -> TrainingStats:
        return self._wrapped_stats

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_stats, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Setattr logic for wrapper of a dataclass with default values.

        1. If name exists directly in self, set it there.
        2. If it exists in self._wrapped_stats, set it there instead.
        3. Special case: if name is in the base TrainingStats class, keep it in sync between self and the _wrapped_stats.
        4. If name doesn't exist in either and attribute setting is frozen, raise an AttributeError.
        """
        # HACK: special sauce for the existing attributes of the base TrainingStats class, see init
        # Need to keep them in sync with the wrapped stats object
        if name in self._training_stats_public_fields:
            setattr(self._wrapped_stats, name, value)
            super().__setattr__(name, value)
            return

        if not self._setattr_frozen:
            super().__setattr__(name, value)
            return

        if not hasattr(self, name):
            raise AttributeError(
                f"Setting new attributes on StatsWrappers outside of init is not allowed. "
                f"Tried to set {name=}, {value=} on {self.__class__.__name__}. \n"
                f"NOTE: you may get this error if you call super().__init__() in your subclass init too early! "
                f"The call to super().__init__() should be the last call in your subclass init.",
            )
        if hasattr(self._wrapped_stats, name):
            setattr(self._wrapped_stats, name, value)
        else:
            super().__setattr__(name, value)


TTrainingStats = TypeVar("TTrainingStats", bound=TrainingStats)


class BasePolicy(nn.Module, Generic[TTrainingStats], ABC):
    """The base class for any RL policy.

    Tianshou aims to modularize RL algorithms. It comes into several classes of
    policies in Tianshou. All policy classes must inherit from
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

    :param action_space: Env's action_space.
    :param observation_space: Env's observation space. TODO: appears unused...
    :param action_scaling: if True, scale the action from [-1, 1] to the range
        of action_space. Only used if the action_space is continuous.
    :param action_bound_method: method to bound action to range [-1, 1].
        Only used if the action_space is continuous.
    :param lr_scheduler: if not None, will be called in `policy.update()`.
    """

    def __init__(
        self,
        *,
        action_space: gym.Space,
        # TODO: does the policy actually need the observation space?
        observation_space: gym.Space | None = None,
        action_scaling: bool = False,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        allowed_action_bound_methods = ("clip", "tanh")
        if (
            action_bound_method is not None
            and action_bound_method not in allowed_action_bound_methods
        ):
            raise ValueError(
                f"Got invalid {action_bound_method=}. "
                f"Valid values are: {allowed_action_bound_methods}.",
            )
        if action_scaling and not isinstance(action_space, Box):
            raise ValueError(
                f"action_scaling can only be True when action_space is Box but "
                f"got: {action_space}",
            )

        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        if isinstance(action_space, Discrete | MultiDiscrete | MultiBinary):
            self.action_type = "discrete"
        elif isinstance(action_space, Box):
            self.action_type = "continuous"
        else:
            raise ValueError(f"Unsupported action space: {action_space}.")
        self.agent_id = 0
        self.updating = False
        self.action_scaling = action_scaling
        self.action_bound_method = action_bound_method
        self.lr_scheduler = lr_scheduler
        self._compile()

    def set_agent_id(self, agent_id: int) -> None:
        """Set self.agent_id = agent_id, for MARL."""
        self.agent_id = agent_id

    # TODO: needed, since for most of offline algorithm, the algorithm itself doesn't
    #  have a method to add noise to action.
    #  So we add the default behavior here. It's a little messy, maybe one can
    #  find a better way to do this.
    def exploration_noise(
        self,
        act: np.ndarray | BatchProtocol,
        batch: RolloutBatchProtocol,
    ) -> np.ndarray | BatchProtocol:
        """Modify the action from policy.forward with exploration noise.

        NOTE: currently does not add any noise! Needs to be overridden by subclasses
        to actually do something.

        :param act: a data batch or numpy.ndarray which is the action taken by
            policy.forward.
        :param batch: the input batch for policy.forward, kept for advanced usage.
        :return: action in the same form of input "act" but with added exploration
            noise.
        """
        return act

    def soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """Softly update the parameters of target module towards the parameters of source module."""
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters(), strict=True):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def compute_action(
        self,
        obs: arr_type,
        info: dict[str, Any] | None = None,
        state: dict | BatchProtocol | np.ndarray | None = None,
    ) -> np.ndarray | int:
        """Get action as int (for discrete env's) or array (for continuous ones) from an env's observation and info.

        :param obs: observation from the gym's env.
        :param info: information given by the gym's env.
        :param state: the hidden state of RNN policy, used for recurrent policy.
        :return: action as int (for discrete env's) or array (for continuous ones).
        """
        # need to add empty batch dimension
        obs = obs[None, :]
        obs_batch = cast(ObsBatchProtocol, Batch(obs=obs, info=info))
        act = self.forward(obs_batch, state=state).act.squeeze()
        if isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()
        act = self.map_action(act)
        if isinstance(self.action_space, Discrete):
            # could be an array of shape (), easier to just convert to int
            act = int(act)  # type: ignore
        return act

    @abstractmethod
    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:
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

    @staticmethod
    def _action_to_numpy(act: arr_type) -> np.ndarray:
        act = to_numpy(act)  # NOTE: to_numpy could confusingly also return a Batch
        if not isinstance(act, np.ndarray):
            raise ValueError(
                f"act should have been be a numpy.ndarray, but got {type(act)}.",
            )
        return act

    def map_action(
        self,
        act: arr_type,
    ) -> np.ndarray:
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
        act = self._action_to_numpy(act)
        if isinstance(self.action_space, gym.spaces.Box):
            if self.action_bound_method == "clip":
                act = np.clip(act, -1.0, 1.0)
            elif self.action_bound_method == "tanh":
                act = np.tanh(act)
            if self.action_scaling:
                assert (
                    np.min(act) >= -1.0 and np.max(act) <= 1.0
                ), f"action scaling only accepts raw action range = [-1, 1], but got: {act}"
                low, high = self.action_space.low, self.action_space.high
                act = low + (high - low) * (act + 1.0) / 2.0
        return act

    def map_action_inverse(
        self,
        act: arr_type,
    ) -> np.ndarray:
        """Inverse operation to :meth:`~tianshou.policy.BasePolicy.map_action`.

        This function is called in :meth:`~tianshou.data.Collector.collect` for
        random initial steps. It scales [action_space.low, action_space.high] to
        the value ranges of policy.forward.

        :param act: a data batch, list or numpy.ndarray which is the action taken
            by gym.spaces.Box.sample().

        :return: action remapped.
        """
        act = self._action_to_numpy(act)
        if isinstance(self.action_space, gym.spaces.Box):
            if self.action_scaling:
                low, high = self.action_space.low, self.action_space.high
                scale = high - low
                eps = np.finfo(np.float32).eps.item()
                scale[scale < eps] += eps
                act = (act - low) * 2.0 / scale - 1.0
            if self.action_bound_method == "tanh":
                act = (np.log(1.0 + act) - np.log(1.0 - act)) / 2.0

        return act

    def process_buffer(self, buffer: TBuffer) -> TBuffer:
        """Pre-process the replay buffer, e.g., to add new keys.

        Used in BaseTrainer initialization method, usually used by offline trainers.

        Note: this will only be called once, when the trainer is initialized!
            If the buffer is empty by then, there will be nothing to process.
            This method is meant to be overridden by policies which will be trained
            offline at some stage, e.g., in a pre-training step.
        """
        return buffer

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        """Pre-process the data from the provided replay buffer.

        Meant to be overridden by subclasses. Typical usage is to add new keys to the
        batch, e.g., to add the value function of the next state. Used in :meth:`update`,
        which is usually called repeatedly during training.

        For modifying the replay buffer only once at the beginning
        (e.g., for offline learning) see :meth:`process_buffer`.
        """
        return batch

    @abstractmethod
    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TTrainingStats:
        """Update policy with a given batch of data.

        :return: A dataclass object, including the data needed to be logged (e.g., loss).

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

    def post_process_fn(
        self,
        batch: BatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> None:
        """Post-process the data from the provided replay buffer.

        This will only have an effect if the buffer has the
        method `update_weight` and the batch has the attribute `weight`.

        Typical usage is to update the sampling weight in prioritized
        experience replay. Used in :meth:`update`.
        """
        if hasattr(buffer, "update_weight"):
            if hasattr(batch, "weight"):
                buffer.update_weight(indices, batch.weight)
            else:
                logger.warning(
                    "batch has no attribute 'weight', but buffer has an "
                    "update_weight method. This is probably a mistake."
                    "Prioritized replay is disabled for this batch.",
                )

    def update(
        self,
        sample_size: int | None,
        buffer: ReplayBuffer | None,
        **kwargs: Any,
    ) -> TTrainingStats:
        """Update the policy network and replay buffer.

        It includes 3 function steps: process_fn, learn, and post_process_fn. In
        addition, this function will change the value of ``self.updating``: it will be
        False before this function and will be True when executing :meth:`update`.
        Please refer to :ref:`policy_state` for more detailed explanation. The return
        value of learn is augmented with the training time within update, while smoothed
        loss values are computed in the trainer.

        :param sample_size: 0 means it will extract all the data from the buffer,
            otherwise it will sample a batch with given sample_size. None also
            means it will extract all the data from the buffer, but it will be shuffled
            first. TODO: remove the option for 0?
        :param buffer: the corresponding replay buffer.

        :return: A dataclass object containing the data needed to be logged (e.g., loss) from
            ``policy.learn()``.
        """
        # TODO: when does this happen?
        # -> this happens never in practice as update is either called with a collector buffer or an assert before
        if buffer is None:
            return TrainingStats()  # type: ignore[return-value]
        start_time = time.time()
        batch, indices = buffer.sample(sample_size)
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)
        training_stat = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False
        training_stat.train_time = time.time() - start_time
        return training_stat

    @staticmethod
    def value_mask(buffer: ReplayBuffer, indices: np.ndarray) -> np.ndarray:
        """Value mask determines whether the obs_next of buffer[indices] is valid.

        For instance, usually "obs_next" after "done" flag is considered to be invalid,
        and its q/advantage value can provide meaningless (even misleading)
        information, and should be set to 0 by hand. But if "done" flag is generated
        because timelimit of game length (info["TimeLimit.truncated"] is set to True in
        gym's settings), "obs_next" will instead be valid. Value mask is typically used
        for assisting in calculating the correct q/advantage value.

        :param buffer: the corresponding replay buffer.
        :param numpy.ndarray indices: indices of replay buffer whose "obs_next" will be
            judged.

        :return: A bool type numpy.ndarray in the same shape with indices. "True" means
            "obs_next" of that buffer[indices] is valid.
        """
        return ~buffer.terminated[indices]

    @staticmethod
    def compute_episodic_return(
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
        v_s_: np.ndarray | torch.Tensor | None = None,
        v_s: np.ndarray | torch.Tensor | None = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Compute returns over given batch.

        Use Implementation of Generalized Advantage Estimator (arXiv:1506.02438)
        to calculate q/advantage value of given batch. Returns are calculated as
        advantage + value, which is exactly equivalent to using :math:`TD(\lambda)`
        for estimating returns.

        :param batch: a data batch which contains several episodes of data in
            sequential order. Mind that the end of each finished episode of batch
            should be marked by done flag, unfinished (or collecting) episodes will be
            recognized by buffer.unfinished_index().
        :param buffer: the corresponding replay buffer.
        :param numpy.ndarray indices: tell batch's location in buffer, batch is equal
            to buffer[indices].
        :param np.ndarray v_s_: the value function of all next states :math:`V(s')`.
            If None, it will be set to an array of 0.
        :param v_s: the value function of all current states :math:`V(s)`.
        :param gamma: the discount factor, should be in [0, 1]. Default to 0.99.
        :param gae_lambda: the parameter for Generalized Advantage Estimation,
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

        end_flag = np.logical_or(batch.terminated, batch.truncated)
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        advantage = _gae_return(v_s, v_s_, rew, end_flag, gamma, gae_lambda)
        returns = advantage + v_s
        # normalization varies from each policy, so we don't do it here
        return returns, advantage

    @staticmethod
    def compute_nstep_return(
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
        target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> BatchWithReturnsProtocol:
        r"""Compute n-step return for Q-learning targets.

        .. math::
            G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +
            \gamma^n (1 - d_{t + n}) Q_{\mathrm{target}}(s_{t + n})

        where :math:`\gamma` is the discount factor, :math:`\gamma \in [0, 1]`,
        :math:`d_t` is the done flag of step :math:`t`.

        :param batch: a data batch, which is equal to buffer[indices].
        :param buffer: the data buffer.
        :param indices: tell batch's location in buffer
        :param function target_q_fn: a function which compute target Q value
            of "obs_next" given data buffer and wanted indices.
        :param gamma: the discount factor, should be in [0, 1]. Default to 0.99.
        :param n_step: the number of estimation step, should be an int greater
            than 0. Default to 1.
        :param rew_norm: normalize the reward to Normal(0, 1), Default to False.
            TODO: passing True is not supported and will cause an error!
        :return: a Batch. The result will be stored in batch.returns as a
            torch.Tensor with the same shape as target_q_fn's return tensor.
        """
        assert not rew_norm, "Reward normalization in computing n-step returns is unsupported now."
        if len(indices) != len(batch):
            raise ValueError(f"Batch size {len(batch)} and indices size {len(indices)} mismatch.")

        rew = buffer.rew
        bsz = len(indices)
        indices = [indices]
        for _ in range(n_step - 1):
            indices.append(buffer.next(indices[-1]))
        indices = np.stack(indices)
        # terminal indicates buffer indexes nstep after 'indices',
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
        return cast(BatchWithReturnsProtocol, batch)

    @staticmethod
    def _compile() -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        b = np.array([False, True], dtype=np.bool_)
        i64 = np.array([[0, 1]], dtype=np.int64)
        _gae_return(f64, f64, f64, b, 0.1, 0.1)
        _gae_return(f32, f32, f64, b, 0.1, 0.1)
        _nstep_return(f64, b, f32.reshape(-1, 1), i64, 0.1, 1)


# TODO: rename? See docstring
@njit
def _gae_return(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    end_flag: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    r"""Computes advantages with GAE.

    Note: doesn't compute returns but rather advantages. The return
    is given by the output of this + v_s. Note that the advantages plus v_s
    is exactly the same as the TD-lambda target, which is computed by the recursive
    formula:

    .. math::
        G_t^\lambda = r_t + \gamma ( \lambda G_{t+1}^\lambda + (1 - \lambda) V_{t+1} )

    The GAE is computed recursively as:

    .. math::
        \delta_t = r_t + \gamma V_{t+1} - V_t \n
        A_t^\lambda= \delta_t + \gamma \lambda A_{t+1}^\lambda

    And the following equality holds:

    .. math::
        G_t^\lambda = A_t^\lambda+ V_t

    :param v_s: values in an episode, i.e. $V_t$
    :param v_s_: next values in an episode, i.e. v_s shifted by 1, equivalent to
        $V_{t+1}$
    :param rew: rewards in an episode, i.e. $r_t$
    :param end_flag: boolean array indicating whether the episode is done
    :param gamma: discount factor
    :param gae_lambda: lambda parameter for GAE, controlling the bias-variance tradeoff
    :return:
    """
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
