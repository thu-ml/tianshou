import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
    ReplayBuffer,
    SequenceSummaryStats,
    to_torch,
    to_torch_as,
)
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    BatchWithReturnsProtocol,
    DistBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.policy import BasePolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.utils import RunningMeanStd

# TODO: Is there a better way to define this type? mypy doesn't like Callable[[torch.Tensor, ...], torch.distributions.Distribution]
TDistributionFunction: TypeAlias = Callable[..., torch.distributions.Distribution]


@dataclass(kw_only=True)
class PGTrainingStats(TrainingStats):
    loss: SequenceSummaryStats


TPGTrainingStats = TypeVar("TPGTrainingStats", bound=PGTrainingStats)


class PGPolicy(BasePolicy[TPGTrainingStats], Generic[TPGTrainingStats]):
    """Implementation of REINFORCE algorithm.

    :param actor: mapping (s->model_output), should follow the rules in
        :class:`~tianshou.policy.BasePolicy`.
    :param optim: optimizer for actor network.
    :param dist_fn: distribution class for computing the action.
        Maps model_output -> distribution. Typically a Gaussian distribution
        taking `model_output=mean,std` as input for continuous action spaces,
        or a categorical distribution taking `model_output=logits`
        for discrete action spaces. Note that as user, you are responsible
        for ensuring that the distribution is compatible with the action space.
    :param action_space: env's action space.
    :param discount_factor: in [0, 1].
    :param reward_normalization: if True, will normalize the *returns*
        by subtracting the running mean and dividing by the running standard deviation.
        Can be detrimental to performance! See TODO in process_fn.
    :param deterministic_eval: if True, will use deterministic action (the dist's mode)
        instead of stochastic one during evaluation. Does not affect training.
    :param observation_space: Env's observation space.
    :param action_scaling: if True, scale the action from [-1, 1] to the range
        of action_space. Only used if the action_space is continuous.
    :param action_bound_method: method to bound action to range [-1, 1].
        Only used if the action_space is continuous.
    :param lr_scheduler: if not None, will be called in `policy.update()`.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed explanation.
    """

    def __init__(
        self,
        *,
        actor: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: TDistributionFunction,
        action_space: gym.Space,
        discount_factor: float = 0.99,
        # TODO: rename to return_normalization?
        reward_normalization: bool = False,
        deterministic_eval: bool = False,
        observation_space: gym.Space | None = None,
        # TODO: why change the default from the base?
        action_scaling: bool = True,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        if action_scaling and not np.isclose(actor.max_action, 1.0):
            warnings.warn(
                "action_scaling and action_bound_method are only intended"
                "to deal with unbounded model action space, but find actor model"
                f"bound action space with max_action={actor.max_action}."
                "Consider using unbounded=True option of the actor model,"
                "or set action_scaling to False and action_bound_method to None.",
            )
        self.actor = actor
        self.optim = optim
        self.dist_fn = dist_fn
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self.gamma = discount_factor
        self.rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self.deterministic_eval = deterministic_eval

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithReturnsProtocol:
        r"""Compute the discounted returns (Monte Carlo estimates) for each transition.

        They are added to the batch under the field `returns`.
        Note: this function will modify the input batch!

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.

        :param batch: a data batch which contains several episodes of data in
            sequential order. Mind that the end of each finished episode of batch
            should be marked by done flag, unfinished (or collecting) episodes will be
            recognized by buffer.unfinished_index().
        :param buffer: the corresponding replay buffer.
        :param numpy.ndarray indices: tell batch's location in buffer, batch is equal
            to buffer[indices].
        """
        v_s_ = np.full(indices.shape, self.ret_rms.mean)
        # gae_lambda = 1.0 means we use Monte Carlo estimate
        unnormalized_returns, _ = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_=v_s_,
            gamma=self.gamma,
            gae_lambda=1.0,
        )
        # TODO: overridden in A2C, where mean is not subtracted. Subtracting mean
        #  can be very detrimental! It also has no theoretical grounding.
        #  This should be addressed soon!
        if self.rew_norm:
            batch.returns = (unnormalized_returns - self.ret_rms.mean) / np.sqrt(
                self.ret_rms.var + self._eps,
            )
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch: BatchWithReturnsProtocol
        return batch

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> DistBatchProtocol:
        """Compute action over the given batch data by applying the actor.

        Will sample from the dist_fn, if appropriate.
        Returns a new object representing the processed batch data
        (contrary to other methods that modify the input batch inplace).

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        # TODO: rename? It's not really logits and there are particular
        #  assumptions about the order of the output and on distribution type
        logits, hidden = self.actor(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)

        # in this case, the dist is unused!
        if self.deterministic_eval and not self.training:
            act = dist.mode
        else:
            act = dist.sample()
        result = Batch(logits=logits, act=act, state=hidden, dist=dist)
        return cast(DistBatchProtocol, result)

    # TODO: why does mypy complain?
    def learn(  # type: ignore
        self,
        batch: BatchWithReturnsProtocol,
        batch_size: int | None,
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> TPGTrainingStats:
        losses = []
        split_batch_size = batch_size or -1
        for _ in range(repeat):
            for minibatch in batch.split(split_batch_size, merge_last=True):
                self.optim.zero_grad()
                result = self(minibatch)
                dist = result.dist
                act = to_torch_as(minibatch.act, result.act)
                ret = to_torch(minibatch.returns, torch.float, result.act.device)
                log_prob = dist.log_prob(act).reshape(len(ret), -1).transpose(0, 1)
                loss = -(log_prob * ret).mean()
                loss.backward()
                self.optim.step()
                losses.append(loss.item())

        loss_summary_stat = SequenceSummaryStats.from_sequence(losses)

        return PGTrainingStats(loss=loss_summary_stat)  # type: ignore[return-value]
