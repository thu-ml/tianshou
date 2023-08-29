from typing import Any, Callable, Literal, Optional, Union, cast

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    BatchWithReturnsProtocol,
    DistBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.policy import BasePolicy
from tianshou.utils import RunningMeanStd

TDistParams = Union[torch.Tensor, tuple[torch.Tensor]]


class PGPolicy(BasePolicy):
    """Implementation of REINFORCE algorithm.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param dist_fn: distribution class for computing the action.
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Callable[[TDistParams], torch.distributions.Distribution],
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        action_scaling: bool = True,
        action_bound_method: Optional[Literal["clip", "tanh"]] = "clip",
        deterministic_eval: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs,
        )
        self.actor = model
        try:
            if action_scaling and not np.isclose(model.max_action, 1.0):  # type: ignore
                import warnings

                warnings.warn(
                    "action_scaling and action_bound_method are only intended"
                    "to deal with unbounded model action space, but find actor model"
                    f"bound action space with max_action={model.max_action}."
                    "Consider using unbounded=True option of the actor model,"
                    "or set action_scaling to False and action_bound_method to None.",
                )
        # TODO: why this try/except? warnings is a standard library module
        except Exception:
            pass
        self.optim = optim
        self.dist_fn = dist_fn
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self._deterministic_eval = deterministic_eval

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
            gamma=self._gamma,
            gae_lambda=1.0,
        )
        if self._rew_norm:
            batch.returns = (unnormalized_returns - self.ret_rms.mean) / np.sqrt(
                self.ret_rms.var + self._eps,
            )
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch: BatchWithReturnsProtocol
        return batch

    def _get_deterministic_action(self, logits: torch.Tensor) -> torch.Tensor:
        if self.action_type == "discrete":
            return logits.argmax(-1)
        if self.action_type == "continuous":
            # assume that the mode of the distribution is the first element
            # of the actor's output (the "logits")
            return logits[0]
        raise RuntimeError(
            f"Unknown action type: {self.action_type}. "
            f"This should not happen and might be a bug."
            f"Supported action types are: 'discrete' and 'continuous'.",
        )

    def forward(
        self,
        batch: RolloutBatchProtocol,
        state: Optional[Union[dict, BatchProtocol, np.ndarray]] = None,
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
        if self._deterministic_eval and not self.training:
            act = self._get_deterministic_action(logits)
        else:
            act = dist.sample()
        result = Batch(logits=logits, act=act, state=hidden, dist=dist)
        return cast(DistBatchProtocol, result)

    # TODO: why does mypy complain?
    def learn(  # type: ignore
        self,
        batch: RolloutBatchProtocol,
        batch_size: int,
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        losses = []
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
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

        return {"loss": losses}
