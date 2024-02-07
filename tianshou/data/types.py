from typing import Protocol

import numpy as np
import torch

from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol, arr_type


class ObsBatchProtocol(BatchProtocol, Protocol):
    """Observations of an environment that a policy can turn into actions.

    Typically used inside a policy's forward
    """

    obs: arr_type | BatchProtocol
    info: arr_type


class RolloutBatchProtocol(ObsBatchProtocol, Protocol):
    """Typically, the outcome of sampling from a replay buffer."""

    obs_next: arr_type | BatchProtocol
    act: arr_type
    rew: np.ndarray
    terminated: arr_type
    truncated: arr_type


class BatchWithReturnsProtocol(RolloutBatchProtocol, Protocol):
    """With added returns, usually computed with GAE."""

    returns: arr_type


class PrioBatchProtocol(RolloutBatchProtocol, Protocol):
    """Contains weights that can be used for prioritized replay."""

    weight: np.ndarray | torch.Tensor


class RecurrentStateBatch(BatchProtocol, Protocol):
    """Used by RNNs in policies, contains `hidden` and `cell` fields."""

    hidden: torch.Tensor
    cell: torch.Tensor


class ActBatchProtocol(BatchProtocol, Protocol):
    """Simplest batch, just containing the action. Useful e.g., for random policy."""

    act: arr_type


class ActStateBatchProtocol(ActBatchProtocol, Protocol):
    """Contains action and state (which can be None), useful for policies that can support RNNs."""

    state: dict | BatchProtocol | np.ndarray | None


class ModelOutputBatchProtocol(ActStateBatchProtocol, Protocol):
    """In addition to state and action, contains model output: (logits)."""

    logits: torch.Tensor


class FQFBatchProtocol(ModelOutputBatchProtocol, Protocol):
    """Model outputs, fractions and quantiles_tau - specific to the FQF model."""

    fractions: torch.Tensor
    quantiles_tau: torch.Tensor


class BatchWithAdvantagesProtocol(BatchWithReturnsProtocol, Protocol):
    """Contains estimated advantages and values.

    Returns are usually computed from GAE of advantages by adding the value.
    """

    adv: torch.Tensor
    v_s: torch.Tensor


class DistBatchProtocol(ModelOutputBatchProtocol, Protocol):
    """Contains dist instances for actions (created by dist_fn).

    Usually categorical or normal.
    """

    dist: torch.distributions.Distribution


class DistLogProbBatchProtocol(DistBatchProtocol, Protocol):
    """Contains dist objects that can be sampled from and log_prob of taken action."""

    log_prob: torch.Tensor


class LogpOldProtocol(BatchWithAdvantagesProtocol, Protocol):
    """Contains logp_old, often needed for importance weights, in particular in PPO.

    Builds on batches that contain advantages and values.
    """

    logp_old: torch.Tensor


class QuantileRegressionBatchProtocol(ModelOutputBatchProtocol, Protocol):
    """Contains taus for algorithms using quantile regression.

    See e.g. https://arxiv.org/abs/1806.06923
    """

    taus: torch.Tensor


class ImitationBatchProtocol(ActBatchProtocol, Protocol):
    """Similar to other batches, but contains imitation_logits and q_value fields."""

    state: dict | Batch | np.ndarray | None
    q_value: torch.Tensor
    imitation_logits: torch.Tensor
