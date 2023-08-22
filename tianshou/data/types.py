from typing import Optional, Union

import numpy as np
import torch

from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol, arr_type


class RolloutBatchProtocol(BatchProtocol):
    """Typically, the outcome of sampling from a replay buffer."""

    obs: Union[arr_type, BatchProtocol]
    act: arr_type
    rew: np.ndarray
    terminated: arr_type
    truncated: arr_type
    info: arr_type


class BatchWithReturnsProtocol(RolloutBatchProtocol):
    """With added returns, usually computed with GAE."""

    returns: arr_type


class PrioBatchProtocol(RolloutBatchProtocol):
    """Contains weights that can be used for prioritized replay."""

    weight: Union[np.ndarray, torch.Tensor]


class RecurrentStateBatch(BatchProtocol):
    """Used by RNNs in policies, contains `hidden` and `cell` fields."""

    hidden: torch.Tensor
    cell: torch.Tensor


class ActBatchProtocol(BatchProtocol):
    """Simplest batch, just containing the action. Useful e.g., for random policy."""

    act: np.ndarray


class ModelOutputBatchProtocol(ActBatchProtocol):
    """Contains model output: (logits) and potentially hidden states."""

    logits: torch.Tensor
    state: Optional[Union[dict, BatchProtocol, np.ndarray]]


class FQFBatchProtocol(ModelOutputBatchProtocol):
    """Model outputs, fractions and quantiles_tau - specific to the FQF model."""

    fractions: torch.Tensor
    quantiles_tau: torch.Tensor


class BatchWithAdvantagesProtocol(BatchWithReturnsProtocol):
    """Contains estimated advantages and values.

    Returns are usually computed from GAE of advantages by adding the value.
    """

    adv: torch.Tensor
    v_s: torch.Tensor


class DistBatchProtocol(ModelOutputBatchProtocol):
    """Contains dist instances for actions (created by dist_fn).

    Usually categorical or normal.
    """

    dist: torch.distributions.Distribution


class DistLogProbBatchProtocol(DistBatchProtocol):
    """Contains dist objects that can be sampled from and log_prob of taken action."""

    log_prob: torch.Tensor


class LogpOldProtocol(BatchWithAdvantagesProtocol):
    """Contains logp_old, often needed for importance weights, in particular in PPO.

    Builds on batches that contain advantages and values.
    """

    logp_old: torch.Tensor


class QuantileRegressionBatchProtocol(ModelOutputBatchProtocol):
    """Contains taus for algorithms using quantile regression.

    See e.g. https://arxiv.org/abs/1806.06923
    """

    taus: torch.Tensor


class ImitationBatchProtocol(ActBatchProtocol):
    """Similar to other batches, but contains imitation_logits and q_value fields."""

    state: Optional[Union[dict, Batch, np.ndarray]]
    q_value: torch.Tensor
    imitation_logits: torch.Tensor
