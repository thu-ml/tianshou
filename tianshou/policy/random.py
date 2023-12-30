from typing import Any, TypeVar, cast

import numpy as np

from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import TrainingStats


class RandomTrainingStats(TrainingStats):
    pass


TRandomTrainingStats = TypeVar("TRandomTrainingStats", bound=RandomTrainingStats)


class RandomPolicy(BasePolicy[TRandomTrainingStats]):
    """A random agent used in multi-agent learning.

    It randomly chooses an action from the legal action.
    """

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:
        """Compute the random action over the given batch data.

        The input should contain a mask in batch.obs, with "True" to be
        available and "False" to be unavailable. For example,
        ``batch.obs.mask == np.array([[False, True, False]])`` means with batch
        size 1, action "1" is available but action "0" and "2" are unavailable.

        :return: A :class:`~tianshou.data.Batch` with "act" key, containing
            the random action.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        mask = batch.obs.mask  # type: ignore
        logits = np.random.rand(*mask.shape)
        logits[~mask] = -np.inf
        result = Batch(act=logits.argmax(axis=-1))
        return cast(ActBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TRandomTrainingStats:  # type: ignore
        """Since a random agent learns nothing, it returns an empty dict."""
        return RandomTrainingStats()  # type: ignore[return-value]
