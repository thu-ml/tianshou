from typing import cast

import gymnasium as gym
import numpy as np

from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.policy import base
from tianshou.policy.base import OffPolicyAlgorithm, TrainingStats


class MARLRandomTrainingStats(TrainingStats):
    pass


class MARLRandomDiscreteMaskedOffPolicyAlgorithm(OffPolicyAlgorithm):
    """A random agent used in multi-agent learning.

    It randomly chooses an action from the legal actions (according to the given mask).
    """

    class Policy(base.Policy):
        """A random agent used in multi-agent learning.

        It randomly chooses an action from the legal actions.
        """

        def __init__(self, action_space: gym.spaces.Space) -> None:
            super().__init__(action_space=action_space)

        def forward(
            self,
            batch: ObsBatchProtocol,
            state: dict | BatchProtocol | np.ndarray | None = None,
            **kwargs: dict,
        ) -> ActBatchProtocol:
            """Compute the random action over the given batch data.

            The input should contain a mask in batch.obs, with "True" to be
            available and "False" to be unavailable. For example,
            ``batch.obs.mask == np.array([[False, True, False]])`` means with batch
            size 1, action "1" is available but action "0" and "2" are unavailable.

            :return: A :class:`~tianshou.data.Batch` with "act" key, containing
                the random action.
            """
            mask = batch.obs.mask  # type: ignore
            logits = np.random.rand(*mask.shape)
            logits[~mask] = -np.inf
            result = Batch(act=logits.argmax(axis=-1))
            return cast(ActBatchProtocol, result)

    def __init__(self, action_space: gym.spaces.Space) -> None:
        """:param action_space: the environment's action space."""
        super().__init__(policy=self.Policy(action_space))

    def _update_with_batch(self, batch: RolloutBatchProtocol) -> MARLRandomTrainingStats:  # type: ignore
        """Since a random agent learns nothing, it returns an empty dict."""
        return MARLRandomTrainingStats()
