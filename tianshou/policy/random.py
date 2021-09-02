from typing import Any, Dict, Optional, Union

import numpy as np

from tianshou.data import Batch
from tianshou.policy import BasePolicy


class RandomPolicy(BasePolicy):
    """A random agent used in multi-agent learning.

    It randomly chooses an action from the legal action.
    """

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
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
        mask = batch.obs.mask
        logits = np.random.rand(*mask.shape)
        logits[~mask] = -np.inf
        return Batch(act=logits.argmax(axis=-1))

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Since a random agent learns nothing, it returns an empty dict."""
        return {}
