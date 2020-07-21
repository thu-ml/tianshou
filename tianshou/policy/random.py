import numpy as np
from typing import Union, Optional, Dict, List

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer


class RandomPolicy(BasePolicy):
    """A random agent used in multi-agent learning. It randomly chooses an
    action from the legal action.
    """

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                **kwargs) -> Batch:
        """Compute the random action over the given batch data. The input
        should contain a mask in batch.obs, with "1" to be available and "0"
        to be unavailable. For example,
        ``batch.obs.mask == np.array([[0, 1, 0]])`` means with batch size 1,
        action "1" is available and action "0" and "2" are unavailable.

        :return: A :class:`~tianshou.data.Batch` with "act" key, containing
            the random action.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        mask = batch.obs.mask
        logits = np.random.rand(*mask.shape)
        logits[np.isclose(mask, 0)] = -np.inf
        return Batch(act=logits.argmax(axis=-1))

    def learn(self, batch: Batch, **kwargs
              ) -> Dict[str, Union[float, List[float]]]:
        """No need of a learn function for a random agent, so it returns an
        empty dict."""
        return {}
