import torch
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Union, Optional

from tianshou.data import Batch, to_torch
from tianshou.policy import BasePolicy


class ImitationPolicy(BasePolicy):
    """Implementation of vanilla imitation learning.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> a)
    :param torch.optim.Optimizer optim: for optimizing the model.
    :param str mode: indicate the imitation type ("continuous" or "discrete"
        action space). Default to "continuous".

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        assert mode in ["continuous", "discrete"], \
            f"Mode {mode} is not in ['continuous', 'discrete']."
        self.mode = mode

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        logits, h = self.model(batch.obs, state=state, info=batch.info)
        if self.mode == "discrete":
            a = logits.max(dim=1)[1]
        else:
            a = logits
        return Batch(logits=logits, act=a, state=h)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        self.optim.zero_grad()
        if self.mode == "continuous":  # regression
            a = self(batch).act
            a_ = to_torch(batch.act, dtype=torch.float32, device=a.device)
            loss = F.mse_loss(a, a_)  # type: ignore
        elif self.mode == "discrete":  # classification
            a = F.log_softmax(self(batch).logits, dim=-1)
            a_ = to_torch(batch.act, dtype=torch.long, device=a.device)
            loss = F.nll_loss(a, a_)  # type: ignore
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}
