from typing import Any, Optional, Union, cast

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, to_torch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ModelOutputBatchProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy


class ImitationPolicy(BasePolicy):
    """Implementation of vanilla imitation learning.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> a)
    :param torch.optim.Optimizer optim: for optimizing the model.
    :param gym.Space action_space: env's action space.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        assert self.action_type in [
            "continuous",
            "discrete",
        ], "Please specify action_space."

    def forward(
        self,
        batch: RolloutBatchProtocol,
        state: Optional[Union[dict, BatchProtocol, np.ndarray]] = None,
        **kwargs: Any,
    ) -> ModelOutputBatchProtocol:
        logits, hidden = self.model(batch.obs, state=state, info=batch.info)
        act = logits.max(dim=1)[1] if self.action_type == "discrete" else logits
        result = Batch(logits=logits, act=act, state=hidden)
        return cast(ModelOutputBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *ags: Any, **kwargs: Any) -> dict[str, float]:
        self.optim.zero_grad()
        if self.action_type == "continuous":  # regression
            act = self(batch).act
            act_target = to_torch(batch.act, dtype=torch.float32, device=act.device)
            loss = F.mse_loss(act, act_target)
        elif self.action_type == "discrete":  # classification
            act = F.log_softmax(self(batch).logits, dim=-1)
            act_target = to_torch(batch.act, dtype=torch.long, device=act.device)
            loss = F.nll_loss(act, act_target)
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}
