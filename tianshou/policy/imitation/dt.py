from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, to_torch, to_torch_as
from tianshou.policy import BasePolicy
from tianshou.utils.net.common import DecisionTransformer


class DTPolicy(BasePolicy):
    """Implementation of Decision Transformer. arXiv:2106.01345.

    :param DecisionTransformer model: a transformer model.
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
        model: DecisionTransformer,
        optim: torch.optim.Optimizer,
        device: Optional[Union[str, torch.device]] = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.device = device
        assert self.action_type in ["continuous", "discrete"], \
            "Please specify action_space."
        self.max_ep_len = model.model.config.max_ep_len  # type: ignore

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        logits, hidden = self.model(batch.obs, state=state, info=batch.info)
        # update hidden state
        actions = hidden["actions"]
        if self.action_type == "discrete":
            act = logits.max(dim=1)[1]
            actions = torch.cat(
                [
                    actions,
                    F.one_hot(act, num_classes=logits.shape[1]
                              ).reshape(logits.shape[0], 1, -1)
                ],
                dim=1
            )
        else:
            act = logits
            actions = torch.cat([actions, act.unsqueeze(1)], dim=1)
        returns_to_go = hidden["returns_to_go"]
        returns_to_go = torch.cat([returns_to_go, returns_to_go[:, -1:, :]], dim=1)
        if not isinstance(batch.rew, Batch):
            returns_to_go[:, -1:, :] -= to_torch_as(batch.rew,
                                                    returns_to_go).view(-1, 1, 1)
        timesteps = hidden["timesteps"]
        timesteps = torch.cat(
            [timesteps, (timesteps[:, -1:] + 1) % self.max_ep_len], dim=1
        )
        new_state = {
            "states": hidden["states"],
            "actions": actions,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps
        }
        return Batch(logits=logits, act=act, state=new_state)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
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
