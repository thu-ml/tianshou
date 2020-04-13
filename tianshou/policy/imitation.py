import torch
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.policy import BasePolicy


class ImitationPolicy(BasePolicy):
    """Implementation of vanilla imitation learning (for continuous action space).

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> a)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """
    def __init__(self, model, optim):
        super().__init__()
        self.model = model
        self.optim = optim

    def forward(self, batch, state=None):
        a, h = self.model(batch.obs, state=state, info=batch.info)
        return Batch(act=a, state=h)

    def learn(self, batch, **kwargs):
        self.optim.zero_grad()
        a = self(batch).act
        a_ = torch.tensor(batch.act, dtype=torch.float, device=a.device)
        loss = F.mse_loss(a, a_)
        loss.backward()
        self.optim.step()
        return {'loss': loss.item()}
