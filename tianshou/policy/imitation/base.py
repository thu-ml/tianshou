import torch
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.policy import BasePolicy


class ImitationPolicy(BasePolicy):
    """Implementation of vanilla imitation learning (for continuous action space).

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> a)
    :param torch.optim.Optimizer optim: for optimizing the model.
    :param str mode: indicate the imitation type ("continuous" or "discrete"
        action space), defaults to "continuous".

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """
    def __init__(self, model, optim, mode='continuous'):
        super().__init__()
        self.model = model
        self.optim = optim
        assert mode in ['continuous', 'discrete'], \
            f'Mode {mode} is not in ["continuous", "discrete"]'
        self.mode = mode

    def forward(self, batch, state=None):
        logits, h = self.model(batch.obs, state=state, info=batch.info)
        if self.mode == 'discrete':
            a = logits.max(dim=1)[1]
        else:
            a = logits
        return Batch(logits=logits, act=a, state=h)

    def learn(self, batch, **kwargs):
        self.optim.zero_grad()
        if self.mode == 'continuous':
            a = self(batch).act
            a_ = torch.tensor(batch.act, dtype=torch.float, device=a.device)
            loss = F.mse_loss(a, a_)
        elif self.mode == 'discrete':  # classification
            a = self(batch).logits
            a_ = torch.tensor(batch.act, dtype=torch.long, device=a.device)
            loss = F.nll_loss(a, a_)
        loss.backward()
        self.optim.step()
        return {'loss': loss.item()}
