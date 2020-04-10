import torch
import numpy as np

from tianshou.data import Batch
from tianshou.policy import BasePolicy


class PGPolicy(BasePolicy):
    """Implementation of Vanilla Policy Gradient.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param torch.distributions.Distribution dist_fn: for computing the action.
    :param float discount_factor: in [0, 1].

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, model, optim, dist_fn=torch.distributions.Categorical,
                 discount_factor=0.99, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.dist_fn = dist_fn
        self._eps = np.finfo(np.float32).eps.item()
        assert 0 <= discount_factor <= 1, 'discount factor should in [0, 1]'
        self._gamma = discount_factor

    def process_fn(self, batch, buffer, indice):
        r"""Compute the discounted returns for each frame:

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        , where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.
        """
        batch.returns = self._vanilla_returns(batch)
        # batch.returns = self._vectorized_returns(batch)
        return batch

    def forward(self, batch, state=None, **kwargs):
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        logits, h = self.model(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        act = dist.sample()
        return Batch(logits=logits, act=act, state=h, dist=dist)

    def learn(self, batch, batch_size=None, repeat=1, **kwargs):
        losses = []
        r = batch.returns
        batch.returns = (r - r.mean()) / (r.std() + self._eps)
        for _ in range(repeat):
            for b in batch.split(batch_size):
                self.optim.zero_grad()
                dist = self(b).dist
                a = torch.tensor(b.act, device=dist.logits.device)
                r = torch.tensor(b.returns, device=dist.logits.device)
                loss = -(dist.log_prob(a) * r).sum()
                loss.backward()
                self.optim.step()
                losses.append(loss.item())
        return {'loss': losses}

    def _vanilla_returns(self, batch):
        returns = batch.rew[:]
        last = 0
        for i in range(len(returns) - 1, -1, -1):
            if not batch.done[i]:
                returns[i] += self._gamma * last
            last = returns[i]
        return returns

    def _vectorized_returns(self, batch):
        # according to my tests, it is slower than _vanilla_returns
        # import scipy.signal
        convolve = np.convolve
        # convolve = scipy.signal.convolve
        rew = batch.rew[::-1]
        batch_size = len(rew)
        gammas = self._gamma ** np.arange(batch_size)
        c = convolve(rew, gammas)[:batch_size]
        T = np.where(batch.done[::-1])[0]
        d = np.zeros_like(rew)
        d[T] += c[T] - rew[T]
        d[T[1:]] -= d[T[:-1]] * self._gamma ** np.diff(T)
        return (c - convolve(d, gammas)[:batch_size])[::-1]
