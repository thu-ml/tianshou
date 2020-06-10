import torch
import numpy as np
from typing import Dict, List, Union, Optional

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as


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

    def __init__(self,
                 model: torch.nn.Module,
                 optim: torch.optim.Optimizer,
                 dist_fn: torch.distributions.Distribution
                 = torch.distributions.Categorical,
                 discount_factor: float = 0.99,
                 reward_normalization: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.dist_fn = dist_fn
        assert 0 <= discount_factor <= 1, 'discount factor should in [0, 1]'
        self._gamma = discount_factor
        self._rew_norm = reward_normalization

    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indice: np.ndarray) -> Batch:
        r"""Compute the discounted returns for each frame:

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        , where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.
        """
        # batch.returns = self._vanilla_returns(batch)
        # batch.returns = self._vectorized_returns(batch)
        # return batch
        return self.compute_episodic_return(
            batch, gamma=self._gamma, gae_lambda=1.)

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                **kwargs) -> Batch:
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

    def learn(self, batch: Batch, batch_size: int, repeat: int,
              **kwargs) -> Dict[str, List[float]]:
        losses = []
        r = batch.returns
        if self._rew_norm and not np.isclose(r.std(), 0):
            batch.returns = (r - r.mean()) / r.std()
        for _ in range(repeat):
            for b in batch.split(batch_size):
                self.optim.zero_grad()
                dist = self(b).dist
                a = to_torch_as(b.act, dist.logits)
                r = to_torch_as(b.returns, dist.logits)
                loss = -(dist.log_prob(a) * r).sum()
                loss.backward()
                self.optim.step()
                losses.append(loss.item())
        return {'loss': losses}

    # def _vanilla_returns(self, batch):
    #     returns = batch.rew[:]
    #     last = 0
    #     for i in range(len(returns) - 1, -1, -1):
    #         if not batch.done[i]:
    #             returns[i] += self._gamma * last
    #         last = returns[i]
    #     return returns

    # def _vectorized_returns(self, batch):
    #     # according to my tests, it is slower than _vanilla_returns
    #     # import scipy.signal
    #     convolve = np.convolve
    #     # convolve = scipy.signal.convolve
    #     rew = batch.rew[::-1]
    #     batch_size = len(rew)
    #     gammas = self._gamma ** np.arange(batch_size)
    #     c = convolve(rew, gammas)[:batch_size]
    #     T = np.where(batch.done[::-1])[0]
    #     d = np.zeros_like(rew)
    #     d[T] += c[T] - rew[T]
    #     d[T[1:]] -= d[T[:-1]] * self._gamma ** np.diff(T)
    #     return (c - convolve(d, gammas)[:batch_size])[::-1]
