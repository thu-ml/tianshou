import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.policy import BasePolicy


class DQNPolicy(BasePolicy):
    """docstring for DQNPolicy"""

    def __init__(self, model, optim, discount_factor=0.99,
                 estimation_step=1, use_target_network=True,
                 target_update_freq=300):
        super().__init__()
        self.model = model
        self.optim = optim
        self.eps = 0
        assert 0 < discount_factor <= 1, 'discount_factor should in (0, 1]'
        self._gamma = discount_factor
        assert estimation_step > 0, 'estimation_step should greater than 0'
        self._n_step = estimation_step
        self._target = use_target_network
        self._freq = target_update_freq
        self._cnt = 0
        if use_target_network:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()

    def set_eps(self, eps):
        self.eps = eps

    def train(self):
        self.training = True
        self.model.train()

    def eval(self):
        self.training = False
        self.model.eval()

    def sync_weight(self):
        self.model_old.load_state_dict(self.model.state_dict())

    def process_fn(self, batch, buffer, indice):
        returns = np.zeros_like(indice)
        gammas = np.zeros_like(indice) + self._n_step
        for n in range(self._n_step - 1, -1, -1):
            now = (indice + n) % len(buffer)
            gammas[buffer.done[now] > 0] = n
            returns[buffer.done[now] > 0] = 0
            returns = buffer.rew[now] + self._gamma * returns
        terminal = (indice + self._n_step - 1) % len(buffer)
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            a = self(buffer[terminal], input='obs_next', eps=0).act
            target_q = self(
                buffer[terminal], model='model_old', input='obs_next').logits
            if isinstance(target_q, torch.Tensor):
                target_q = target_q.detach().cpu().numpy()
            target_q = target_q[np.arange(len(a)), a]
        else:
            target_q = self(buffer[terminal], input='obs_next').logits
            if isinstance(target_q, torch.Tensor):
                target_q = target_q.detach().cpu().numpy()
            target_q = target_q.max(axis=1)
        target_q[gammas != self._n_step] = 0
        returns += (self._gamma ** gammas) * target_q
        batch.update(returns=returns)
        return batch

    def __call__(self, batch, state=None,
                 model='model', input='obs', eps=None):
        model = getattr(self, model)
        obs = getattr(batch, input)
        q, h = model(obs, state=state, info=batch.info)
        act = q.max(dim=1)[1].detach().cpu().numpy()
        # add eps to act
        if eps is None:
            eps = self.eps
        for i in range(len(q)):
            if np.random.rand() < eps:
                act[i] = np.random.randint(q.shape[1])
        return Batch(logits=q, act=act, state=h)

    def learn(self, batch, batch_size=None, repeat=1):
        if self._target and self._cnt % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        r = batch.returns
        if isinstance(r, np.ndarray):
            r = torch.tensor(r, device=q.device, dtype=q.dtype)
        loss = F.mse_loss(q, r)
        loss.backward()
        self.optim.step()
        self._cnt += 1
        return {'loss': loss.detach().cpu().numpy()}
