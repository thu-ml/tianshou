import torch
import numpy as np
from torch import nn


class Actor(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape,
                 max_action, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(128, np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)
        self._max = max_action

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        logits = self._max * torch.tanh(logits)
        return logits, None


class ActorProb(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape,
                 max_action, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(128, np.prod(action_shape))
        self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))
        # self.sigma = nn.Linear(128, np.prod(action_shape))
        self._max = max_action

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        mu = self.mu(logits)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        # assert sigma.shape == mu.shape
        # mu = self._max * torch.tanh(self.mu(logits))
        # sigma = torch.exp(self.sigma(logits))
        return (mu, sigma), None


class Critic(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape) + np.prod(action_shape), 128),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(128, 1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, a=None, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            a = a.view(batch, -1)
            s = torch.cat([s, a], dim=1)
        logits = self.model(s)
        return logits


class RecurrentActorProb(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape,
                 max_action, device='cpu'):
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(input_size=np.prod(state_shape), hidden_size=128,
                          num_layers=layer_num, batch_first=True)
        self.mu = nn.Linear(128, np.prod(action_shape))
        self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            bsz, dim = s.shape
            length = 1
        else:
            bsz, length, dim = s.shape
        s = s.view(bsz, length, -1)
        logits, _ = self.nn(s)
        logits = logits[:, -1]
        mu = self.mu(logits)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), None


class RecurrentCritic(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(input_size=np.prod(state_shape), hidden_size=128,
                          num_layers=layer_num, batch_first=True)
        self.fc2 = nn.Linear(128 + np.prod(action_shape), 1)

    def forward(self, s, a=None):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        assert len(s.shape) == 3
        self.nn.flatten_parameters()
        s, (h, c) = self.nn(s)
        s = s[:, -1]
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            s = torch.cat([s, a], dim=1)
        s = self.fc2(s)
        return s
