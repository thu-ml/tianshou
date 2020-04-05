"""
Networks with view mask.
"""

import torch
import numpy as np
from torch import nn


class ActorWithView(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape,
                 max_action, view_mask, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(int(np.prod(state_shape)), 256),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(256, 256), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(256, int(np.prod(action_shape)))]
        self.model = nn.Sequential(*self.model)
        self._max = max_action
        self._view_mask = view_mask if isinstance(view_mask, torch.Tensor) \
            else torch.tensor(view_mask)

    def forward(self, s, **kwargs):
        s = torch.tensor(s, device=self.device, dtype=torch.float)
        s *= self._view_mask
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        logits = self._max * torch.tanh(logits)
        return logits, None


class ActorProbWithView(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape,
                 max_action, view_mask, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(int(np.prod(state_shape)), 256),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(256, 256), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(256, int(np.prod(action_shape)))
        self.sigma = nn.Linear(256, int(np.prod(action_shape)))
        self._max = max_action
        self._view_mask = view_mask

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        s *= self._view_mask
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        mu = self._max * torch.tanh(self.mu(logits))
        sigma = torch.exp(self.sigma(logits))
        return (mu, sigma), None


class CriticWithView(nn.Module):
    def __init__(self, layer_num, state_shape, view_mask, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape) + np.prod(action_shape), 256),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(256, 256), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(256, 1)]
        self.model = nn.Sequential(*self.model)
        self._view_mask = view_mask

    def forward(self, s, a=None):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        s *= self._view_mask
        if a is not None and not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        if a is None:
            logits = self.model(s)
        else:
            a = a.view(batch, -1)
            logits = self.model(torch.cat([s, a], dim=1))
        return logits
