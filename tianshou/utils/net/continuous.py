import torch
import numpy as np
from torch import nn

from tianshou.data import to_torch


class Actor(nn.Module):
    """For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, preprocess_net, action_shape,
                 max_action, device='cpu', hidden_layer_size=128):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self._max = max_action

    def forward(self, s, state=None, info={}):
        """s -> logits -> action"""
        logits, h = self.preprocess(s, state)
        logits = self._max * torch.tanh(self.last(logits))
        return logits, h


class Critic(nn.Module):
    """For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, preprocess_net, device='cpu', hidden_layer_size=128):
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.last = nn.Linear(hidden_layer_size, 1)

    def forward(self, s, a=None, **kwargs):
        """(s, a) -> logits -> Q(s, a)"""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        s = s.flatten(1)
        if a is not None:
            a = to_torch(a, device=self.device, dtype=torch.float32)
            a = a.flatten(1)
            s = torch.cat([s, a], dim=1)
        logits, h = self.preprocess(s)
        logits = self.last(logits)
        return logits


class ActorProb(nn.Module):
    """For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, preprocess_net, action_shape, max_action,
                 device='cpu', unbounded=False, hidden_layer_size=128):
        super().__init__()
        self.preprocess = preprocess_net
        self.device = device
        self.mu = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))
        self._max = max_action
        self._unbounded = unbounded

    def forward(self, s, state=None, **kwargs):
        """s -> logits -> (mu, sigma)"""
        logits, h = self.preprocess(s, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), None


class RecurrentActorProb(nn.Module):
    """For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, layer_num, state_shape, action_shape,
                 max_action, device='cpu', hidden_layer_size=128):
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(input_size=np.prod(state_shape),
                          hidden_size=hidden_layer_size,
                          num_layers=layer_num, batch_first=True)
        self.mu = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))

    def forward(self, s, **kwargs):
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            s = s.unsqueeze(-2)
        logits, _ = self.nn(s)
        logits = logits[:, -1]
        mu = self.mu(logits)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), None


class RecurrentCritic(nn.Module):
    """For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, layer_num, state_shape,
                 action_shape=0, device='cpu', hidden_layer_size=128):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(input_size=np.prod(state_shape),
                          hidden_size=hidden_layer_size,
                          num_layers=layer_num, batch_first=True)
        self.fc2 = nn.Linear(hidden_layer_size + np.prod(action_shape), 1)

    def forward(self, s, a=None):
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        assert len(s.shape) == 3
        self.nn.flatten_parameters()
        s, (h, c) = self.nn(s)
        s = s[:, -1]
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float32)
            s = torch.cat([s, a], dim=1)
        s = self.fc2(s)
        return s
