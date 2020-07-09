"""
Commonly used MLP-backbone.
"""
import numpy as np
import torch
from torch import nn

from tianshou.data import to_torch


class Net(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu',
                 softmax=False, concat=False):
        """
        Simple MLP backbone.
        :param concat: whether the input shape is concatenated by state_shape
         and action_shape. If it is True, ``action_shape`` is not the output
         shape, but affects the input shape.
        """
        super().__init__()
        self.device = device
        input_size = np.prod(state_shape)
        if concat:
            input_size += np.prod(action_shape)
        self.model = [
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        if action_shape and not concat:
            self.model += [nn.Linear(128, np.prod(action_shape))]
        if softmax:
            self.model += [nn.Softmax(dim=-1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, state=None, info={}):
        s = to_torch(s, device=self.device, dtype=torch.float32)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, state


class Recurrent(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape, device='cpu'):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(input_size=128, hidden_size=128,
                          num_layers=layer_num, batch_first=True)
        self.fc1 = nn.Linear(np.prod(state_shape), 128)
        self.fc2 = nn.Linear(128, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        s = to_torch(s, device=self.device, dtype=torch.float32)
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            bsz, dim = s.shape
            length = 1
        else:
            bsz, length, dim = s.shape
        s = self.fc1(s.view([bsz * length, dim]))
        s = s.view(bsz, length, -1)
        self.nn.flatten_parameters()
        if state is None:
            s, (h, c) = self.nn(s)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            s, (h, c) = self.nn(s, (state['h'].transpose(0, 1).contiguous(),
                                    state['c'].transpose(0, 1).contiguous()))
        s = self.fc2(s[:, -1])
        # please ensure the first dim is batch size: [bsz, len, ...]
        return s, {'h': h.transpose(0, 1).detach(),
                   'c': c.transpose(0, 1).detach()}
