import numpy as np
import torch
from torch import nn

from tianshou.data import to_torch


class Net(nn.Module):
    """Simple MLP backbone. For advanced usage (how to customize the network),
    please refer to :ref:`build_the_network`.

    :param concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape.
    """
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu',
                 softmax=False, concat=False, hidden_layer_size=128,
                 dualing=None, layernorm=False):
        super().__init__()
        self.device = device
        self.dualing = dualing
        self.layernorm = layernorm
        input_size = np.prod(state_shape)
        if concat:
            input_size += np.prod(action_shape)

        self.model = [
            nn.Linear(input_size, hidden_layer_size)]
        if self.layernorm:
            self.model += [nn.LayerNorm(hidden_layer_size)]
        self.model += [nn.ReLU(inplace=True)]

        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size)]
            if self.layernorm:
                self.model += [nn.LayerNorm(hidden_layer_size)]
            self.model += [nn.ReLU(inplace=True)]

        if self.dualing is None:
            if action_shape and not concat:
                self.model += [nn.Linear(hidden_layer_size,
                                         np.prod(action_shape))]
            if softmax:
                self.model += [nn.Softmax(dim=-1)]
            self.model = nn.Sequential(*self.model)
        else:
            self.model = nn.Sequential(*self.model)
            assert isinstance(self.dualing, tuple)
            q_layer_num = self.dualing[0]
            self.qvalue = []
            for i in range(q_layer_num):
                self.qvalue += [nn.Linear(hidden_layer_size,
                                          hidden_layer_size)]
                if self.layernorm:
                    self.qvalue += [nn.LayerNorm(hidden_layer_size)]
                self.qvalue += [nn.ReLU(inplace=True)]
            if action_shape and not concat:
                self.qvalue += [nn.Linear(hidden_layer_size,
                                          np.prod(action_shape))]
            if softmax:
                self.qvalue += [nn.Softmax(dim=-1)]
            self.qvalue = nn.Sequential(*self.qvalue)

            s_layer_num = self.dualing[1]
            self.svalue = []
            for i in range(s_layer_num):
                self.svalue += [nn.Linear(hidden_layer_size,
                                          hidden_layer_size)]
                if self.layernorm:
                    self.svalue += [nn.LayerNorm(hidden_layer_size)]
                self.svalue += [nn.ReLU(inplace=True)]
            if action_shape and not concat:
                self.svalue += [nn.Linear(hidden_layer_size, 1)]
            self.svalue = nn.Sequential(*self.svalue)

    def forward(self, s, state=None, info={}):
        """s -> flatten -> logits"""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        s = s.reshape(s.size(0), -1)
        if self.dualing is not None:
            logits = self.model(s)
            qvalue = self.qvalue(logits)
            advantage = qvalue - qvalue.mean(dim=1).reshape(-1, 1)
            svalue = self.svalue(logits)
            return advantage + svalue, state
        else:
            logits = self.model(s)
        return logits, state


class Recurrent(nn.Module):
    """Simple Recurrent network based on LSTM. For advanced usage (how to
    customize the network), please refer to :ref:`build_the_network`.
    """
    def __init__(self, layer_num, state_shape, action_shape,
                 device='cpu', hidden_layer_size=128):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(input_size=hidden_layer_size,
                          hidden_size=hidden_layer_size,
                          num_layers=layer_num, batch_first=True)
        self.fc1 = nn.Linear(np.prod(state_shape), hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        """In the evaluation mode, s should be with shape ``[bsz, dim]``; in
        the training mode, s should be with shape ``[bsz, len, dim]``. See the
        code and comment for more detail.
        """
        s = to_torch(s, device=self.device, dtype=torch.float32)
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            s = s.unsqueeze(-2)
        s = self.fc1(s)
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
