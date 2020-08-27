import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Actor(nn.Module):
    """For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, preprocess_net, action_shape, hidden_layer_size=128):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(hidden_layer_size, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        r"""s -> Q(s, \*)"""
        logits, h = self.preprocess(s, state)
        logits = F.softmax(self.last(logits), dim=-1)
        return logits, h


class Critic(nn.Module):
    """For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, preprocess_net, hidden_layer_size=128):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(hidden_layer_size, 1)

    def forward(self, s, **kwargs):
        """s -> V(s)"""
        logits, h = self.preprocess(s, state=kwargs.get('state', None))
        logits = self.last(logits)
        return logits


class DQN(nn.Module):
    """For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    Reference paper: "Human-level control through deep reinforcement learning".
    """

    def __init__(self, c, h, w, action_shape, device='cpu'):
        super(DQN, self).__init__()
        self.device = device

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def conv2d_layers_size_out(size,
                                   kernel_size_1=8, stride_1=4,
                                   kernel_size_2=4, stride_2=2,
                                   kernel_size_3=3, stride_3=1):
            size = conv2d_size_out(size, kernel_size_1, stride_1)
            size = conv2d_size_out(size, kernel_size_2, stride_2)
            size = conv2d_size_out(size, kernel_size_3, stride_3)
            return size

        convw = conv2d_layers_size_out(w)
        convh = conv2d_layers_size_out(h)
        linear_input_size = convw * convh * 64

        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(linear_input_size, 512),
            nn.Linear(512, np.prod(action_shape))
        )

    def forward(self, x, state=None, info={}):
        r"""x -> Q(x, \*)"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        return self.net(x), state
