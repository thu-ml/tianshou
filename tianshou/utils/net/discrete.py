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
    """

    def __init__(self, h, w, action_shape, device='cpu'):
        super(DQN, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.fc = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, action_shape)

    def forward(self, x, state=None, info={}):
        r"""x -> Q(x, \*)"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.fc(x.reshape(x.size(0), -1))
        return self.head(x), state
