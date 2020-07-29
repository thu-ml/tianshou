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

    In paper "Human-level control through deep reinforcement learning"
    url: https://www.nature.com/articles/nature14236
    researchers use DQN to play Atari games
    In this paper, researches contribute the network as below:
    input size: 84*84, deal with m most recent frames and stacks them to
                produce the input to the (m = 4)
    First hidden convolution Layer: 32 filters with kernel size 8
                                    and stride 4, then a rectifier nonlinearity
    Second hidden convolution Layer: 64 filters of with kernel size 4
                                    and stride 2, then a rectifier nonlinearity
    Third hidden convolutional Layer: 64 filters of with kernel size 3
                                    and stride 1, then a rectifier nonlinearity
    Fourth hidden Fully-connected Layer: 512 rectifier units
    Output Fully-connected Layer: linear layer with a single output for
                                   each valid action
    """

    def __init__(self, h, w, action_shape, device='cpu'):
        super(DQN, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

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
        self.fc = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, action_shape)

    def forward(self, x, state=None, info={}):
        r"""x -> Q(x, \*)"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.fc(x.reshape(x.size(0), -1))
        return self.head(x), state
