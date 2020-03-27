import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        if action_shape:
            self.model += [nn.Linear(128, np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, state


class Actor(nn.Module):
    def __init__(self, preprocess_net, action_shape):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        logits, h = self.preprocess(s, state)
        logits = F.softmax(self.last(logits), dim=-1)
        return logits, h


class Critic(nn.Module):
    def __init__(self, preprocess_net):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, 1)

    def forward(self, s):
        logits, h = self.preprocess(s, None)
        logits = self.last(logits)
        return logits


class DQN(nn.Module):

    def __init__(self, h, w, action_shape, device='cpu'):
        super(DQN, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
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
        self.head = nn.Linear(linear_input_size, action_shape)

    def forward(self, x, state=None, info={}):
        if not isinstance(x, torch.Tensor):
            s = torch.tensor(x, device=self.device, dtype=torch.float)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1)), state