
<h1 align="center">Tianshou</h1>

![PyPI](https://img.shields.io/pypi/v/tianshou)
![Unittest](https://github.com/thu-ml/tianshou/workflows/Unittest/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/tianshou/badge/?version=latest)](https://tianshou.readthedocs.io/en/latest/?badge=latest)
[![GitHub stars](https://img.shields.io/github/stars/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/network)
[![GitHub issues](https://img.shields.io/github/issues/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/issues)
[![GitHub license](https://img.shields.io/github/license/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/blob/master/LICENSE)

**Tianshou**(天授) is a reinforcement learning platform based on pure PyTorch. Unlike existing reinforcement learning libraries, which are mainly based on TensorFlow, have many nested classes, unfriendly api, or slow-speed, Tianshou provides a fast-speed framework and pythonic api for building the deep reinforcement learning agent. The supported interface algorithms include:


- [Policy Gradient (PG)](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
- [Deep Q-Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Double DQN (DDQN)](https://arxiv.org/pdf/1509.06461.pdf)
- [Advantage Actor-Critic (A2C)](http://incompleteideas.net/book/RLbook2018.pdf)
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)

Tianshou supports parallel environment training for all algorithms as well.

Tianshou is still under development. More algorithms are going to be added and we always welcome contributions to help make Tianshou better. If you would like to contribute, please check out the [guidelines](/CONTRIBUTING.md).

## Installation

Tianshou is currently hosted on [pypi](https://pypi.org/project/tianshou/). You can simply install Tianshou with the following command:

```bash
pip3 install tianshou
```

## Documentation

The tutorials and api documentations are hosted on https://tianshou.readthedocs.io/en/latest/.

The example scripts are under [test/discrete](/test/discrete) (CartPole) and [test/continuous](/test/continuous) (Pendulum).

## Why Tianshou?

Tianshou is a lightweight but high-speed reinforcement learning platform. For example, here is a test on a laptop (i7-8750H + GTX1060). It only use 3 seconds for training a policy gradient agent on CartPole-v0 task.

![testpg](docs/_static/images/testpg.gif)

Here is the table for other algorithms and platforms:

TODO: a TABLE

Tianshou also has unit tests. Different from other platforms, **the unit tests include the agent training procedure for all of the implemented algorithms**. It will be failed when it cannot train an agent to perform well enough on limited epochs on toy scenarios. The unit tests secure the reproducibility of our platform.

## Quick start

This is an example of Policy Gradient. You can also run the full script under [test/discrete/test_pg.py](/test/discrete/test_pg.py).

First, import the relevant packages:

```python
import gym, torch, numpy as np, torch.nn as nn

from tianshou.policy import PGPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
```

Define some hyper-parameters:

```python
task = 'CartPole-v0' 
seed = 1626 
lr = 3e-4 
gamma = 0.9 
epoch = 10 
step_per_epoch = 1000 
collect_per_step = 10 
repeat_per_collect = 2 
batch_size = 64 
train_num = 8 
test_num = 100 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

Define the network:

```python
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
```

Make envs and fix seed:

```python
env = gym.make(task)
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
train_envs = SubprocVectorEnv([lambda: gym.make(task) for _ in range(train_num)])
test_envs = SubprocVectorEnv([lambda: gym.make(task) for _ in range(test_num)])
np.random.seed(seed)
torch.manual_seed(seed)
train_envs.seed(seed)
test_envs.seed(seed)
```

Setup policy and collector:

```python
net = Net(3, state_shape, action_shape, device).to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)
policy = PGPolicy(net, optim, torch.distributions.Categorical, gamma)
train_collector = Collector(policy, train_envs, ReplayBuffer(20000))
test_collector = Collector(policy, test_envs)
```

Let's train it:

```python
result = onpolicy_trainer(policy, train_collector, test_collector, epoch, step_per_epoch, collect_per_step, repeat_per_collect, test_num, batch_size, stop_fn=lambda x: x >= env.spec.reward_threshold)
```

Saving / loading trained policy (it's the same as PyTorch nn.module):

```python
torch.save(policy.state_dict(), 'pg.pth')
policy.load_state_dict(torch.load('pg.pth', map_location=device))
```

Watch the performance with 35 FPS:

```python3
collecter = Collector(policy, env)
collecter.collect(n_episode=1, render=1/35)
```

## Citing Tianshou

If you find Tianshou useful, please cite it in your publications.

```
@misc{tianshou,
  author = {Jiayi Weng},
  title = {Tianshou},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/thu-ml/tianshou}},
}
```

## Miscellaneous

Tianshou was [previously](https://github.com/thu-ml/tianshou/tree/priv) a reinforcement learning platform based on TensorFlow. You can checkout the branch `priv` for more detail.