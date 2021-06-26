<div align="center">
  <a href="http://tianshou.readthedocs.io"><img width="300px" height="auto" src="docs/_static/images/tianshou-logo.png"></a>
</div>

---

[![PyPI](https://img.shields.io/pypi/v/tianshou)](https://pypi.org/project/tianshou/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/tianshou)](https://github.com/conda-forge/tianshou-feedstock)
[![Read the Docs](https://img.shields.io/readthedocs/tianshou)](https://tianshou.readthedocs.io/en/master)
[![Read the Docs](https://img.shields.io/readthedocs/tianshou-docs-zh-cn?label=%E4%B8%AD%E6%96%87%E6%96%87%E6%A1%A3)](https://tianshou.readthedocs.io/zh/latest/)
[![Unittest](https://github.com/thu-ml/tianshou/workflows/Unittest/badge.svg?branch=master)](https://github.com/thu-ml/tianshou/actions)
[![codecov](https://img.shields.io/codecov/c/gh/thu-ml/tianshou)](https://codecov.io/gh/thu-ml/tianshou)
[![GitHub issues](https://img.shields.io/github/issues/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/issues)
[![GitHub stars](https://img.shields.io/github/stars/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/network)
[![GitHub license](https://img.shields.io/github/license/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/blob/master/LICENSE)

**Tianshou** ([天授](https://baike.baidu.com/item/%E5%A4%A9%E6%8E%88)) is a reinforcement learning platform based on pure PyTorch. Unlike existing reinforcement learning libraries, which are mainly based on TensorFlow, have many nested classes, unfriendly API, or slow-speed, Tianshou provides a fast-speed modularized framework and pythonic API for building the deep reinforcement learning agent with the least number of lines of code. The supported interface algorithms currently include:


- [Deep Q-Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)
- [Dueling DQN](https://arxiv.org/pdf/1511.06581.pdf)
- [Categorical DQN (C51)](https://arxiv.org/pdf/1707.06887.pdf)
- [Quantile Regression DQN (QRDQN)](https://arxiv.org/pdf/1710.10044.pdf)
- [Implicit Quantile Network (IQN)](https://arxiv.org/pdf/1806.06923.pdf)
- [Fully-parameterized Quantile Function (FQF)](https://arxiv.org/pdf/1911.02140.pdf)
- [Policy Gradient (PG)](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
- [Natural Policy Gradient (NPG)](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)
- [Advantage Actor-Critic (A2C)](https://openai.com/blog/baselines-acktr-a2c/)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477.pdf)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)
- [Discrete Soft Actor-Critic (SAC-Discrete)](https://arxiv.org/pdf/1910.07207.pdf)
- Vanilla Imitation Learning
- [Discrete Batch-Constrained deep Q-Learning (BCQ-Discrete)](https://arxiv.org/pdf/1910.01708.pdf)
- [Discrete Conservative Q-Learning (CQL-Discrete)](https://arxiv.org/pdf/2006.04779.pdf)
- [Discrete Critic Regularized Regression (CRR-Discrete)](https://arxiv.org/pdf/2006.15134.pdf)
- [Prioritized Experience Replay (PER)](https://arxiv.org/pdf/1511.05952.pdf)
- [Generalized Advantage Estimator (GAE)](https://arxiv.org/pdf/1506.02438.pdf)
- [Posterior Sampling Reinforcement Learning (PSRL)](https://www.ece.uvic.ca/~bctill/papers/learning/Strens_2000.pdf)

Here is Tianshou's other features:

- Elegant framework, using only ~3000 lines of code
- State-of-the-art [MuJoCo benchmark](https://github.com/thu-ml/tianshou/tree/master/examples/mujoco) for REINFORCE/A2C/TRPO/PPO/DDPG/TD3/SAC algorithms
- Support parallel environment simulation (synchronous or asynchronous) for all algorithms [Usage](https://tianshou.readthedocs.io/en/latest/tutorials/cheatsheet.html#parallel-sampling)
- Support recurrent state representation in actor network and critic network (RNN-style training for POMDP) [Usage](https://tianshou.readthedocs.io/en/latest/tutorials/cheatsheet.html#rnn-style-training)
- Support any type of environment state/action (e.g. a dict, a self-defined class, ...) [Usage](https://tianshou.readthedocs.io/en/latest/tutorials/cheatsheet.html#user-defined-environment-and-different-state-representation)
- Support customized training process [Usage](https://tianshou.readthedocs.io/en/latest/tutorials/cheatsheet.html#customize-training-process)
- Support n-step returns estimation and prioritized experience replay for all Q-learning based algorithms; GAE, nstep and PER are very fast thanks to numba jit function and vectorized numpy operation
- Support multi-agent RL [Usage](https://tianshou.readthedocs.io/en/latest/tutorials/cheatsheet.html##multi-agent-reinforcement-learning)
- Comprehensive documentation, PEP8 code-style checking, type checking and [unit tests](https://github.com/thu-ml/tianshou/actions)

In Chinese, Tianshou means divinely ordained and is derived to the gift of being born with. Tianshou is a reinforcement learning platform, and the RL algorithm does not learn from humans. So taking "Tianshou" means that there is no teacher to study with, but rather to learn by themselves through constant interaction with the environment.

“天授”意指上天所授，引申为与生具有的天赋。天授是强化学习平台，而强化学习算法并不是向人类学习的，所以取“天授”意思是没有老师来教，而是自己通过跟环境不断交互来进行学习。

## Installation

Tianshou is currently hosted on [PyPI](https://pypi.org/project/tianshou/) and [conda-forge](https://github.com/conda-forge/tianshou-feedstock). It requires Python >= 3.6.

You can simply install Tianshou from PyPI with the following command:

```bash
$ pip install tianshou
```

If you use Anaconda or Miniconda, you can install Tianshou from conda-forge through the following command:

```bash
$ conda -c conda-forge install tianshou
```

You can also install with the newest version through GitHub:

```bash
$ pip install git+https://github.com/thu-ml/tianshou.git@master --upgrade
```

After installation, open your python console and type

```python
import tianshou
print(tianshou.__version__)
```

If no error occurs, you have successfully installed Tianshou.

## Documentation

The tutorials and API documentation are hosted on [tianshou.readthedocs.io](https://tianshou.readthedocs.io/).

The example scripts are under [test/](https://github.com/thu-ml/tianshou/blob/master/test) folder and [examples/](https://github.com/thu-ml/tianshou/blob/master/examples) folder.

中文文档位于 [https://tianshou.readthedocs.io/zh/latest/](https://tianshou.readthedocs.io/zh/latest/)。

<!-- 这里有一份天授平台简短的中文简介：https://www.zhihu.com/question/377263715 -->

## Why Tianshou?

### Fast-speed

Tianshou is a lightweight but high-speed reinforcement learning platform. For example, here is a test on a laptop (i7-8750H + GTX1060). It only uses 3 seconds for training an agent based on vanilla policy gradient on the CartPole-v0 task: (seed may be different across different platform and device)

```bash
$ python3 test/discrete/test_pg.py --seed 0 --render 0.03
```

<div align="center">
  <img src="docs/_static/images/testpg.gif"></a>
</div>

We select some of famous reinforcement learning platforms: 2 GitHub repos with most stars in all RL platforms (OpenAI Baseline and RLlib) and 2 GitHub repos with most stars in PyTorch RL platforms (PyTorch DRL and rlpyt). Here is the benchmark result for other algorithms and platforms on toy scenarios: (tested on the same laptop as mentioned above)

| RL Platform     | [Tianshou](https://github.com/thu-ml/tianshou)               | [Baselines](https://github.com/openai/baselines)             | [Stable-Baselines](https://github.com/hill-a/stable-baselines) | [Ray/RLlib](https://github.com/ray-project/ray/tree/master/rllib/) | [PyTorch-DRL](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch) | [rlpyt](https://github.com/astooke/rlpyt)                    |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| GitHub Stars    | [![GitHub stars](https://img.shields.io/github/stars/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/stargazers) | [![GitHub stars](https://img.shields.io/github/stars/openai/baselines)](https://github.com/openai/baselines/stargazers) | [![GitHub stars](https://img.shields.io/github/stars/hill-a/stable-baselines)](https://github.com/hill-a/stable-baselines/stargazers) | [![GitHub stars](https://img.shields.io/github/stars/ray-project/ray)](https://github.com/ray-project/ray/stargazers) | [![GitHub stars](https://img.shields.io/github/stars/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/stargazers) | [![GitHub stars](https://img.shields.io/github/stars/astooke/rlpyt)](https://github.com/astooke/rlpyt/stargazers) |
| Algo - Task     | PyTorch                                                      | TensorFlow                                                   | TensorFlow                                                   | TF/PyTorch                                                   | PyTorch                                                      | PyTorch                                                      |
| PG - CartPole   | 9.02±6.79s                                                   | None                                                         | None                                                         | 19.26±2.29s                                                  | None                                                         | ?                                                            |
| DQN - CartPole  | 6.72±1.28s                                                   | 1046.34±291.27s                                              | 93.47±58.05s                                                 | 28.56±4.60s                                                  | 31.58±11.30s \*\*                                            | ?                                                            |
| A2C - CartPole  | 15.33±4.48s                                                  | \*(~1612s)                                                   | 57.56±12.87s                                                 | 57.92±9.94s                                                  | \*(Not converged)                                            | ?                                                            |
| PPO - CartPole  | 6.01±1.14s                                                   | \*(~1179s)                                                   | 34.79±17.02s                                                 | 44.60±17.04s                                                 | 23.99±9.26s \*\*                                             | ?                                                            |
| PPO - Pendulum  | 16.18±2.49s                                                  | 745.43±160.82s                                               | 259.73±27.37s                                                | 123.62±44.23s                                                | Runtime Error                                                | ?                                                            |
| DDPG - Pendulum | 37.26±9.55s                                                  | \*(>1h)                                                      | 277.52±92.67s                                                | 314.70±7.92s                                                 | 59.05±10.03s \*\*                                            | 172.18±62.48s                                                |
| TD3 - Pendulum  | 44.04±6.37s                                                  | None                                                         | 99.75±21.63s                                                 | 149.90±7.54s                                                 | 57.52±17.71s \*\*                                            | 210.31±76.30s                                                |
| SAC - Pendulum  | 36.02±0.77s                                                  | None                                                         | 124.85±79.14s                                                | 97.42±4.75s                                                  | 63.80±27.37s \*\*                                            | 295.92±140.85s                                               |

*\*: Could not reach the target reward threshold in 1e6 steps in any of 5 runs. The total runtime is in the brackets.*

*\*\*: Since no specific evaluation function is implemented in PyTorch-DRL, the condition is relaxed to "The average total reward for 20 consecutive complete games during training is greater than or equal to threshold".*

*?: We have tried but it is nontrivial for running non-Atari game on rlpyt. See [here](https://github.com/astooke/rlpyt/issues/135).*

All of the platforms use 5 different seeds for testing. We erase those trials which failed for training. The reward threshold is 195.0 in CartPole and -250.0 in Pendulum over consecutive 100 episodes' mean returns (except for PyTorch-DRL). 

The Atari/Mujoco benchmark results are under [examples/atari/](examples/atari/) and [examples/mujoco/](examples/mujoco/) folders.

### Reproducible

Tianshou has its unit tests. Different from other platforms, **the unit tests include the full agent training procedure for all of the implemented algorithms**. It would be failed once if it could not train an agent to perform well enough on limited epochs on toy scenarios. The unit tests secure the reproducibility of our platform. 

Check out the [GitHub Actions](https://github.com/thu-ml/tianshou/actions) page for more detail.

### Modularized Policy

We decouple all of the algorithms roughly into the following parts:

- `__init__`: initialize the policy;
- `forward`: to compute actions over given observations;
- `process_fn`: to preprocess data from replay buffer (since we have reformulated all algorithms to replay-buffer based algorithms);
- `learn`: to learn from a given batch data;
- `post_process_fn`: to update the replay buffer from the learning process (e.g., prioritized replay buffer needs to update the weight);
- `update`: the main interface for training, i.e., `process_fn -> learn -> post_process_fn`.

Within this API, we can interact with different policies conveniently.

### Elegant and Flexible

Currently, the overall code of Tianshou platform is less than 2500 lines. Most of the implemented algorithms are less than 100 lines of python code. It is quite easy to go through the framework and understand how it works. We provide many flexible API as you wish, for instance, if you want to use your policy to interact with the environment with (at least) `n` steps:

```python
result = collector.collect(n_step=n)
```
If you have 3 environments in total and want to collect 4 episodes:

```python
result = collector.collect(n_episode=4)
```

Collector will collect exactly 4 episodes without any bias of episode length despite we only have 3 parallel environments.

If you want to train the given policy with a sampled batch:

```python
result = policy.update(batch_size, collector.buffer)
```

You can check out the [documentation](https://tianshou.readthedocs.io) for further usage.

## Quick Start

This is an example of Deep Q Network. You can also run the full script at [test/discrete/test_dqn.py](https://github.com/thu-ml/tianshou/blob/master/test/discrete/test_dqn.py).

First, import some relevant packages:

```python
import gym, torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
```

Define some hyper-parameters:

```python
task = 'CartPole-v0'
lr, epoch, batch_size = 1e-3, 10, 64
train_num, test_num = 10, 100
gamma, n_step, target_freq = 0.9, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_epoch, step_per_collect = 10000, 10
writer = SummaryWriter('log/dqn')  # tensorboard is also supported!
logger = ts.utils.BasicLogger(writer)
```

Make environments:

```python
# you can also try with SubprocVectorEnv
train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])
```

Define the network:

```python
from tianshou.utils.net.common import Net
# you can define other net by following the API:
# https://tianshou.readthedocs.io/en/latest/tutorials/dqn.html#build-the-network
env = gym.make(task)
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
optim = torch.optim.Adam(net.parameters(), lr=lr)
```

Setup policy and collectors:

```python
policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method
```

Let's train it:

```python
result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
    test_num, batch_size, update_per_step=1 / step_per_collect,
    train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
    logger=logger)
print(f'Finished training! Use {result["duration"]}')
```

Save / load the trained policy (it's exactly the same as PyTorch `nn.module`):

```python
torch.save(policy.state_dict(), 'dqn.pth')
policy.load_state_dict(torch.load('dqn.pth'))
```

Watch the performance with 35 FPS:

```python
policy.eval()
policy.set_eps(eps_test)
collector = ts.data.Collector(policy, env, exploration_noise=True)
collector.collect(n_episode=1, render=1 / 35)
```

Look at the result saved in tensorboard: (with bash script in your terminal)

```bash
$ tensorboard --logdir log/dqn
```

You can check out the [documentation](https://tianshou.readthedocs.io) for advanced usage.

## Contributing

Tianshou is still under development. More algorithms and features are going to be added and we always welcome contributions to help make Tianshou better. If you would like to contribute, please check out [this link](https://tianshou.readthedocs.io/en/latest/contributing.html).

## TODO

Check out the [Project](https://github.com/thu-ml/tianshou/projects) page for more detail.

## Citing Tianshou

If you find Tianshou useful, please cite it in your publications.

```latex
@misc{tianshou,
  author = {Jiayi Weng, Huayu Chen, Alexis Duburcq, Kaichao You, Minghao Zhang, Dong Yan, Hang Su, Jun Zhu},
  title = {Tianshou},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/thu-ml/tianshou}},
}
```

## Acknowledgment

Tianshou was previously a reinforcement learning platform based on TensorFlow. You can check out the branch [`priv`](https://github.com/thu-ml/tianshou/tree/priv) for more detail. Many thanks to [Haosheng Zou](https://github.com/HaoshengZou)'s pioneering work for Tianshou before version 0.1.1.

We would like to thank [TSAIL](http://ml.cs.tsinghua.edu.cn/) and [Institute for Artificial Intelligence, Tsinghua University](http://ml.cs.tsinghua.edu.cn/thuai/) for providing such an excellent AI research platform.
