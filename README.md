<div align="center">
  <a href="http://tianshou.readthedocs.io"><img width="300px" height="auto" src="https://github.com/thu-ml/tianshou/raw/master/docs/_static/images/tianshou-logo.png"></a>
</div>

---

[![PyPI](https://img.shields.io/pypi/v/tianshou)](https://pypi.org/project/tianshou/) [![Conda](https://img.shields.io/conda/vn/conda-forge/tianshou)](https://github.com/conda-forge/tianshou-feedstock) [![Read the Docs](https://img.shields.io/readthedocs/tianshou)](https://tianshou.readthedocs.io/en/master) [![Read the Docs](https://img.shields.io/readthedocs/tianshou-docs-zh-cn?label=%E4%B8%AD%E6%96%87%E6%96%87%E6%A1%A3)](https://tianshou.readthedocs.io/zh/master/) [![Unittest](https://github.com/thu-ml/tianshou/workflows/Unittest/badge.svg?branch=master)](https://github.com/thu-ml/tianshou/actions) [![codecov](https://img.shields.io/codecov/c/gh/thu-ml/tianshou)](https://codecov.io/gh/thu-ml/tianshou) [![GitHub issues](https://img.shields.io/github/issues/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/issues) [![GitHub stars](https://img.shields.io/github/stars/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/stargazers) [![GitHub forks](https://img.shields.io/github/forks/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/network) [![GitHub license](https://img.shields.io/github/license/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/blob/master/LICENSE)

**Tianshou** ([天授](https://baike.baidu.com/item/%E5%A4%A9%E6%8E%88)) is a reinforcement learning platform based on pure PyTorch. Unlike existing reinforcement learning libraries, which are mainly based on TensorFlow, have many nested classes, unfriendly API, or slow-speed, Tianshou provides a fast-speed modularized framework and pythonic API for building the deep reinforcement learning agent with the least number of lines of code. The supported interface algorithms currently include:


- [Deep Q-Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)
- [Dueling DQN](https://arxiv.org/pdf/1511.06581.pdf)
- [Branching DQN](https://arxiv.org/pdf/1711.08946.pdf)
- [Categorical DQN (C51)](https://arxiv.org/pdf/1707.06887.pdf)
- [Rainbow DQN (Rainbow)](https://arxiv.org/pdf/1710.02298.pdf)
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
- [Randomized Ensembled Double Q-Learning (REDQ)](https://arxiv.org/pdf/2101.05982.pdf)
- [Discrete Soft Actor-Critic (SAC-Discrete)](https://arxiv.org/pdf/1910.07207.pdf)
- Vanilla Imitation Learning
- [Batch-Constrained deep Q-Learning (BCQ)](https://arxiv.org/pdf/1812.02900.pdf)
- [Conservative Q-Learning (CQL)](https://arxiv.org/pdf/2006.04779.pdf)
- [Twin Delayed DDPG with Behavior Cloning (TD3+BC)](https://arxiv.org/pdf/2106.06860.pdf)
- [Discrete Batch-Constrained deep Q-Learning (BCQ-Discrete)](https://arxiv.org/pdf/1910.01708.pdf)
- [Discrete Conservative Q-Learning (CQL-Discrete)](https://arxiv.org/pdf/2006.04779.pdf)
- [Discrete Critic Regularized Regression (CRR-Discrete)](https://arxiv.org/pdf/2006.15134.pdf)
- [Generative Adversarial Imitation Learning (GAIL)](https://arxiv.org/pdf/1606.03476.pdf)
- [Prioritized Experience Replay (PER)](https://arxiv.org/pdf/1511.05952.pdf)
- [Generalized Advantage Estimator (GAE)](https://arxiv.org/pdf/1506.02438.pdf)
- [Posterior Sampling Reinforcement Learning (PSRL)](https://www.ece.uvic.ca/~bctill/papers/learning/Strens_2000.pdf)
- [Intrinsic Curiosity Module (ICM)](https://arxiv.org/pdf/1705.05363.pdf)

Here are Tianshou's other features:

- Elegant framework, using only ~4000 lines of code
- State-of-the-art [MuJoCo benchmark](https://github.com/thu-ml/tianshou/tree/master/examples/mujoco) for REINFORCE/A2C/TRPO/PPO/DDPG/TD3/SAC algorithms
- Support vectorized environment (synchronous or asynchronous) for all algorithms [Usage](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#parallel-sampling)
- Support super-fast vectorized environment [EnvPool](https://github.com/sail-sg/envpool/) for all algorithms [Usage](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#envpool-integration)
- Support recurrent state representation in actor network and critic network (RNN-style training for POMDP) [Usage](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#rnn-style-training)
- Support any type of environment state/action (e.g. a dict, a self-defined class, ...) [Usage](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#user-defined-environment-and-different-state-representation)
- Support customized training process [Usage](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#customize-training-process)
- Support n-step returns estimation and prioritized experience replay for all Q-learning based algorithms; GAE, nstep and PER are very fast thanks to numba jit function and vectorized numpy operation
- Support multi-agent RL [Usage](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#multi-agent-reinforcement-learning)
- Support both [TensorBoard](https://www.tensorflow.org/tensorboard) and [W&B](https://wandb.ai/) log tools
- Support multi-GPU training [Usage](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#multi-gpu)
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
$ conda install -c conda-forge tianshou
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

中文文档位于 [https://tianshou.readthedocs.io/zh/master/](https://tianshou.readthedocs.io/zh/master/)。

<!-- 这里有一份天授平台简短的中文简介：https://www.zhihu.com/question/377263715 -->

## Why Tianshou?

### Comprehensive Functionality

| RL Platform                                                  | GitHub Stars                                                 | # of Alg. <sup>(1)</sup> | Custom Env                  | Batch Training                    | RNN Support        | Nested Observation | Backend    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------ | --------------------------- | --------------------------------- | ------------------ | ------------------ | ---------- |
| [Baselines](https://github.com/openai/baselines)             | [![GitHub stars](https://img.shields.io/github/stars/openai/baselines)](https://github.com/openai/baselines/stargazers) | 9                        | :heavy_check_mark: (gym)    | :heavy_minus_sign: <sup>(2)</sup> | :heavy_check_mark: | :x:                | TF1        |
| [Stable-Baselines](https://github.com/hill-a/stable-baselines) | [![GitHub stars](https://img.shields.io/github/stars/hill-a/stable-baselines)](https://github.com/hill-a/stable-baselines/stargazers) | 11                       | :heavy_check_mark: (gym)    | :heavy_minus_sign: <sup>(2)</sup> | :heavy_check_mark: | :x:                | TF1        |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | [![GitHub stars](https://img.shields.io/github/stars/DLR-RM/stable-baselines3)](https://github.com/DLR-RM/stable-baselines3/stargazers) | 7<sup> (3)</sup>         | :heavy_check_mark: (gym)    | :heavy_minus_sign: <sup>(2)</sup> | :x:                | :heavy_check_mark: | PyTorch    |
| [Ray/RLlib](https://github.com/ray-project/ray/tree/master/rllib/) | [![GitHub stars](https://img.shields.io/github/stars/ray-project/ray)](https://github.com/ray-project/ray/stargazers) | 16                       | :heavy_check_mark:          | :heavy_check_mark:                | :heavy_check_mark: | :heavy_check_mark: | TF/PyTorch |
| [SpinningUp](https://github.com/openai/spinningup)           | [![GitHub stars](https://img.shields.io/github/stars/openai/spinningup)](https://github.com/openai/spinningupstargazers) | 6                        | :heavy_check_mark: (gym)    | :heavy_minus_sign: <sup>(2)</sup> | :x:                | :x:                | PyTorch    |
| [Dopamine](https://github.com/google/dopamine)               | [![GitHub stars](https://img.shields.io/github/stars/google/dopamine)](https://github.com/google/dopamine/stargazers) | 7                        | :x:                         | :x:                               | :x:                | :x:                | TF/JAX     |
| [ACME](https://github.com/deepmind/acme)                     | [![GitHub stars](https://img.shields.io/github/stars/deepmind/acme)](https://github.com/deepmind/acme/stargazers) | 14                       | :heavy_check_mark: (dm_env) | :heavy_check_mark:                | :heavy_check_mark: | :heavy_check_mark: | TF/JAX     |
| [keras-rl](https://github.com/keras-rl/keras-rl)             | [![GitHub stars](https://img.shields.io/github/stars/keras-rl/keras-rl)](https://github.com/keras-rl/keras-rlstargazers) | 7                        | :heavy_check_mark: (gym)    | :x:                               | :x:                | :x:                | Keras      |
| [rlpyt](https://github.com/astooke/rlpyt)                    | [![GitHub stars](https://img.shields.io/github/stars/astooke/rlpyt)](https://github.com/astooke/rlpyt/stargazers) | 11                       | :x:                         | :heavy_check_mark:                | :heavy_check_mark: | :heavy_check_mark: | PyTorch    |
| [ChainerRL](https://github.com/chainer/chainerrl)            | [![GitHub stars](https://img.shields.io/github/stars/chainer/chainerrl)](https://github.com/chainer/chainerrl/stargazers) | 18                       | :heavy_check_mark: (gym)    | :heavy_check_mark:                | :heavy_check_mark: | :x:                | Chainer    |
| [Sample Factory](https://github.com/alex-petrenko/sample-factory) | [![GitHub stars](https://img.shields.io/github/stars/alex-petrenko/sample-factory)](https://github.com/alex-petrenko/sample-factory/stargazers) | 1<sup> (4)</sup>         | :heavy_check_mark: (gym)    | :heavy_check_mark:                | :heavy_check_mark: | :heavy_check_mark: | PyTorch    |
|                                                              |                                                              |                          |                             |                                   |                    |                    |            |
| [Tianshou](https://github.com/thu-ml/tianshou)               | [![GitHub stars](https://img.shields.io/github/stars/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/stargazers) | 20                       | :heavy_check_mark: (gym)    | :heavy_check_mark:                | :heavy_check_mark: | :heavy_check_mark: | PyTorch    |

<sup>(1): access date: 2021-08-08</sup>

<sup>(2): not all algorithms support this feature</sup>

<sup>(3): TQC and QR-DQN in [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) instead of main repo</sup>

<sup>(4): super fast APPO!</sup>


### High quality software engineering standard

| RL Platform                                                  | Documentation                                                | Code Coverage                                                | Type Hints         | Last Update                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------ | ------------------------------------------------------------ |
| [Baselines](https://github.com/openai/baselines)             | :x:                                                          | :x:                                                          | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/openai/baselines?label=last%20update) |
| [Stable-Baselines](https://github.com/hill-a/stable-baselines) | [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines.readthedocs.io/en/master/?badge=master) | [![coverage](https://img.shields.io/badge/coverage-86%25-brightgreen.svg?style=flat)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&utm_medium=referral&utm_content=hill-a/stable-baselines&utm_campaign=Badge_Coverage) <!-- https://github.com/thu-ml/tianshou/issues/249#issuecomment-895882193 --> | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/hill-a/stable-baselines?label=last%20update) |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines3.readthedocs.io/en/master/?badge=master) | [![coverage report](https://gitlab.com/araffin/stable-baselines3/badges/master/coverage.svg)](https://gitlab.com/araffin/stable-baselines3/-/commits/master) | :heavy_check_mark: | ![GitHub last commit](https://img.shields.io/github/last-commit/DLR-RM/stable-baselines3?label=last%20update) |
| [Ray/RLlib](https://github.com/ray-project/ray/tree/master/rllib/) | [![](https://readthedocs.org/projects/ray/badge/?version=master)](http://docs.ray.io/en/master/rllib.html) | :heavy_minus_sign:<sup>(1)</sup>                             | :heavy_check_mark: | ![GitHub last commit](https://img.shields.io/github/last-commit/ray-project/ray?label=last%20update) |
| [SpinningUp](https://github.com/openai/spinningup)           | [![](https://img.shields.io/readthedocs/spinningup)](https://spinningup.openai.com/) | :x:                                                          | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/openai/spinningup?label=last%20update) |
| [Dopamine](https://github.com/google/dopamine)               | [![](https://img.shields.io/badge/docs-passing-green)](https://github.com/google/dopamine/tree/master/docs) | :x:                                                          | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/google/dopamine?label=last%20update) |
| [ACME](https://github.com/deepmind/acme)                     | [![](https://img.shields.io/badge/docs-passing-green)](https://github.com/deepmind/acme/blob/master/docs/index.md) | :heavy_minus_sign:<sup>(1)</sup>                             | :heavy_check_mark: | ![GitHub last commit](https://img.shields.io/github/last-commit/deepmind/acme?label=last%20update) |
| [keras-rl](https://github.com/keras-rl/keras-rl)             | [![Documentation](https://readthedocs.org/projects/keras-rl/badge/)](http://keras-rl.readthedocs.io/) | :heavy_minus_sign:<sup>(1)</sup>                             | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/keras-rl/keras-rl?label=last%20update) |
| [rlpyt](https://github.com/astooke/rlpyt)                    | [![Docs](https://readthedocs.org/projects/rlpyt/badge/?version=latest&style=flat)](https://rlpyt.readthedocs.io/en/latest/) | [![codecov](https://codecov.io/gh/astooke/rlpyt/graph/badge.svg)](https://codecov.io/gh/astooke/rlpyt) | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/astooke/rlpyt?label=last%20update) |
| [ChainerRL](https://github.com/chainer/chainerrl)            | [![Documentation Status](https://readthedocs.org/projects/chainerrl/badge/?version=latest)](http://chainerrl.readthedocs.io/en/latest/?badge=latest) | [![Coverage Status](https://coveralls.io/repos/github/chainer/chainerrl/badge.svg?branch=master)](https://coveralls.io/github/chainer/chainerrl?branch=master) | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/chainer/chainerrl?label=last%20update) |
| [Sample Factory](https://github.com/alex-petrenko/sample-factory) | [:heavy_minus_sign:](https://arxiv.org/abs/2006.11751)       | [![codecov](https://codecov.io/gh/alex-petrenko/sample-factory/branch/master/graph/badge.svg)](https://codecov.io/gh/alex-petrenko/sample-factory) | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/alex-petrenko/sample-factory?label=last%20update) |
|                                                              |                                                              |                                                              |                    |                                                              |
| [Tianshou](https://github.com/thu-ml/tianshou)               | [![Read the Docs](https://img.shields.io/readthedocs/tianshou)](https://tianshou.readthedocs.io/en/master) | [![codecov](https://img.shields.io/codecov/c/gh/thu-ml/tianshou)](https://codecov.io/gh/thu-ml/tianshou) | :heavy_check_mark: | ![GitHub last commit](https://img.shields.io/github/last-commit/thu-ml/tianshou?label=last%20update) |

<sup>(1): it has continuous integration but the coverage rate is not available</sup>


### Reproducible and High Quality Result

Tianshou has its unit tests. Different from other platforms, **the unit tests include the full agent training procedure for all of the implemented algorithms**. It would be failed once if it could not train an agent to perform well enough on limited epochs on toy scenarios. The unit tests secure the reproducibility of our platform. Check out the [GitHub Actions](https://github.com/thu-ml/tianshou/actions) page for more detail.

The Atari/Mujoco benchmark results are under [examples/atari/](examples/atari/) and [examples/mujoco/](examples/mujoco/) folders. **Our Mujoco result can beat most of existing benchmark.**

### Modularized Policy

We decouple all of the algorithms roughly into the following parts:

- `__init__`: initialize the policy;
- `forward`: to compute actions over given observations;
- `process_fn`: to preprocess data from replay buffer (since we have reformulated all algorithms to replay-buffer based algorithms);
- `learn`: to learn from a given batch data;
- `post_process_fn`: to update the replay buffer from the learning process (e.g., prioritized replay buffer needs to update the weight);
- `update`: the main interface for training, i.e., `process_fn -> learn -> post_process_fn`.

Within this API, we can interact with different policies conveniently.

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
logger = ts.utils.TensorboardLogger(SummaryWriter('log/dqn'))  # TensorBoard is supported!
# For other loggers: https://tianshou.readthedocs.io/en/master/tutorials/logger.html
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
# https://tianshou.readthedocs.io/en/master/tutorials/dqn.html#build-the-network
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

It's worth a try: here is a test on a laptop (i7-8750H + GTX1060). It only uses **3** seconds for training an agent based on vanilla policy gradient on the CartPole-v0 task: (seed may be different across different platform and device)

```bash
$ python3 test/discrete/test_pg.py --seed 0 --render 0.03
```

<div align="center">
  <img src="https://github.com/thu-ml/tianshou/raw/master/docs/_static/images/testpg.gif"></a>
</div>

## Contributing

Tianshou is still under development. More algorithms and features are going to be added and we always welcome contributions to help make Tianshou better. If you would like to contribute, please check out [this link](https://tianshou.readthedocs.io/en/master/contributing.html).

## Citing Tianshou

If you find Tianshou useful, please cite it in your publications.

```latex
@article{tianshou,
  title={Tianshou: A Highly Modularized Deep Reinforcement Learning Library},
  author={Weng, Jiayi and Chen, Huayu and Yan, Dong and You, Kaichao and Duburcq, Alexis and Zhang, Minghao and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2107.14171},
  year={2021}
}
```

## Acknowledgment

Tianshou was previously a reinforcement learning platform based on TensorFlow. You can check out the branch [`priv`](https://github.com/thu-ml/tianshou/tree/priv) for more detail. Many thanks to [Haosheng Zou](https://github.com/HaoshengZou)'s pioneering work for Tianshou before version 0.1.1.

We would like to thank [TSAIL](http://ml.cs.tsinghua.edu.cn/) and [Institute for Artificial Intelligence, Tsinghua University](http://ml.cs.tsinghua.edu.cn/thuai/) for providing such an excellent AI research platform.
