Deep Q Network
==============

Deep reinforcement learning has achieved significant successes in various applications.
**Deep Q Network** (DQN) :cite:`DQN` is the pioneer one.
In this tutorial, we will show how to train a DQN agent on CartPole with Tianshou step by step.
The full script is at `test/discrete/test_dqn.py <https://github.com/thu-ml/tianshou/blob/master/test/discrete/test_dqn.py>`_.

Contrary to existing Deep RL libraries such as `RLlib <https://github.com/ray-project/ray/tree/master/rllib/>`_, which could only accept a config specification of hyperparameters, network, and others, Tianshou provides an easy way of construction through the code-level.


Make an Environment
-------------------

First of all, you have to make an environment for your agent to interact with. For environment interfaces, we follow the convention of `OpenAI Gym <https://github.com/openai/gym>`_. In your Python code, simply import Tianshou and make the environment:
::

    import gym
    import tianshou as ts

    env = gym.make('CartPole-v0')

CartPole-v0 is a simple environment with a discrete action space, for which DQN applies. You have to identify whether the action space is continuous or discrete and apply eligible algorithms. DDPG :cite:`DDPG`, for example, could only be applied to continuous action spaces, while almost all other policy gradient methods could be applied to both, depending on the probability distribution on the action.


Setup Multi-environment Wrapper
-------------------------------

It is available if you want the original ``gym.Env``: 
::

    train_envs = gym.make('CartPole-v0')
    test_envs = gym.make('CartPole-v0')

Tianshou supports parallel sampling for all algorithms. It provides three types of vectorized environment wrapper: :class:`~tianshou.env.VectorEnv`, :class:`~tianshou.env.SubprocVectorEnv`, and :class:`~tianshou.env.RayVectorEnv`. It can be used as follows: 
::

    train_envs = ts.env.VectorEnv([lambda: gym.make('CartPole-v0') for _ in range(8)])
    test_envs = ts.env.VectorEnv([lambda: gym.make('CartPole-v0') for _ in range(100)])

Here, we set up 8 environments in ``train_envs`` and 100 environments in ``test_envs``.

For the demonstration, here we use the second block of codes.


Build the Network
-----------------

Tianshou supports any user-defined PyTorch networks and optimizers but with the limitation of input and output API. Here is an example code: 
::

    import torch, numpy as np
    from torch import nn

    class Net(nn.Module):
        def __init__(self, state_shape, action_shape):
            super().__init__()
            self.model = nn.Sequential(*[
                nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, np.prod(action_shape))
            ])
        def forward(self, obs, state=None, info={}):
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float)
            batch = obs.shape[0]
            logits = self.model(obs.view(batch, -1))
            return logits, state

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

The rules of self-defined networks are:

1. Input: observation ``obs`` (may be a ``numpy.ndarray`` or ``torch.Tensor``), hidden state ``state`` (for RNN usage), and other information ``info`` provided by the environment.
2. Output: some ``logits`` and the next hidden state ``state``. The logits could be a tuple instead of a ``torch.Tensor``. It depends on how the policy process the network output. For example, in PPO :cite:`PPO`, the return of the network might be ``(mu, sigma), state`` for Gaussian policy.


Setup Policy
------------

We use the defined ``net`` and ``optim``, with extra policy hyper-parameters, to define a policy. Here we define a DQN policy with using a target network: 
::

    policy = ts.policy.DQNPolicy(net, optim,
        discount_factor=0.9, estimation_step=3,
        use_target_network=True, target_update_freq=320)


Setup Collector
---------------

The collector is a key concept in Tianshou. It allows the policy to interact with different types of environments conveniently. 
In each step, the collector will let the policy perform (at least) a specified number of steps or episodes and store the data in a replay buffer.
::

    train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(size=20000))
    test_collector = ts.data.Collector(policy, test_envs)


Train Policy with a Trainer
---------------------------

Tianshou provides :class:`~tianshou.trainer.onpolicy_trainer` and :class:`~tianshou.trainer.offpolicy_trainer`. The trainer will automatically stop training when the policy reach the stop condition ``stop_fn`` on test collector. Since DQN is an off-policy algorithm, we use the :class:`~tianshou.trainer.offpolicy_trainer` as follows:
::

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10, step_per_epoch=1000, collect_per_step=10,
        episode_per_test=100, batch_size=64,
        train_fn=lambda e: policy.set_eps(0.1),
        test_fn=lambda e: policy.set_eps(0.05),
        stop_fn=lambda x: x >= env.spec.reward_threshold,
        writer=None)
    print(f'Finished training! Use {result["duration"]}')

The meaning of each parameter is as follows:

* ``max_epoch``: The maximum of epochs for training. The training process might be finished before reaching the ``max_epoch``;
* ``step_per_epoch``: The number of step for updating policy network in one epoch;
* ``collect_per_step``: The number of frames the collector would collect before the network update. For example, the code above means "collect 10 frames and do one policy network update";
* ``episode_per_test``: The number of episodes for one policy evaluation.
* ``batch_size``: The batch size of sample data, which is going to feed in the policy network.
* ``train_fn``: A function receives the current number of epoch index and performs some operations at the beginning of training in this epoch. For example, the code above means "reset the epsilon to 0.1 in DQN before training".
* ``test_fn``: A function receives the current number of epoch index and performs some operations at the beginning of testing in this epoch. For example, the code above means "reset the epsilon to 0.05 in DQN before testing".
* ``stop_fn``: A function receives the average undiscounted returns of the testing result, return a boolean which indicates whether reaching the goal.
* ``writer``: See below.

The trainer supports `TensorBoard <https://www.tensorflow.org/tensorboard>`_ for logging. It can be used as:
::

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('log/dqn')

Pass the writer into the trainer, and the training result will be recorded into the TensorBoard.

The returned result is a dictionary as follows:
::

    {
        'train_step': 9246,
        'train_episode': 504.0,
        'train_time/collector': '0.65s',
        'train_time/model': '1.97s',
        'train_speed': '3518.79 step/s',
        'test_step': 49112,
        'test_episode': 400.0,
        'test_time': '1.38s',
        'test_speed': '35600.52 step/s',
        'best_reward': 199.03,
        'duration': '4.01s'
    }

It shows that within approximately 4 seconds, we finished training a DQN agent on CartPole. The mean returns over 100 consecutive episodes is 199.03.


Save/Load Policy
----------------

Since the policy inherits the ``torch.nn.Module`` class, saving and loading the policy are exactly the same as a torch module:
::

    torch.save(policy.state_dict(), 'dqn.pth')
    policy.load_state_dict(torch.load('dqn.pth'))


Watch the Agent's Performance
-----------------------------

:class:`~tianshou.data.Collector` supports rendering. Here is the example of watching the agent's performance in 35 FPS:
::

    collector = ts.data.Collector(policy, env)
    collector.collect(n_episode=1, render=1 / 35)
    collector.close()


.. _customized_trainer:

Train a Policy with Customized Codes
------------------------------------

"I don't want to use your provided trainer. I want to customize it!"

No problem! Tianshou supports user-defined training code. Here is the usage:
::

    # pre-collect 5000 frames with random action before training
    policy.set_eps(1)
    train_collector.collect(n_step=5000)

    policy.set_eps(0.1)
    for i in range(int(1e6)):  # total step
        collect_result = train_collector.collect(n_step=10)

        # once if the collected episodes' mean returns reach the threshold,
        # or every 1000 steps, we test it on test_collector
        if collect_result['rew'] >= env.spec.reward_threshold or i % 1000 == 0:
            policy.set_eps(0.05)
            result = test_collector.collect(n_episode=100)
            if result['rew'] >= env.spec.reward_threshold:
                print(f'Finished training! Test mean returns: {result["rew"]}')
                break
            else:
                # back to training eps
                policy.set_eps(0.1)

        # train policy with a sampled batch data
        losses = policy.learn(train_collector.sample(batch_size=64))

For further usage, you can refer to :doc:`/tutorials/cheatsheet`.

.. rubric:: References

.. bibliography:: /refs.bib
    :style: unsrtalpha
