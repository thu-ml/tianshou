Multi-Agent RL
==============

In this section, we describe how to use Tianshou to implement multi-agent reinforcement learning. Specifically, we will design an algorithm to learn how to play `Tic Tac Toe <https://en.wikipedia.org/wiki/Tic-tac-toe>`_ (see the image below) against a random opponent.

.. image:: ../_static/images/tic-tac-toe.png
    :align: center

Tic-Tac-Toe Environment
-----------------------

The scripts are located at ``test/multiagent/``. We have implemented a Tic-Tac-Toe environment that supports Tic-Tac-Toe of any scale. Let's first explore the environment. The 3x3 Tic-Tac-Toe is too easy, so we will focus on 6x6 Tic-Tac-Toe where 4 same signs in a row are considered to win.
::

    >>> from tic_tac_toe_env import TicTacToeEnv    # the module tic_tac_toe_env is in test/multiagent/
    >>> board_size = 6                              # the size of board size
    >>> win_size = 4                                # how many signs in a row are considered to win
    >>> 
    >>> # This board has 6 rows and 6 cols (36 places in total)
    >>> # Players place 'x' and 'o' in turn on the board
    >>> # The player who first gets 4 consecutive 'x's or 'o's wins
    >>> 
    >>> env = TicTacToeEnv(size=board_size, win_size=win_size)
    >>> obs = env.reset()
    >>> env.render()                                # render the empty board
    board:
    =================
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    =================
    >>> print(obs)                                  # let's see the shape of the observation
    {'agent_id': 1,
     'obs': array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]], dtype=int32),
     'legal_actions': {0, 1, 2, ..., 34, 35}}

The observation variable ``obs`` returned from the environment is a ``dict``, with three keys ``agent_id``, ``obs``, ``legal_actions``. This is a general structure in multi-agent RL where agents take turns. The meaning of these keys are:

- ``agent_id``: the id of the current acting agent, where agent_id :math:`\in [1, N]`, N is the number of agents. In our Tic-Tac-Toe case, N is 2. The agent_id starts from 1 because we reserve 0 for the environment itself. Sometimes the developer may want to control the behavior of the environment, for example, to determine how to dispatch cards in Poker.

- ``obs``: the actual observation of the environment. In the Tic-Tac-Toe game above, the observation variable ``obs`` is a ``np.ndarray`` with the shape of (6, 6). The values can be "0/1/-1": 0 for empty, 1 for ``x``, -1 for ``o``. Agent 1 places ``x`` on the board, while agent 2 places ``o`` on the board.

- ``legal_actions``: the legal actions in the current timestep. In board games or card games, legal actions vary with time. The legal_actions is a set of action indices. For Tic-Tac-Toe, index ``i`` means the place of ``i/N`` th row and ``i%N`` th column is empty and the player can place an ``x`` or ``o`` at that position. Now the board is empty, so the legal_actions contains all the positions on the board.

Let's play two steps to have an intuitive understanding of the environment.

::

    >>> import numpy as np
    >>> action = np.array([0])                      # action is an np.ndarray with one element
    >>> obs, reward, done, info = env.step(action)  # the env.step follows the api of OpenAI Gym
    >>> print(obs)                                  # notice the change in the observation
    {'agent_id': 2,
     'obs': array([[1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0]], dtype=int32),
     'legal_actions': {1, 2, 3, ..., 34, 35}}
    >>> # reward has two items, one for each player: 1 for win, -1 for lose, and 0 otherwise
    >>> print(reward)
    [0. 0.]
    >>> print(done)                                 # done indicates whether the game is over
    False
    >>> # info is always an empty dict in Tic-Tac-Toe, but may contain some useful information in environments other than Tic-Tac-Toe.
    >>> print(info)
    {}

One worth-noting case is that the game is over when there is only one empty position, rather than when there is no position. This is because the player just has one choice (literally no choice) in this game.
::

    >>> # omitted actions: 6, 1, 7, 2, 8
    >>> action = np.array([3])
    >>> obs, reward, done, info = env.step(action)  # player 1 wins
    >>> print((reward, done))
    (array([ 1., -1.], dtype=float32), array(True))
    >>> env.render()                                # 'X' and 'O' indicate the last action
    board:
    =================
    ===x x x X _ _===
    ===o o o _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    =================

After being familiar with the environment, let's try to play with random agents first!

Multi-Agent RL Construction
---------------------------

Tianshou already provides some builtin classes for multi-agent learning. You can checkout the API documentation for details. Here we will use :class:`~tianshou.policy.RandomMultiAgentPolicy` and :class:`~tianshou.policy.MultiAgentPolicyManager`.

::

    >>> from tianshou.policy import RandomMultiAgentPolicy, MultiAgentPolicyManager
    >>> from tianshou.data import Collector
    >>>
    >>> # agents should be wrapped into one policy, which is responsible for calling the acting agent correctly
    >>> # here we use two random agents
    >>> policy = MultiAgentPolicyManager([RandomMultiAgentPolicy(), RandomMultiAgentPolicy()])
    >>>
    >>> # use collectors to collect a episode of trajectories
    >>> collector = Collector(policy, env, reward_metric=lambda x: x[0])
    >>> # you will see a long trajectory showing the board status at each timestep
    >>> result = collector.collect(n_episode=1, render=0.1)  # (only show the last three steps)
    ...
    board:
    =================
    ===_ _ x o _ _===
    ===x o o _ x x===
    ===o _ x o x _===
    ===o x O _ _ x===
    ===_ o o _ _ x===
    ===_ _ o _ _ x===
    =================
    board:
    =================
    ===_ X x o _ _===
    ===x o o _ x x===
    ===o _ x o x _===
    ===o x o _ _ x===
    ===_ o o _ _ x===
    ===_ _ o _ _ x===
    =================
    board:
    =================
    ===_ x x o _ _===
    ===x o o _ x x===
    ===o _ x o x _===
    ===o x o _ _ x===
    ===_ o o _ _ x===
    ===O _ o _ _ x===
    =================
    >>> collector.close()

Random agents perform badly. In the above game, although agent 2 wins at last, it is clear that a smart agent 1 would place an ``x`` at row 2 col 5 to win directly. So, let's start to train our Tic-Tac-Toe agent!

First, import some required modules.
::

    import os
    import torch
    import argparse
    import numpy as np
    from copy import deepcopy
    from torch.utils.tensorboard import SummaryWriter

    from tianshou.env import VectorEnv
    from tianshou.policy import (MultiAgentDQNPolicy,
                                 MultiAgentPolicyManager,
                                 RandomMultiAgentPolicy,
                                 BaseMultiAgentPolicy)
    from tianshou.utils.net.common import Net
    from tianshou.data import Collector, ReplayBuffer
    from tianshou.trainer import offpolicy_trainer

    from tic_tac_toe_env import TicTacToeEnv

The explanation of each Tianshou class/function will be deferred to their first usages.

Here we define some arguments and hyperparameters of the experiment. The meaning of arguments is clear by just looking at their names.
::

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=1626)
        parser.add_argument('--eps-test', type=float, default=0.05)
        parser.add_argument('--eps-train', type=float, default=0.1)
        parser.add_argument('--buffer-size', type=int, default=20000)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--gamma', type=float, default=0.1,
                            help='a smaller gamma favors earlier win')
        parser.add_argument('--n-step', type=int, default=3)
        parser.add_argument('--target-update-freq', type=int, default=320)
        parser.add_argument('--epoch', type=int, default=5)
        parser.add_argument('--step-per-epoch', type=int, default=1000)
        parser.add_argument('--collect-per-step', type=int, default=10)
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--layer-num', type=int, default=3)
        parser.add_argument('--training-num', type=int, default=8)
        parser.add_argument('--test-num', type=int, default=100)
        parser.add_argument('--logdir', type=str, default='log')
        parser.add_argument('--render', type=float, default=0.1)
        parser.add_argument('--board_size', type=int, default=6)
        parser.add_argument('--win_size', type=int, default=4)
        parser.add_argument('--watch', default=False, action='store_true',
                            help='no training, '
                                 'watch the play of pre-trained models')
        parser.add_argument('--agent_id', type=int, default=2,
                            help='the learned agent plays as the'
                                 ' agent_id-th player. choices are 1 and 2.')
        parser.add_argument('--resume_path', type=str, default='',
                            help='the path of agent pth file '
                                 'for resuming from a pre-trained agent')
        parser.add_argument('--opponent_path', type=str, default='',
                            help='the path of opponent agent pth file '
                                 'for resuming from a pre-trained agent')
        parser.add_argument(
            '--device', type=str,
            default='cuda' if torch.cuda.is_available() else 'cpu')
        args = parser.parse_known_args()[0]
        return args

The following ``get_agents`` function returns agents and their optimizers from either constructing a new policy, or loading from disk, or using the pass-in arguments. The action model we use is an instance of :class:`~tianshou.utils.net.common.Net`, essentially a multi-layer perceptron with the ReLU activation function. The network model is passed to a :class:`~tianshou.policy.MultiAgentDQNPolicy`, the multi-agent version of DQN (actions are selected according to legal actions and their Q-values). For the opponent, it can be either a random agent :class:`~tianshou.policy.RandomMultiAgentPolicy` that randomly chooses an action from legal actions, or it can be a pre-trained :class:`~tianshou.policy.MultiAgentDQNPolicy` to allow learned agents play with themselves. Both agents are passed to :class:`~tianshou.policy.MultiAgentPolicyManager`, which is responsible to call the correct agent according to the ``agent_id`` in the observation. :class:`~tianshou.policy.MultiAgentPolicyManager` also dispatches data to each agent according to ``agent_id``, so that each agent seems to play with a virtual single-agent environment.
::

    def get_agents(args=get_args(),
                   agent_learn=None,     # BaseMultiAgentPolicy
                   agent_opponent=None,  # BaseMultiAgentPolicy
                   optim=None,           # torch.optim.Optimizer
                   ):  # return a tuple of (BaseMultiAgentPolicy, torch.optim.Optimizer)
        env_func = lambda: TicTacToeEnv(args.board_size, args.win_size)
        env = env_func()
        args.state_shape = env.observation_space.shape or env.observation_space.n
        args.action_shape = env.action_space.shape or env.action_space.n
        if agent_learn is None:
            # model
            net = Net(args.layer_num, args.state_shape, args.action_shape, args.device).to(args.device)
            if optim is None:
                optim = torch.optim.Adam(net.parameters(), lr=args.lr)
            agent_learn = MultiAgentDQNPolicy(
                net, optim, args.gamma, args.n_step,
                target_update_freq=args.target_update_freq)
            if args.resume_path:
                agent_learn.load_state_dict(torch.load(args.resume_path))

        if agent_opponent is None:
            if args.opponent_path:
                agent_opponent = deepcopy(agent_learn)
                agent_opponent.load_state_dict(torch.load(args.opponent_path))
            else:
                agent_opponent = RandomMultiAgentPolicy()

        if args.agent_id == 1:
            agents = [agent_learn, agent_opponent]
        else:
            agents = [agent_opponent, agent_learn]
        policy = MultiAgentPolicyManager(agents)
        return policy, optim

With the above preparation, we are close to get the first learned agent. The following code is almost the same as the code of the DQN tutorial.

::

    args = get_args() # parse arguments
    # the reward is a vector, we need a scalar metric to monitor the training.
    # we choose the reward of the learning agent
    Collector._default_rew_metric = lambda x: x[args.agent_id - 1]

    # ======== a test function that tests a pre-trained agent and exit ======
    def watch(
            args: argparse.Namespace = get_args(),
            agent_learn: Optional[BaseMultiAgentPolicy] = None,
            agent_opponent: Optional[BaseMultiAgentPolicy] = None,
            ) -> None:
        def env_func():
            return TicTacToeEnv(args.board_size, args.win_size)
        env = env_func()
        policy, optim = get_agents(
            args, agent_learn=agent_learn, agent_opponent=agent_opponent)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()
    if args.watch:
        watch(args)
        exit(0)

    # ======== environment setup =========
    def env_func():
        return TicTacToeEnv(args.board_size, args.win_size)
    train_envs = VectorEnv([env_func for _ in range(args.training_num)])
    test_envs = VectorEnv([env_func for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # ======== agent setup =========
    policy, optim = get_agents()

    # ======== collector setup =========
    train_collector = Collector(
                        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size)

    # ======== tensorboard logging setup =========
    if not hasattr(args, 'writer'):
        log_path = os.path.join(args.logdir, 'tic_tac_toe', 'dqn')
        writer = SummaryWriter(log_path)
    else:
        writer = args.writer

    # ======== callback functions used during training =========

    def save_fn(policy):
        if hasattr(args, 'model_save_path'):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, 'tic_tac_toe', 'dqn', 'policy.pth')
        torch.save(
            policy.policies[args.agent_id - 1].state_dict(),
            model_save_path)

    def stop_fn(x):
        return x >= 0.9

    def train_fn(x):
        policy.policies[args.agent_id - 1].set_eps(args.eps_train)

    def test_fn(x):
        policy.policies[args.agent_id - 1].set_eps(args.eps_test)

    # start training, this may require about three minutes
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, save_fn=save_fn, writer=writer)

    train_collector.close()
    test_collector.close()

    agent = policy.policies[args.agent_id - 1]
    # let's watch the match!
    watch(args, agent)

That's it. By executing the code, you will see a progress bar indicating the progress of training. After about three minutes, the agent has finished training, and you can see how it plays against the random agent. Here is an example:

::

    board:
    =================
    ===_ _ X _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    =================
    board:
    =================
    ===_ _ x _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ O _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    =================
    board:
    =================
    ===_ _ x _ _ X===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ o _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    =================
    board:
    =================
    ===_ _ x _ _ x===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ o O===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    =================
    board:
    =================
    ===_ _ x _ _ x===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ X===
    ===_ _ _ _ o o===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    =================
    board:
    =================
    ===_ _ x _ _ x===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ x===
    ===_ _ O _ o o===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    =================
    board:
    =================
    ===_ _ x _ _ x===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ x===
    ===X _ o _ o o===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    =================
    board:
    =================
    ===_ _ x _ _ x===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ x===
    ===x _ o O o o===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    =================
    Final reward: 1.0, length: 8.0

Notice that, our learned agent plays the role of agent 2, placing ``o`` on the board. The agent performs pretty well against the random opponent! It learns the rule of the game by trial and error, and learns that four consecutive ``o`` means winning, so it does!

The above code can be executed in a python shell or can be saved as a script file (we have saved it in ``test/multiagent/test_tic_tac_toe.py``). In the latter case, you can train an agent by

.. code-block:: console

    $ python test_tic_tac_toe.py

By default, the trained agent is stored in ``log/tic_tac_toe/dqn/policy.pth``. You can also make the trained agent play against itself, by

.. code-block:: console

    $ python test_tic_tac_toe.py --watch --resume_path=log/tic_tac_toe/dqn/policy.pth --opponent_path=log/tic_tac_toe/dqn/policy.pth

Here is my output:

::

    board:
    =================
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ X _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    =================
    board:
    =================
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ x O===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    =================
    board:
    =================
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ X _ x o===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    =================
    board:
    =================
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ x _ x o===
    ===_ _ _ _ _ _===
    ===_ _ _ _ O _===
    =================
    board:
    =================
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ x X x o===
    ===_ _ _ _ _ _===
    ===_ _ _ _ o _===
    =================
    board:
    =================
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ O _ _ _===
    ===_ _ x x x o===
    ===_ _ _ _ _ _===
    ===_ _ _ _ o _===
    =================
    board:
    =================
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ o _ _ _===
    ===_ _ x x x o===
    ===_ _ _ _ _ _===
    ===_ _ _ _ o X===
    =================
    board:
    =================
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ o _ _ _===
    ===_ _ x x x o===
    ===_ _ O _ _ _===
    ===_ _ _ _ o x===
    =================
    board:
    =================
    ===_ _ _ _ _ _===
    ===_ _ _ _ _ _===
    ===_ _ o _ X _===
    ===_ _ x x x o===
    ===_ _ o _ _ _===
    ===_ _ _ _ o x===
    =================
    board:
    =================
    ===_ _ _ _ _ _===
    ===O _ _ _ _ _===
    ===_ _ o _ x _===
    ===_ _ x x x o===
    ===_ _ o _ _ _===
    ===_ _ _ _ o x===
    =================
    board:
    =================
    ===_ _ _ _ _ _===
    ===o _ _ _ _ _===
    ===_ _ o _ x _===
    ===_ _ x x x o===
    ===_ _ o _ X _===
    ===_ _ _ _ o x===
    =================
    board:
    =================
    ===_ _ _ _ _ _===
    ===o _ _ _ _ _===
    ===_ O o _ x _===
    ===_ _ x x x o===
    ===_ _ o _ x _===
    ===_ _ _ _ o x===
    =================
    board:
    =================
    ===_ _ _ _ _ _===
    ===o _ _ _ _ _===
    ===_ o o X x _===
    ===_ _ x x x o===
    ===_ _ o _ x _===
    ===_ _ _ _ o x===
    =================
    board:
    =================
    ===_ _ _ _ _ _===
    ===o _ _ _ _ _===
    ===O o o x x _===
    ===_ _ x x x o===
    ===_ _ o _ x _===
    ===_ _ _ _ o x===
    =================
    board:
    =================
    ===_ _ _ _ _ _===
    ===o _ _ _ _ _===
    ===o o o x x _===
    ===_ _ x x x o===
    ===_ _ o X x _===
    ===_ _ _ _ o x===
    =================
    board:
    =================
    ===_ _ _ _ O _===
    ===o _ _ _ _ _===
    ===o o o x x _===
    ===_ _ x x x o===
    ===_ _ o x x _===
    ===_ _ _ _ o x===
    =================
    board:
    =================
    ===_ _ _ _ o _===
    ===o _ _ _ _ _===
    ===o o o x x _===
    ===X _ x x x o===
    ===_ _ o x x _===
    ===_ _ _ _ o x===
    =================
    board:
    =================
    ===_ _ _ _ o _===
    ===o _ _ _ _ _===
    ===o o o x x _===
    ===x _ x x x o===
    ===_ _ o x x O===
    ===_ _ _ _ o x===
    =================
    board:
    =================
    ===_ _ _ _ o _===
    ===o _ _ _ _ _===
    ===o o o x x _===
    ===x _ x x x o===
    ===_ X o x x o===
    ===_ _ _ _ o x===
    =================
    board:
    =================
    ===_ O _ _ o _===
    ===o _ _ _ _ _===
    ===o o o x x _===
    ===x _ x x x o===
    ===_ x o x x o===
    ===_ _ _ _ o x===
    =================
    board:
    =================
    ===_ o _ _ o _===
    ===o _ _ X _ _===
    ===o o o x x _===
    ===x _ x x x o===
    ===_ x o x x o===
    ===_ _ _ _ o x===
    =================

Well, although the learned agent plays well against the random agent, it is far away from intelligence.

Next, maybe you can try to build more intelligent agents by letting the agent learn from self-play, just like AlphaZero!

In this tutorial, we show an example of how to use Tianshou for multi-agent RL. Tianshou is a flexible and easy to use RL library. Make the best of Tianshou by yourself!
