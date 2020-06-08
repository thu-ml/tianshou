Cheat Sheet
===========

This page shows some code snippets of how to use Tianshou to develop new algorithms / apply algorithms to new scenarios.

By the way, some of these issues can be resolved by using a ``gym.wrapper``. It can be a solution in the policy-environment interaction.

.. _network_api:

Build Policy Network
--------------------

See :ref:`build_the_network`.

.. _new_policy:

Build New Policy
----------------

See :class:`~tianshou.policy.BasePolicy`.

.. _customize_training:

Customize Training Process
--------------------------

See :ref:`customized_trainer`.

.. _parallel_sampling:

Parallel Sampling
-----------------

Use :class:`~tianshou.env.VectorEnv` or :class:`~tianshou.env.SubprocVectorEnv`.
::

    env_fns = [
        lambda: MyTestEnv(size=2),
        lambda: MyTestEnv(size=3),
        lambda: MyTestEnv(size=4),
        lambda: MyTestEnv(size=5),
    ]
    venv = SubprocVectorEnv(env_fns)

where ``env_fns`` is a list of callable env hooker. The above code can be written in for-loop as well:
::

    env_fns = [lambda x=i: MyTestEnv(size=x) for i in [2, 3, 4, 5]]
    venv = SubprocVectorEnv(env_fns)

.. _rnn_training:

RNN-style Training
------------------

This is related to `Issue 19 <https://github.com/thu-ml/tianshou/issues/19>`_.

First, add an argument ``stack_num`` to :class:`~tianshou.data.ReplayBuffer`:
::

    buf = ReplayBuffer(size=size, stack_num=stack_num)

Then, change the network to recurrent-style, for example, class ``Recurrent`` in `code snippet 1 <https://github.com/thu-ml/tianshou/blob/master/test/discrete/net.py>`_, or ``RecurrentActor`` and ``RecurrentCritic`` in `code snippet 2 <https://github.com/thu-ml/tianshou/blob/master/test/continuous/net.py>`_.

The above code supports only stacked-observation. If you want to use stacked-action (for Q(stacked-s, stacked-a)), stacked-reward, or other stacked variables, you can add a ``gym.wrapper`` to modify the state representation. For example, if we add a wrapper that map [s, a] pair to a new state:

- Before: (s, a, s', r, d) stored in replay buffer, and get stacked s;
- After applying wrapper: ([s, a], a, [s', a'], r, d) stored in replay buffer, and get both stacked s and a.

.. _self_defined_env:

User-defined Environment and Different State Representation
-----------------------------------------------------------

This is related to `Issue 38 <https://github.com/thu-ml/tianshou/issues/38>`_ and `Issue 69 <https://github.com/thu-ml/tianshou/issues/69>`_.

First of all, your self-defined environment must follow the Gym's API, some of them are listed below:

- reset() -> state

- step(action) -> state, reward, done, info

- seed(s) -> None

- render(mode) -> None

- close() -> None

The state can be a ``numpy.ndarray`` or a Python dictionary. Take ``FetchReach-v1`` as an example:
::

    >>> e = gym.make('FetchReach-v1')
    >>> e.reset()
    {'observation': array([ 1.34183265e+00,  7.49100387e-01,  5.34722720e-01,  1.97805133e-04,
             7.15193042e-05,  7.73933014e-06,  5.51992816e-08, -2.42927453e-06,
             4.73325650e-06, -2.28455228e-06]),
     'achieved_goal': array([1.34183265, 0.74910039, 0.53472272]),
     'desired_goal': array([1.24073906, 0.77753463, 0.63457791])}

It shows that the state is a dictionary which has 3 keys. It will stored in :class:`~tianshou.data.ReplayBuffer` as:
::

    >>> from tianshou.data import ReplayBuffer
    >>> b = ReplayBuffer(size=3)
    >>> b.add(obs=e.reset(), act=0, rew=0, done=0)
    >>> print(b)
    ReplayBuffer(
        act: array([0, 0, 0]),
        done: array([0, 0, 0]),
        info: Batch(),
        obs: Batch(
                 achieved_goal: array([[1.34183265, 0.74910039, 0.53472272],
                                       [0.        , 0.        , 0.        ],
                                       [0.        , 0.        , 0.        ]]),
                 desired_goal: array([[1.42154265, 0.62505137, 0.62929863],
                                      [0.        , 0.        , 0.        ],
                                      [0.        , 0.        , 0.        ]]),
                 observation: array([[ 1.34183265e+00,  7.49100387e-01,  5.34722720e-01,
                                       1.97805133e-04,  7.15193042e-05,  7.73933014e-06,
                                       5.51992816e-08, -2.42927453e-06,  4.73325650e-06,
                                      -2.28455228e-06],
                                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                       0.00000000e+00],
                                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                       0.00000000e+00]]),
             ),
        policy: Batch(),
        rew: array([0, 0, 0]),
    )
    >>> print(b.obs.achieved_goal)
    [[1.34183265 0.74910039 0.53472272]
     [0.         0.         0.        ]
     [0.         0.         0.        ]]

And the data batch sampled from this replay buffer:
::

    >>> batch, indice = b.sample(2)
    >>> batch.keys()
    ['act', 'done', 'info', 'obs', 'obs_next', 'policy', 'rew']
    >>> batch.obs[-1]
    Batch(
        achieved_goal: array([1.34183265, 0.74910039, 0.53472272]),
        desired_goal: array([1.42154265, 0.62505137, 0.62929863]),
        observation: array([ 1.34183265e+00,  7.49100387e-01,  5.34722720e-01,  1.97805133e-04,
                             7.15193042e-05,  7.73933014e-06,  5.51992816e-08, -2.42927453e-06,
                             4.73325650e-06, -2.28455228e-06]),
    )
    >>> batch.obs.desired_goal[-1]  # recommended
    array([1.42154265, 0.62505137, 0.62929863])
    >>> batch.obs[-1].desired_goal  # not recommended
    array([1.42154265, 0.62505137, 0.62929863])
    >>> batch[-1].obs.desired_goal  # not recommended
    array([1.42154265, 0.62505137, 0.62929863])

Thus, in your self-defined network, just change the ``forward`` function as:
::

    def forward(self, s, ...):
        # s is a batch
        observation = s.observation
        achieved_goal = s.achieved_goal
        desired_goal = s.desired_goal
        ...

For self-defined class, the replay buffer will store the reference into a ``numpy.ndarray``, e.g.:
::

    >>> import networkx as nx
    >>> b = ReplayBuffer(size=3)
    >>> b.add(obs=nx.Graph(), act=0, rew=0, done=0)
    >>> print(b)
    ReplayBuffer(
        act: array([0, 0, 0]),
        done: array([0, 0, 0]),
        info: Batch(),
        obs: array([<networkx.classes.graph.Graph object at 0x7f5c607826a0>, None,
                    None], dtype=object),
        policy: Batch(),
        rew: array([0, 0, 0]),
    )

But the state stored in the buffer may be a shallow-copy. To make sure each of your state stored in the buffer is distinct, please return the deep-copy version of your state in your env:
::

    def reset():
        return copy.deepcopy(self.graph)
    def step(a):
        ...
        return copy.deepcopy(self.graph), reward, done, {}
