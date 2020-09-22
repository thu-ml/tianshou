Basic concepts in Tianshou
==========================

Tianshou splits a Reinforcement Learning agent training procedure into these parts: trainer, collector, policy, and data buffer. The general control flow can be described as:

.. image:: /_static/images/concepts_arch.png
    :align: center
    :height: 300


Here is a more detailed description, where ``Env`` is the environment and ``Model`` is the neural network:

.. image:: /_static/images/concepts_arch2.png
    :align: center
    :height: 300


Batch
-----

Tianshou provides :class:`~tianshou.data.Batch` as the internal data structure to pass any kind of data to other methods, for example, a collector gives a :class:`~tianshou.data.Batch` to policy for learning. Let's take a look at this script:
::

    >>> import torch, numpy as np
    >>> from tianshou.data import Batch
    >>> data = Batch(a=4, b=[5, 5], c='2312312', d=('a', -2, -3))
    >>> # the list will automatically be converted to numpy array
    >>> data.b
    array([5, 5])
    >>> data.b = np.array([3, 4, 5])
    >>> print(data)
    Batch(
        a: 4,
        b: array([3, 4, 5]),
        c: '2312312',
        d: array(['a', '-2', '-3'], dtype=object),
    )
    >>> data = Batch(obs={'index': np.zeros((2, 3))}, act=torch.zeros((2, 2)))
    >>> data[:, 1] += 6
    >>> print(data[-1])
    Batch(
        obs: Batch(
                 index: array([0., 6., 0.]),
             ),
        act: tensor([0., 6.]),
    )

In short, you can define a :class:`~tianshou.data.Batch` with any key-value pair, and perform some common operations over it.

:ref:`batch_concept` is a dedicated tutorial for :class:`~tianshou.data.Batch`. We strongly recommend every user to read it so as to correctly understand and use :class:`~tianshou.data.Batch`.


Buffer
------

.. automodule:: tianshou.data.ReplayBuffer
   :members:
   :noindex:

Tianshou provides other type of data buffer such as :class:`~tianshou.data.ListReplayBuffer` (based on list), :class:`~tianshou.data.PrioritizedReplayBuffer` (based on Segment Tree and ``numpy.ndarray``). Check out :class:`~tianshou.data.ReplayBuffer` for more detail.


Policy
------

Tianshou aims to modularizing RL algorithms. It comes into several classes of policies in Tianshou. All of the policy classes must inherit :class:`~tianshou.policy.BasePolicy`.

A policy class typically has the following parts:

* :meth:`~tianshou.policy.BasePolicy.__init__`: initialize the policy, including copying the target network and so on;
* :meth:`~tianshou.policy.BasePolicy.forward`: compute action with given observation;
* :meth:`~tianshou.policy.BasePolicy.process_fn`: pre-process data from the replay buffer;
* :meth:`~tianshou.policy.BasePolicy.learn`: update policy with a given batch of data.
* :meth:`~tianshou.policy.BasePolicy.post_process_fn`: update the buffer with a given batch of data.
* :meth:`~tianshou.policy.BasePolicy.update`: the main interface for training. This function samples data from buffer, pre-process data (such as computing n-step return), learn with the data, and finally post-process the data (such as updating prioritized replay buffer); in short, ``process_fn -> learn -> post_process_fn``.


.. _policy_state:

States for policy
^^^^^^^^^^^^^^^^^

During the training process, the policy has two main states: training state and testing state. The training state can be further divided into the collecting state and updating state.

The meaning of training and testing state is obvious: the agent interacts with environment, collects training data and performs update, that's training state; the testing state is to evaluate the performance of the current policy during training process.

As for the collecting state, it is defined as interacting with environments and collecting training data into the buffer;
we define the updating state as performing a model update by :meth:`~tianshou.policy.BasePolicy.update` during training process.


In order to distinguish these states, you can check the policy state by ``policy.training`` and ``policy.updating``. The state setting is as follows:

+-----------------------------------+-----------------+-----------------+
|          State for policy         | policy.training | policy.updating |
+================+==================+=================+=================+
|                | Collecting state |       True      |      False      |
| Training state +------------------+-----------------+-----------------+
|                |  Updating state  |       True      |      True       |
+----------------+------------------+-----------------+-----------------+
|           Testing state           |       False     |      False      |
+-----------------------------------+-----------------+-----------------+

``policy.updating`` is helpful to distinguish the different exploration state, for example, in DQN we don't have to use epsilon-greedy in a pure network update, so ``policy.updating`` is helpful for setting epsilon in this case.


policy.forward
^^^^^^^^^^^^^^

The ``forward`` function computes the action over given observations. The input and output is algorithm-specific but generally, the function is a mapping of ``(batch, state, ...) -> batch``.

The input batch is the environment data (e.g., observation, reward, done flag and info). It comes from either :meth:`~tianshou.data.Collector.collect` or :meth:`~tianshou.data.ReplayBuffer.sample`. The first dimension of all variables in the input ``batch`` should be equal to the batch-size.

The output is also a Batch which must contain "act" (action) and may contain "state" (hidden state of policy), "policy" (the intermediate result of policy which needs to save into the buffer, see :meth:`~tianshou.policy.BasePolicy.forward`), and some other algorithm-specific keys.

For example, if you try to use your policy to evaluate one episode (and don't want to use :meth:`~tianshou.data.Collector.collect`), use the following code-snippet:
::

    # assume env is a gym.Env
    obs, done = env.reset(), False
    while not done:
        batch = Batch(obs=[obs])  # the first dimension is batch-size
        act = policy(batch).act[0]  # policy.forward return a batch, use ".act" to extract the action
        obs, rew, done, info = env.step(act)

Here, ``Batch(obs=[obs])`` will automatically create the 0-dimension to be the batch-size. Otherwise, the network cannot determine the batch-size.


.. _process_fn:

policy.process_fn
^^^^^^^^^^^^^^^^^

The ``process_fn`` function computes some variables that depends on time-series. For example, compute the N-step or GAE returns.

Take 2-step return DQN as an example. The 2-step return DQN compute each frame's return as:

.. math::

    G_t = r_t + \gamma r_{t + 1} + \gamma^2 \max_a Q(s_{t + 2}, a)

where :math:`\gamma` is the discount factor, :math:`\gamma \in [0, 1]`. Here is the pseudocode showing the training process **without Tianshou framework**:
::

    # pseudocode, cannot work
    s = env.reset()
    buffer = Buffer(size=10000)
    agent = DQN()
    for i in range(int(1e6)):
        a = agent.compute_action(s)
        s_, r, d, _ = env.step(a)
        buffer.store(s, a, s_, r, d)
        s = s_
        if i % 1000 == 0:
            b_s, b_a, b_s_, b_r, b_d = buffer.get(size=64)
            # compute 2-step returns. How?
            b_ret = compute_2_step_return(buffer, b_r, b_d, ...)
            # update DQN policy
            agent.update(b_s, b_a, b_s_, b_r, b_d, b_ret)

Thus, we need a time-related interface for calculating the 2-step return. :meth:`~tianshou.policy.BasePolicy.process_fn` finishes this work by providing the replay buffer, the sample index, and the sample batch data. Since we store all the data in the order of time, you can simply compute the 2-step return as:
::

    class DQN_2step(BasePolicy):
        """some code"""

        def process_fn(self, batch, buffer, indice):
            buffer_len = len(buffer)
            batch_2 = buffer[(indice + 2) % buffer_len]
            # this will return a batch data where batch_2.obs is s_t+2
            # we can also get s_t+2 through:
            #   batch_2_obs = buffer.obs[(indice + 2) % buffer_len]
            # in short, buffer.obs[i] is equal to buffer[i].obs, but the former is more effecient.
            Q = self(batch_2, eps=0)  # shape: [batchsize, action_shape]
            maxQ = Q.max(dim=-1)
            batch.returns = batch.rew \
                + self._gamma * buffer.rew[(indice + 1) % buffer_len] \
                + self._gamma ** 2 * maxQ
            return batch

This code does not consider the done flag, so it may not work very well. It shows two ways to get :math:`s_{t + 2}` from the replay buffer easily in :meth:`~tianshou.policy.BasePolicy.process_fn`.

For other method, you can check out :doc:`/api/tianshou.policy`. We give the usage of policy class a high-level explanation in :ref:`pseudocode`.


Collector
---------

The :class:`~tianshou.data.Collector` enables the policy to interact with different types of environments conveniently.

:meth:`~tianshou.data.Collector.collect` is the main method of Collector: it let the policy perform (at least) a specified number of step ``n_step`` or episode ``n_episode`` and store the data in the replay buffer.

Why do we mention **at least** here? For multiple environments, we could not directly store the collected data into the replay buffer, since it breaks the principle of storing data chronologically.

The proposed solution is to add some cache buffers inside the collector. Once collecting **a full episode of trajectory**, it will move the stored data from the cache buffer to the main buffer. To satisfy this condition, the collector will interact with environments that may exceed the given step number or episode number.

The general explanation is listed in :ref:`pseudocode`. Other usages of collector are listed in :class:`~tianshou.data.Collector` documentation.


Trainer
-------

Once you have a collector and a policy, you can start writing the training method for your RL agent. Trainer, to be honest, is a simple wrapper. It helps you save energy for writing the training loop. You can also construct your own trainer: :ref:`customized_trainer`.

Tianshou has two types of trainer: :func:`~tianshou.trainer.onpolicy_trainer` and :func:`~tianshou.trainer.offpolicy_trainer`, corresponding to on-policy algorithms (such as Policy Gradient) and off-policy algorithms (such as DQN). Please check out :doc:`/api/tianshou.trainer` for the usage.


.. _pseudocode:

A High-level Explanation
------------------------

We give a high-level explanation through the pseudocode used in section :ref:`process_fn`:
::

    # pseudocode, cannot work                                       # methods in tianshou
    s = env.reset()
    buffer = Buffer(size=10000)                                     # buffer = tianshou.data.ReplayBuffer(size=10000)
    agent = DQN()                                                   # policy.__init__(...)
    for i in range(int(1e6)):                                       # done in trainer
        a = agent.compute_action(s)                                 # act = policy(batch, ...).act
        s_, r, d, _ = env.step(a)                                   # collector.collect(...)
        buffer.store(s, a, s_, r, d)                                # collector.collect(...)
        s = s_                                                      # collector.collect(...)
        if i % 1000 == 0:                                           # done in trainer
                                                                    # the following is done in policy.update(batch_size, buffer)
            b_s, b_a, b_s_, b_r, b_d = buffer.get(size=64)          # batch, indice = buffer.sample(batch_size)
            # compute 2-step returns. How?
            b_ret = compute_2_step_return(buffer, b_r, b_d, ...)    # policy.process_fn(batch, buffer, indice)
            # update DQN policy
            agent.update(b_s, b_a, b_s_, b_r, b_d, b_ret)           # policy.learn(batch, ...)


Conclusion
----------

So far, we go through the overall framework of Tianshou. Really simple, isn't it?
