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

.. _policy_concept:

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

:class:`~tianshou.data.Collector` has one main method :meth:`~tianshou.data.Collector.collect`: it let the policy perform (at least) a specified number of step ``n_step`` or episode ``n_episode`` and store the data in the replay buffer.

Why do we mention **at least** here? For multiple environments, we could not directly store the collected data into the replay buffer, since it breaks the principle of storing data chronologically.

The solution is to add some cache buffers inside the collector. Once collecting **a full episode of trajectory**, it will move the stored data from the cache buffer to the main buffer. To satisfy this condition, the collector will interact with environments that may exceed the given step number or episode number.

The general explanation is listed in :ref:`pseudocode`. Other usages of collector are listed in :class:`~tianshou.data.Collector` documentation.


Trainer
-------

Once you have a collector and a policy, you can start writing the training method for your RL agent. Trainer, to be honest, is a simple wrapper. It helps you save energy for writing the training loop. You can also construct your own trainer: :ref:`customized_trainer`.

Tianshou has two types of trainer: :func:`~tianshou.trainer.onpolicy_trainer` and :func:`~tianshou.trainer.offpolicy_trainer`, corresponding to on-policy algorithms (such as Policy Gradient) and off-policy algorithms (such as DQN). Please check out :doc:`/api/tianshou.trainer` for the usage.


.. _pseudocode:

A High-level Explanation
------------------------

We give a high-level explanation through the pseudocode used in section :ref:`policy_concept`:
::

    # pseudocode, cannot work                                       # methods in tianshou
    s = env.reset()
    buffer = Buffer(size=10000)                                     # buffer = tianshou.data.ReplayBuffer(size=10000)
    agent = DQN()                                                   # policy.__init__(...)
    for i in range(int(1e6)):                                       # done in trainer
        a = agent.compute_action(s)                                 # policy(batch, ...)
        s_, r, d, _ = env.step(a)                                   # collector.collect(...)
        buffer.store(s, a, s_, r, d)                                # collector.collect(...)
        s = s_                                                      # collector.collect(...)
        if i % 1000 == 0:                                           # done in trainer
                                                                    # the following is done in policy.update(batch_size, buffer)
            b_s, b_a, b_s_, b_r, b_d = buffer.get(size=64)          # buffer.sample(batch_size)
            # compute 2-step returns. How?
            b_ret = compute_2_step_return(buffer, b_r, b_d, ...)    # policy.process_fn(batch, buffer, indice)
            # update DQN policy
            agent.update(b_s, b_a, b_s_, b_r, b_d, b_ret)           # policy.learn(batch, ...)


Conclusion
----------

So far, we go through the overall framework of Tianshou. Really simple, isn't it?
