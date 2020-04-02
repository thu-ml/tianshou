Basic Concepts in Tianshou
==========================

Tianshou splits a Reinforcement Learning agent training procedure into these parts: trainer, collector, policy, and data buffer. The general control flow can be described as:

.. image:: ../_static/images/concepts_arch.png
    :align: center
    :height: 300


Data Batch
----------

Tianshou provides :class:`~tianshou.data.Batch` as the internal data structure to pass any kind of data to other methods, for example, a collector gives a :class:`~tianshou.data.Batch` to policy for learning. Here is its usage:
::

    >>> import numpy as np
    >>> from tianshou.data import Batch
    >>> data = Batch(a=4, b=[5, 5], c='2312312')
    >>> data.b
    [5, 5]
    >>> data.b = np.array([3, 4, 5])
    >>> len(data.b)
    3
    >>> data.b[-1]
    5

In short, you can define a :class:`~tianshou.data.Batch` with any key-value pair.

The current implementation of Tianshou typically use 6 keys in :class:`~tianshou.data.Batch`:

* ``obs``: the observation of step :math:`t` ;
* ``act``: the action of step :math:`t` ;
* ``rew``: the reward of step :math:`t` ;
* ``done``: the done flag of step :math:`t` ;
* ``obs_next``: the observation of step :math:`t+1` ;
* ``info``: the info of step :math:`t` (in ``gym.Env``, the ``env.step()`` function return 4 arguments, and the last one is ``info``);

:class:`~tianshou.data.Batch` has other methods, including ``__getitem__``, ``append``, and ``split``:
::

    >>> data = Batch(obs=np.array([0, 11, 22]), rew=np.array([6, 6, 6]))
    >>> # here we test __getitem__
    >>> index = [2, 1]
    >>> data[index].obs
    array([22, 11])

    >>> data.append(data)  # how we use a list
    >>> data.obs
    array([0, 11, 22, 0, 11, 22])

    >>> # split whole data into multiple small batch
    >>> for d in data.split(size=2, permute=False):
    ...     print(d.obs, d.rew)
    [ 0 11] [6 6]
    [22  0] [6 6]
    [11 22] [6 6]


Data Buffer
-----------

:class:`~tianshou.data.ReplayBuffer` stores data generated from interaction between the policy and environment. It stores basically 6 types of data as mentioned above (7 types with importance weight in :class:`~tianshou.data.PrioritizedReplayBuffer`). Here is the :class:`~tianshou.data.ReplayBuffer`'s usage:
::

    >>> from tianshou.data import ReplayBuffer
    >>> buf = ReplayBuffer(size=20)
    >>> for i in range(3):
    ...     buf.add(obs=i, act=i, rew=i, done=i, obs_next=i + 1, info={})
    >>> buf.obs
    # since we set size = 20, len(buf.obs) == 20.
    array([0., 1., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0.])

    >>> buf2 = ReplayBuffer(size=10)
    >>> for i in range(15):
    ...     buf2.add(obs=i, act=i, rew=i, done=i, obs_next=i + 1, info={})
    >>> buf2.obs
    # since its size = 10, it only stores the last 10 steps' result.
    array([10., 11., 12., 13., 14.,  5.,  6.,  7.,  8.,  9.])

    >>> # move buf2's result into buf (keep it chronologically meanwhile)
    >>> buf.update(buf2)
    array([ 0.,  1.,  2.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.])

    >>> # get a random sample from buffer, the batch_data is equal to buf[incide].
    >>> batch_data, indice = buf.sample(batch_size=4)
    >>> batch_data.obs == buf[indice].obs
    array([ True,  True,  True,  True])

The :class:`~tianshou.data.ReplayBuffer` is based on ``numpy.ndarray``. Tianshou provides other type of data buffer such as :class:`~tianshou.data.ListReplayBuffer` (based on list), :class:`~tianshou.data.PrioritizedReplayBuffer` (based on Segment Tree and ``numpy.ndarray``). Check out the API documentation for more detail.


Policy
------

Tianshou aims to modularizing RL algorithms. It comes into several classes of policies in Tianshou. All of the policy classes must inherit :class:`~tianshou.policy.BasePolicy`.

A policy class typically has four parts: 

* :meth:`~tianshou.policy.BasePolicy.__init__`: initialize the policy, including coping the target network and so on;
* :meth:`~tianshou.policy.BasePolicy.__call__`: compute action with given observation;
* :meth:`~tianshou.policy.BasePolicy.process_fn`: pre-process data from the replay buffer (this function can interact with replay buffer);
* :meth:`~tianshou.policy.BasePolicy.learn`: update policy with a given batch of data.

Take 2-step return DQN as an example. The 2-step return DQN compute each frame's return as:

.. math::

    G_t = r_t + \gamma r_{t + 1} + \gamma^2 \max_a Q(s_{t + 2}, a)

Here is the pseudocode showing the training process **without Tianshou framework**:
::

    # pseudocode, cannot work
    buffer = Buffer(size=10000)
    s = env.reset()
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
            # update DQN
            agent.update(b_s, b_a, b_s_, b_r, b_d, b_ret)

Thus, we need a time-dependent interface for calculating the 2-step return. :meth:`~tianshou.policy.BasePolicy.process_fn` provides this interface by giving the replay buffer, the sample index, and the sample batch data. Since we store all the data in the order of time, you can simply compute the 2-step return as:
::

    class DQN_2step(BasePolicy):
        """some code"""

        def process_fn(self, batch, buffer, indice):
            buffer_len = len(buffer)
            batch_2 = buffer[(indice + 2) % buffer_len]
            # this will return a batch data where batch_2.obs is s_t+2
            # we can also get s_t+2 through:
            # batch_2_obs = buffer.obs[(indice + 2) % buffer_len]
            Q = self(batch_2, eps=0)  # shape: [batchsize, action_shape]
            maxQ = Q.max(dim=-1)
            batch.returns = batch.rew \
                + self._gamma * buffer.rew[(indice + 1) % buffer_len] \
                + self._gamma ** 2 * maxQ
            return batch

This code does not consider the done flag, so it may not work very well. It shows two ways to get :math:`s_{t + 2}` from the replay buffer easily in :meth:`~tianshou.policy.BasePolicy.process_fn`.

For other method, you can check out the API documentation for more detail. We give a high-level explanation through the same pseudocode:
::

    # pseudocode, cannot work
    buffer = Buffer(size=10000)
    s = env.reset()
    agent = DQN()                                                   # done in policy.__init__(...)
    for i in range(int(1e6)):                                       # done in trainer
        a = agent.compute_action(s)                                 # done in policy.__call__(batch, ...)
        s_, r, d, _ = env.step(a)                                   # done in collector.collect(...)
        buffer.store(s, a, s_, r, d)                                # done in collector.collect(...)
        s = s_                                                      # done in collector.collect(...)
        if i % 1000 == 0:                                           # done in trainer
            b_s, b_a, b_s_, b_r, b_d = buffer.get(size=64)          # done in collector.sample(batch_size)
            # compute 2-step returns. How?
            b_ret = compute_2_step_return(buffer, b_r, b_d, ...)    # done in policy.process_fn(batch, buffer, indice)
            # update DQN
            agent.update(b_s, b_a, b_s_, b_r, b_d, b_ret)           # done in policy.learn(batch, ...)


Collector
---------

The collector enables the policy to interact with different types of environments conveniently. 

* :meth:`~tianshou.data.Collector.collect`: let the policy perform (at least) a specified number of steps ``n_step`` or episodes ``n_episode`` and store the data in the replay buffer;
* :meth:`~tianshou.data.Collector.sample`: sample a data batch from replay buffer; it will call :meth:`~tianshou.policy.BasePolicy.process_fn` before returning the final batch data.

Why do we mention **at least** here? For a single environment, the collector will finish exactly ``n_step`` or ``n_episode``. However, for multiple environments, we could not directly store the collected data into the replay buffer, since it breaks the principle of storing data chronologically.

The solution is to add some cache buffers inside the collector. Once collecting **a full episode of trajectory**, it will move the stored data from the cache buffer to the main buffer. To satisfy this condition, the collector will interact with environments that may exceed the given step number or episode number.

The general explanation is listed in the pseudocode above. Other usages of collector are listed in :class:`~tianshou.data.Collector` documentation.


Trainer
-------

Once you have a collector and a policy, you can start writing the training method for your RL agent. Trainer, to be honest, is a simple wrapper. It helps you save energy for writing the training loop. You can also construct your own trainer: :ref:`customized_trainer`.

Tianshou has two types of trainer: :meth:`~tianshou.trainer.onpolicy_trainer` and :meth:`~tianshou.trainer.offpolicy_trainer`, corresponding to on-policy algorithms (such as Policy Gradient) and off-policy algorithms (such as DQN). Please check out the API documentation for the usage.

There will be more types of trainers, for instance, multi-agent trainer.


Conclusion
----------

So far, we go through the overall framework of Tianshou. Really simple, isn't it?
