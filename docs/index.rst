
Welcome to Tianshou!
====================

**Tianshou** (`天授 <https://baike.baidu.com/item/%E5%A4%A9%E6%8E%88>`_) is a reinforcement learning platform based on pure PyTorch. Unlike existing reinforcement learning libraries, which are mainly based on TensorFlow, have many nested classes, unfriendly API, or slow-speed, Tianshou provides a fast-speed framework and pythonic API for building the deep reinforcement learning agent. The supported interface algorithms include:

* :class:`~tianshou.algorithm.modelfree.dqn.DQN` `Deep Q-Network <https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf>`_
* :class:`~tianshou.algorithm.modelfree.dqn.DQN` `Double DQN <https://arxiv.org/pdf/1509.06461.pdf>`_
* :class:`~tianshou.algorithm.modelfree.dqn.DQN` `Dueling DQN <https://arxiv.org/pdf/1511.06581.pdf>`_
* :class:`~tianshou.algorithm.modelfree.bdqn.BDQN` `Branching DQN <https://arxiv.org/pdf/1711.08946.pdf>`_
* :class:`~tianshou.algorithm.modelfree.c51.C51` `Categorical DQN <https://arxiv.org/pdf/1707.06887.pdf>`_
* :class:`~tianshou.algorithm.modelfree.rainbow.RainbowDQN` `Rainbow DQN <https://arxiv.org/pdf/1710.02298.pdf>`_
* :class:`~tianshou.algorithm.modelfree.qrdqn.QRDQN` `Quantile Regression DQN <https://arxiv.org/pdf/1710.10044.pdf>`_
* :class:`~tianshou.algorithm.modelfree.iqn.IQN` `Implicit Quantile Network <https://arxiv.org/pdf/1806.06923.pdf>`_
* :class:`~tianshou.algorithm.modelfree.fqf.FQF` `Fully-parameterized Quantile Function <https://arxiv.org/pdf/1911.02140.pdf>`_
* :class:`~tianshou.algorithm.modelfree.reinforce.Reinforce` `Reinforce/Vanilla Policy Gradients <https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>`_
* :class:`~tianshou.algorithm.modelfree.npg.NPG` `Natural Policy Gradient <https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf>`_
* :class:`~tianshou.algorithm.modelfree.a2c.A2C` `Advantage Actor-Critic <https://openai.com/blog/baselines-acktr-a2c/>`_
* :class:`~tianshou.algorithm.modelfree.trpo.TRPO` `Trust Region Policy Optimization <https://arxiv.org/pdf/1502.05477.pdf>`_
* :class:`~tianshou.algorithm.modelfree.ppo.PPO` `Proximal Policy Optimization <https://arxiv.org/pdf/1707.06347.pdf>`_
* :class:`~tianshou.algorithm.modelfree.ddpg.DDPG` `Deep Deterministic Policy Gradient <https://arxiv.org/pdf/1509.02971.pdf>`_
* :class:`~tianshou.algorithm.modelfree.td3.TD3` `Twin Delayed DDPG <https://arxiv.org/pdf/1802.09477.pdf>`_
* :class:`~tianshou.algorithm.modelfree.sac.SAC` `Soft Actor-Critic <https://arxiv.org/pdf/1812.05905.pdf>`_
* :class:`~tianshou.algorithm.modelfree.redq.REDQ` `Randomized Ensembled Double Q-Learning <https://arxiv.org/pdf/2101.05982.pdf>`_
* :class:`~tianshou.algorithm.modelfree.discrete_sac.DiscreteSAC` `Discrete Soft Actor-Critic <https://arxiv.org/pdf/1910.07207.pdf>`_
* :class:`~tianshou.algorithm.imitation.imitation_base.ImitationPolicy` Imitation Learning
* :class:`~tianshou.algorithm.imitation.bcq.BCQ` `Batch-Constrained deep Q-Learning <https://arxiv.org/pdf/1812.02900.pdf>`_
* :class:`~tianshou.algorithm.imitation.cql.CQL` `Conservative Q-Learning <https://arxiv.org/pdf/2006.04779.pdf>`_
* :class:`~tianshou.algorithm.imitation.td3_bc.TD3BC` `Twin Delayed DDPG with Behavior Cloning <https://arxiv.org/pdf/2106.06860.pdf>`_
* :class:`~tianshou.algorithm.imitation.discrete_cql.DiscreteCQL` `Discrete Conservative Q-Learning <https://arxiv.org/pdf/2006.04779.pdf>`_
* :class:`~tianshou.algorithm.imitation.discrete_bcq.DiscreteBCQ` `Discrete Batch-Constrained deep Q-Learning <https://arxiv.org/pdf/1910.01708.pdf>`_
* :class:`~tianshou.algorithm.imitation.discrete_crr.DiscreteCRR` `Critic Regularized Regression <https://arxiv.org/pdf/2006.15134.pdf>`_
* :class:`~tianshou.algorithm.imitation.gail.GAIL` `Generative Adversarial Imitation Learning <https://arxiv.org/pdf/1606.03476.pdf>`_
* :class:`~tianshou.algorithm.modelbased.psrl.PSRLPolicy` `Posterior Sampling Reinforcement Learning <https://www.ece.uvic.ca/~bctill/papers/learning/Strens_2000.pdf>`_
* :class:`~tianshou.algorithm.modelbased.icm.ICMOffPolicyWrapper`, :class:`~tianshou.algorithm.modelbased.icm.ICMOnPolicyWrapper` `Intrinsic Curiosity Module <https://arxiv.org/pdf/1705.05363.pdf>`_
* :class:`~tianshou.data.buffer.prio.PrioritizedReplayBuffer` `Prioritized Experience Replay <https://arxiv.org/pdf/1511.05952.pdf>`_
* :meth:`~tianshou.algorithm.algorithm_base.Algorithm.compute_episodic_return` `Generalized Advantage Estimator <https://arxiv.org/pdf/1506.02438.pdf>`_
* :class:`~tianshou.data.buffer.her.HERReplayBuffer` `Hindsight Experience Replay <https://arxiv.org/pdf/1707.01495.pdf>`_


Installation
------------

Tianshou is available through `PyPI <https://pypi.org/project/tianshou/>`_.
New releases require Python >= 3.11.

Install Tianshou with the following command:

.. code-block:: bash

    $ pip install tianshou

Alternatively, install the current version on GitHub:

.. code-block:: bash

    $ pip install git+https://github.com/thu-ml/tianshou.git@master --upgrade

After installation, open your python console and type
::

    import tianshou
    print(tianshou.__version__)

If no error occurs, you have successfully installed Tianshou.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
