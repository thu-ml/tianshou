.. Tianshou documentation master file, created by
   sphinx-quickstart on Sat Mar 28 15:58:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Tianshou!
====================

**Tianshou** (`天授 <https://baike.baidu.com/item/%E5%A4%A9%E6%8E%88>`_) is a reinforcement learning platform based on pure PyTorch. Unlike existing reinforcement learning libraries, which are mainly based on TensorFlow, have many nested classes, unfriendly API, or slow-speed, Tianshou provides a fast-speed framework and pythonic API for building the deep reinforcement learning agent. The supported interface algorithms include:

* :class:`~tianshou.policy.PGPolicy` `Policy Gradient <https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>`_
* :class:`~tianshou.policy.DQNPolicy` `Deep Q-Network <https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf>`_
* :class:`~tianshou.policy.DQNPolicy` `Double DQN <https://arxiv.org/pdf/1509.06461.pdf>`_ with n-step returns
* :class:`~tianshou.policy.A2CPolicy` `Advantage Actor-Critic <https://openai.com/blog/baselines-acktr-a2c/>`_
* :class:`~tianshou.policy.DDPGPolicy` `Deep Deterministic Policy Gradient <https://arxiv.org/pdf/1509.02971.pdf>`_
* :class:`~tianshou.policy.PPOPolicy` `Proximal Policy Optimization <https://arxiv.org/pdf/1707.06347.pdf>`_
* :class:`~tianshou.policy.TD3Policy` `Twin Delayed DDPG <https://arxiv.org/pdf/1802.09477.pdf>`_
* :class:`~tianshou.policy.SACPolicy` `Soft Actor-Critic <https://arxiv.org/pdf/1812.05905.pdf>`_
* :class:`~tianshou.policy.ImitationPolicy` Imitation Learning
* :meth:`~tianshou.policy.BasePolicy.compute_episodic_return` `Generalized Advantage Estimation <https://arxiv.org/pdf/1506.02438.pdf>`_


Tianshou supports parallel workers for all algorithms as well. All of these algorithms are reformatted as replay-buffer based algorithms.


Installation
------------

Tianshou is currently hosted on `PyPI <https://pypi.org/project/tianshou/>`_. You can simply install Tianshou with the following command:
::

    pip3 install tianshou -U

You can also install with the newest version through GitHub:
::

    pip3 install git+https://github.com/thu-ml/tianshou.git@master

After installation, open your python console and type
::

    import tianshou as ts
    print(ts.__version__)

If no error occurs, you have successfully installed Tianshou.

Tianshou is still under development, you can also check out the documents in stable version through `tianshou.readthedocs.io/en/stable/ <https://tianshou.readthedocs.io/en/stable/>`_.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/dqn
   tutorials/concepts
   tutorials/trick
   tutorials/cheatsheet

.. toctree::
   :maxdepth: 1
   :caption: API Docs

   api/tianshou.data
   api/tianshou.env
   api/tianshou.policy
   api/tianshou.trainer
   api/tianshou.exploration
   api/tianshou.utils

.. toctree::
   :maxdepth: 1
   :caption: Community

   contributing
   contributor


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
