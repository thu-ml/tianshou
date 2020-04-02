.. Tianshou documentation master file, created by
   sphinx-quickstart on Sat Mar 28 15:58:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Tianshou!
====================

**Tianshou** (`天授 <https://baike.baidu.com/item/%E5%A4%A9%E6%8E%88>`_) is a reinforcement learning platform based on pure PyTorch. Unlike existing reinforcement learning libraries, which are mainly based on TensorFlow, have many nested classes, unfriendly API, or slow-speed, Tianshou provides a fast-speed framework and pythonic API for building the deep reinforcement learning agent. The supported interface algorithms include:

* `Policy Gradient (PG) <https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>`_
* `Deep Q-Network (DQN) <https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf>`_
* `Double DQN (DDQN) <https://arxiv.org/pdf/1509.06461.pdf>`_ with n-step returns
* `Advantage Actor-Critic (A2C) <https://openai.com/blog/baselines-acktr-a2c/>`_
* `Deep Deterministic Policy Gradient (DDPG) <https://arxiv.org/pdf/1509.02971.pdf>`_
* `Proximal Policy Optimization (PPO) <https://arxiv.org/pdf/1707.06347.pdf>`_
* `Twin Delayed DDPG (TD3) <https://arxiv.org/pdf/1802.09477.pdf>`_
* `Soft Actor-Critic (SAC) <https://arxiv.org/pdf/1812.05905.pdf>`_


Tianshou supports parallel workers for all algorithms as well. All of these algorithms are reformatted as replay-buffer based algorithms.


Installation
------------

Tianshou is currently hosted on `PyPI <https://pypi.org/project/tianshou/>`_. You can simply install Tianshou with the following command:
::

    pip3 install tianshou

You can also install with the newest version through GitHub:
::

    pip3 install git+https://github.com/thu-ml/tianshou.git@master

After installation, open your python console and type
::

    import tianshou as ts
    print(ts.__version__)

If no error occurs, you have successfully installed Tianshou.


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/dqn
   tutorials/concepts

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


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
