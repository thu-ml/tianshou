.. Tianshou documentation master file, created by
   sphinx-quickstart on Sat Mar 28 15:58:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Tianshou!
====================

**Tianshou** (`天授 <https://baike.baidu.com/item/%E5%A4%A9%E6%8E%88>`_) is a reinforcement learning platform based on pure PyTorch. Unlike existing reinforcement learning libraries, which are mainly based on TensorFlow, have many nested classes, unfriendly API, or slow-speed, Tianshou provides a fast-speed framework and pythonic API for building the deep reinforcement learning agent. The supported interface algorithms include:

* :class:`~tianshou.policy.PGPolicy` `Policy Gradient <https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>`_
* :class:`~tianshou.policy.DQNPolicy` `Deep Q-Network <https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf>`_
* :class:`~tianshou.policy.DQNPolicy` `Double DQN <https://arxiv.org/pdf/1509.06461.pdf>`_
* :class:`~tianshou.policy.DQNPolicy` `Dueling DQN <https://arxiv.org/pdf/1511.06581.pdf>`_
* :class:`~tianshou.policy.A2CPolicy` `Advantage Actor-Critic <https://openai.com/blog/baselines-acktr-a2c/>`_
* :class:`~tianshou.policy.DDPGPolicy` `Deep Deterministic Policy Gradient <https://arxiv.org/pdf/1509.02971.pdf>`_
* :class:`~tianshou.policy.PPOPolicy` `Proximal Policy Optimization <https://arxiv.org/pdf/1707.06347.pdf>`_
* :class:`~tianshou.policy.TD3Policy` `Twin Delayed DDPG <https://arxiv.org/pdf/1802.09477.pdf>`_
* :class:`~tianshou.policy.SACPolicy` `Soft Actor-Critic <https://arxiv.org/pdf/1812.05905.pdf>`_
* :class:`~tianshou.policy.ImitationPolicy` Imitation Learning
* :class:`~tianshou.data.PrioritizedReplayBuffer` `Prioritized Experience Replay <https://arxiv.org/pdf/1511.05952.pdf>`_
* :meth:`~tianshou.policy.BasePolicy.compute_episodic_return` `Generalized Advantage Estimator <https://arxiv.org/pdf/1506.02438.pdf>`_

Here is Tianshou's other features:

* Elegant framework, using only ~2000 lines of code
* Support parallel environment sampling for all algorithms: :ref:`parallel_sampling`
* Support recurrent state representation in actor network and critic network (RNN-style training for POMDP): :ref:`rnn_training`
* Support any type of environment state (e.g. a dict, a self-defined class, ...): :ref:`self_defined_env`
* Support customized training process: :ref:`customize_training`
* Support n-step returns estimation :meth:`~tianshou.policy.BasePolicy.compute_nstep_return` and prioritized experience replay for all Q-learning based algorithms
* Support multi-agent RL: :doc:`/tutorials/tictactoe`

中文文档位于 `https://tianshou.readthedocs.io/zh/latest/ <https://tianshou.readthedocs.io/zh/latest/>`_

Installation
------------

Tianshou is currently hosted on `PyPI <https://pypi.org/project/tianshou/>`_. You can simply install Tianshou with the following command (with Python >= 3.6):

.. code-block:: bash

    $ pip install tianshou

You can also install with the newest version through GitHub:

.. code-block:: bash

    # latest release
    $ pip install git+https://github.com/thu-ml/tianshou.git@master
    # develop version
    $ pip install git+https://github.com/thu-ml/tianshou.git@dev

If you use Anaconda or Miniconda, you can install Tianshou through the following command lines:

.. code-block:: bash

    # create a new virtualenv and install pip, change the env name if you like
    $ conda create -n myenv pip
    # activate the environment
    $ conda activate myenv
    # install tianshou
    $ pip install tianshou

After installation, open your python console and type
::

    import tianshou as ts
    print(ts.__version__)

If no error occurs, you have successfully installed Tianshou.

Tianshou is still under development, you can also check out the documents in stable version through `tianshou.readthedocs.io/en/stable/ <https://tianshou.readthedocs.io/en/stable/>`_ and the develop version through `tianshou.readthedocs.io/en/dev/ <https://tianshou.readthedocs.io/en/dev/>`_.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/dqn
   tutorials/concepts
   tutorials/batch
   tutorials/tictactoe
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
