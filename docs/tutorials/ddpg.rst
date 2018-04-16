DDPG (Deep Deterministic Policy Gradient) with TianShou
=======================================================

DDPG (Deep Deterministic Policy Gradient) :cite:`lillicrap2015continuous` is a popular RL algorithm
for continuous control. In this tutorial, we show, step by step, how to write neural networks and
use DDPG to train the networks with Tianshou. .. The full script is at

TianShou is built following a very simple idea: Deep RL still trains deep neural nets with some loss
functions or optimizers on minibatches of data. The only differences between Deep RL and supervised learning
are the RL-specific loss functions/optimizers and acquisition of training data.
Therefore, we wrap up the RL-specific parts in TianShou, while still expose the TensorFlow-level interfaces
for you to train your neural policies/value functions. As a result,
doing Deep RL with TianShou is almost as simple as
doing supervised learning with TensorFlow.

We now demonstrate a typical routine of doing Deep RL with TianShou.


Make an Environment
-------------------

Always first of all, you have to make an environment for your agent to act in. For the environment interfaces
we follow the convention of `OpenAI Gym <https://github.com/openai/gym>`_. Just do ::

    pip install gym

in your terminal
if you haven't installed it yet, and you will be able to run the simple example scripts
provided by us.

Then, in your Python code, simply make the environment::

    env = gym.make('Pendulum-v0')

Pendulum-v0 is a simple environment with a continuous action space, for which DDPG applies. You have to
identify the whether the action space is continuous or discrete, and apply eligible algorithms.
DQN :cite:`mnih2015human`, for example, could only be applied to discrete action spaces, while almost all
other policy gradient methods could be applied to both, depending on the probability distribution on
action.


Build the Networks
------------------

As in supervised learning, we proceed to build the neural networks with TensorFlow. Contrary to existing
Deep RL libraries (
`keras-rl <https://github.com/keras-rl/keras-rl>`_,
`rllab <https://github.com/rll/rllab>`_,
`TensorForce <https://github.com/reinforceio/tensorforce>`_
), which could only accept a config specification of network layers and neurons,
TianShou naturally supports **all** TensorFlow APIs when building the neural networks. In fact, the networks
in TianShou are still built with direct TensorFlow APIs without any encapsulation, making it fairly
easy to use dropout, batch-norm, skip-connections and other advanced neural architectures.

As usual, we start with placeholders that define the network input: ::

    observation_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    observation_ph = tf.placeholder(tf.float32, shape=(None,) + observation_dim)
    action_ph = tf.placeholder(tf.float32, shape=(None,) + action_dim)

And build MLPs for this simple environment. DDPG requires both an actor (the deterministic policy)
and a critic (:math:`Q(s, a)`)::

    net = tf.layers.dense(observation_ph, 32, activation=tf.nn.relu)
    net = tf.layers.dense(net, 32, activation=tf.nn.relu)
    action = tf.layers.dense(net, action_dim[0], activation=None)

    action_value_input = tf.concat([observation_ph, action_ph], axis=1)
    net = tf.layers.dense(action_value_input, 64, activation=tf.nn.relu)
    net = tf.layers.dense(net, 64, activation=tf.nn.relu)
    action_value = tf.layers.dense(net, 1, activation=None)

However, DDPG also requires a slowly-tracking copy of these networks as the "target networks". Target networks
are common in RL algorithms, where many off-policy algorithms explicitly requires them to stabilize training
:cite:`mnih2015human,lillicrap2015continuous`, and they could simplify the construction of other objectives
such as the probability ration or the KL divergence
between the new and old action distribution :cite:`schulman2015trust,schulman2017proximal`.

Due to such universality of an old-copy the neural networks (we term it "old net", considering not all
such networks are used to compute targets),
we introduce the first paradigm of TianShou: ::

    All parts of the TensorFlow graph construction, except placeholder instantiation,
    have to be wrapped in a single parameter-less Python function by you.
    The function must return a doublet, (policy head, value head),
    with the unnecessary head (if any) set to ``None``.

.. note::

    This paradigm also prescribes the return value of the network function. Such architecture with two
    "head"s is established by the indispensable role of policies and value functions in RL, and also
    supported by the use of both networks in, for example,
    :cite:`mnih2016asynchronous,lillicrap2015continuous,silver2017mastering`. This paradigm also allows
    arbitrary layer sharing between the policy and value networks.

TianShou will then call this function to create the network graphs and optionally the "old net"s according to
a single parameter set by you, as in::

    def my_network():
        net = tf.layers.dense(observation_ph, 32, activation=tf.nn.relu)
        net = tf.layers.dense(net, 32, activation=tf.nn.relu)
        action = tf.layers.dense(net, action_dim[0], activation=None)

        action_value_input = tf.concat([observation_ph, action_ph], axis=1)
        net = tf.layers.dense(action_value_input, 64, activation=tf.nn.relu)
        net = tf.layers.dense(net, 64, activation=tf.nn.relu)
        action_value = tf.layers.dense(net, 1, activation=None)

        return action, action_value

    actor = policy.Deterministic(my_network, observation_placeholder=observation_ph,
                                 has_old_net=True)
    critic = value_function.ActionValue(my_network, observation_placeholder=observation_ph,
                                        action_placeholder=action_ph, has_old_net=True)

You pass the function handler ``my_network`` to TianShou's policy and value network wrappers,
and also the corresponding placeholders. The ``has_old_net`` controls the construction of the
old net, and is ``False`` by default. When set to ``True`` as in this tutorial, the ``actor``
and ``critic`` will automatically create two sets of networks, the current network and the
old net, and manages them together.

The only behavior provided by the network wrappers on old net is :func:`sync_weights`, which copies
the weights of the current network to the old net. Although it's sufficient for other scenarios with old nets
:cite:`mnih2015human,schulman2015trust,schulman2017proximal`, DDPG proposes soft update on the
old nets. Therefore, TianShou provides an additional utility for such soft update: ::

    soft_update_op = get_soft_update_op(1e-2, [actor, critic])

For detailed usage please refer to the API doc of :func:`tianshou.core.utils.get_soft_update_op`. This utility
function gives you the runnable TensorFlow ops the perform soft update, i.e., you can simply do
``sess.run(soft_update_op)`` whenever you want soft update.


Construct Optimization Methods
------------------------------

One of the two key differences between Deep RL and supervised learning is the optimization algorithms.
Contrary to existing
Deep RL projects (
`OpenAI Baselines <https://github.com/openai/baselines>`_,
`Coach <https://github.com/NervanaSystems/coach>`_,
`keras-rl <https://github.com/keras-rl/keras-rl>`_,
`rllab <https://github.com/rll/rllab>`_,
`TensorForce <https://github.com/reinforceio/tensorforce>`_
), which wraps all the optimization operations in one class, we provide optimization techniques only
to the least necessary level, allowing natural combination of, for example,
native TensorFlow optimizers and gradient clipping operations. We identify three levels of optimization
encapsulation, namely loss, gradient and optimizer, and implement RL techniques to one of these levels.

TianShou's ``loss`` resembles ``tf.losses``, and to apply L2 loss on the critic in DDPG you could simply do::

    critic_loss = losses.value_mse(critic)
    critic_optimizer = tf.train.AdamOptimizer(1e-3)
    critic_train_op = critic_optimizer.minimize(critic_loss, var_list=list(critic.trainable_variables))

.. note::

    The ``trainable_variables`` property of network wrappers returns a Python **set** rather than
    a Python list. This is for the cases where actor and critic have shared layers, so you have to
    explicitly convert it to a list.

For the deterministic policy gradient :cite:`lillicrap2015continuous` which is difficulty to be
conceptualized as gradients over a loss function under TianShou's paradigm, we wrap it up into the
``gradient`` level, which directly computes and returns gradients as
:func:`tf.train.Optimizer.compute_gradients` does. It can then be seamlessly combined with
:func:`tf.train.Optimizer.apply_gradients` to optimize the actor: ::

    dpg_grads_vars = opt.DPG(actor, critic)
    actor_optimizer = tf.train.AdamOptimizer(1e-3)
    actor_train_op = actor_optimizer.apply_gradients(dpg_grads_vars)


Specify Data Acquisition
------------------------

The other key differences between Deep RL and supervised learning is the data acquisition process.
Contrary to existing
Deep RL projects (
`OpenAI Baselines <https://github.com/openai/baselines>`_,
`Coach <https://github.com/NervanaSystems/coach>`_,
`keras-rl <https://github.com/keras-rl/keras-rl>`_,
`rllab <https://github.com/rll/rllab>`_,
`TensorForce <https://github.com/reinforceio/tensorforce>`_
), which mixes up data acquisition and all the optimization operations in one class, we separate it from
optimization, facilitating more opportunities of combinations.

First, we instantiate a replay buffer to store the off-policy experiences ::

    data_buffer = VanillaReplayBuffer(capacity=10000, nstep=1)

All data buffers in TianShou store only the raw data of each episode, i.e., frames of data in the canonical
RL form of tuple: (observation, action, reward, done_flag). Such raw data have to be processed before feeding
to the optimization algorithms, so we specify the processing functions in a Python list ::

    process_functions = [advantage_estimation.ddpg_return(actor, critic)]

We are now ready to fully specify the data acquisition process ::

    data_collector = DataCollector(
        env=env,
        policy=actor,
        data_buffer=data_buffer,
        process_functions=process_functions,
        managed_networks=[actor, critic]
    )

The ``process_functions`` should be a list of Python callables, which you could also implement your own
following the APIs in :mod:`tianshou.data.advantage_estimation`. You should also pass a Python list of
network wrappers, ``managed_networks`` (in this case ``[actor, critic]``), to ``DataCollector``, which
brings up the second paradigm of TianShou: ::

    All canonical RL placeholders (observation, action, return/advantage)
    are automatically managed by TianShou.
    You only have to create at most the placeholders for observation and action.

Other placeholders, such as the dropout ratio and batch-norm phase, should be managed by you, though.
We provide an entry ``my_feed_dict`` in all functions that may involve such cases.


Start Training!
---------------

Finally, we are all set and let the training begin!::

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # assign actor to pi_old
        actor.sync_weights()
        critic.sync_weights()

        start_time = time.time()
        data_collector.collect(num_timesteps=5000)  # warm-up
        for i in range(int(1e8)):
            # collect data
            data_collector.collect(num_timesteps=1, episode_cutoff=200)

            # train critic
            feed_dict = data_collector.next_batch(batch_size)
            sess.run(critic_train_op, feed_dict=feed_dict)

            # recompute action
            data_collector.denoise_action(feed_dict)

            # train actor
            sess.run(actor_train_op, feed_dict=feed_dict)

            # update target networks
            sess.run(soft_update_op)

            # test every 1000 training steps
            if i % 1000 == 0:
                print('Step {}, elapsed time: {:.1f} min'.format(i, (time.time() - start_time) / 60))
                test_policy_in_env(actor, env, num_episodes=5, episode_cutoff=200)

Note that, to optimize the actor in DDPG, we have to use the noiseless action computed by the current
actor rather than the sampled action during interaction with the environment, hence
``data_collector.denoise_action(feed_dict)`` before running ``actor_train_op``.

We've made the effort for the training process in TianShou also resembles conventional supervised learning
with TensorFlow. Our ``DataCollector`` automatically the ``feed_dict`` for the canonical RL placeholders.
Enjoy and have fun!




.. rubric:: References

.. bibliography:: ../refs.bib
    :style: unsrtalpha
