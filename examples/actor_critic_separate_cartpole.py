#!/usr/bin/env python
from __future__ import absolute_import

import tensorflow as tf
import time
import numpy as np

# our lib imports here! It's ok to append path in examples
import sys
sys.path.append('..')
from tianshou.core import losses
from tianshou.data.batch import Batch
import tianshou.data.advantage_estimation as advantage_estimation
import tianshou.core.policy.stochastic as policy  # TODO: fix imports as zhusuan so that only need to import to policy
import tianshou.core.value_function.state_value as value_function

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize


# for tutorial purpose, placeholders are explicitly appended with '_ph' suffix

if __name__ == '__main__':
    env = normalize(CartpoleEnv())
    observation_dim = env.observation_space.shape
    action_dim = env.action_space.flat_dim

    clip_param = 0.2
    num_batches = 10
    batch_size = 128

    seed = 10
    np.random.seed(seed)
    tf.set_random_seed(seed)

    ### 1. build network with pure tf
    observation_ph = tf.placeholder(tf.float32, shape=(None,) + observation_dim)

    def my_network():
        net = tf.layers.dense(observation_ph, 32, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 32, activation=tf.nn.tanh)

        action_mean = tf.layers.dense(net, action_dim, activation=None)
        action_logstd = tf.get_variable('action_logstd', shape=(action_dim, ))

        net = tf.layers.dense(observation_ph, 32, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 32, activation=tf.nn.tanh)
        value = tf.layers.dense(net, 1, activation=None)

        return action_mean, action_logstd, value

    ### 2. build policy, critic, loss, optimizer
    actor = policy.Normal(my_network, observation_placeholder=observation_ph, weight_update=1)
    critic = value_function.StateValue(my_network, observation_placeholder=observation_ph)

    actor_loss = losses.REINFORCE(actor)
    critic_loss = losses.state_value_mse(critic)

    actor_optimizer = tf.train.AdamOptimizer(1e-4)
    actor_train_op = actor_optimizer.minimize(actor_loss, var_list=actor.trainable_variables)

    critic_optimizer = tf.train.RMSPropOptimizer(1e-4)
    critic_train_op = critic_optimizer.minimize(critic_loss, var_list=critic.trainable_variables)

    ### 3. define data collection
    data_collector = Batch(env, actor,
                           [advantage_estimation.gae_lambda(1, critic), advantage_estimation.nstep_return(1, critic)],
                           [actor, critic])

    ### 4. start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        start_time = time.time()
        for i in range(100):
            # collect data
            data_collector.collect(num_episodes=20)

            # print current return
            print('Epoch {}:'.format(i))
            data_collector.statistics()

            # update network
            for _ in range(num_batches):
                feed_dict = data_collector.next_batch(batch_size)
                sess.run([actor_train_op, critic_train_op], feed_dict=feed_dict)

            print('Elapsed time: {:.1f} min'.format((time.time() - start_time) / 60))