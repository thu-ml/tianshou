#!/usr/bin/env python
from __future__ import absolute_import

import tensorflow as tf
import gym
import numpy as np
import time

# our lib imports here! It's ok to append path in examples
import sys
sys.path.append('..')
from tianshou.core import losses
# from tianshou.data.batch import Batch
import tianshou.data.advantage_estimation as advantage_estimation
import tianshou.core.policy.dqn as policy  # TODO: fix imports as zhusuan so that only need to import to policy
import tianshou.core.value_function.action_value as value_function

import tianshou.data.replay as replay
import tianshou.data.data_collector as data_collector


# TODO: why this solves cartpole even without training?


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    observation_dim = env.observation_space.shape
    action_dim = env.action_space.n

    clip_param = 0.2
    num_batches = 10
    batch_size = 512

    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)

    ### 1. build network with pure tf
    observation_ph = tf.placeholder(tf.float32, shape=(None,) + observation_dim)

    def my_network():
        net = tf.layers.dense(observation_ph, 32, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 32, activation=tf.nn.tanh)

        action_values = tf.layers.dense(net, action_dim, activation=None)

        return None, action_values  # no policy head

    ### 2. build policy, loss, optimizer
    dqn = value_function.DQN(my_network, observation_placeholder=observation_ph, weight_update=100)
    pi = policy.DQN(dqn)

    dqn_loss = losses.qlearning(dqn)

    total_loss = dqn_loss
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(total_loss, var_list=dqn.trainable_variables)

    ### 3. define data collection
    replay_buffer = replay()

    data_collector = data_collector(env, pi, [advantage_estimation.nstep_q_return(1, dqn)], [dqn], replay_buffer)

    ### 4. start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # assign actor to pi_old
        pi.sync_weights()  # TODO: automate this for policies with target network

        start_time = time.time()
        for i in range(100):
            # collect data
            data_collector.collect(num_episodes=50)

            # print current return
            print('Epoch {}:'.format(i))
            data_collector.statistics()

            # update network
            for _ in range(num_batches):
                feed_dict = data_collector.next_batch(batch_size)
                sess.run(train_op, feed_dict=feed_dict)

            print('Elapsed time: {:.1f} min'.format((time.time() - start_time) / 60))
