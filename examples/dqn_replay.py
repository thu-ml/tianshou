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

from tianshou.data.replay_buffer.vanilla import VanillaReplayBuffer
from tianshou.data.data_collector import DataCollector


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
    replay_buffer = VanillaReplayBuffer(capacity=1e5, nstep=1)

    process_functions = [advantage_estimation.nstep_q_return(1, dqn)]
    managed_networks = [dqn]

    data_collector = DataCollector(
        env=env,
        policy=pi,
        data_buffer=replay_buffer,
        process_functions=process_functions,
        managed_networks=managed_networks
    )

    ### 4. start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # assign actor to pi_old
        pi.sync_weights()  # TODO: automate this for policies with target network

        start_time = time.time()
        epsilon = 0.5
        pi.set_epsilon_train(epsilon)
        data_collector.collect(num_timesteps=1e3)  # warm-up
        for i in range(int(1e8)):  # number of training steps
            # anneal epsilon step-wise
            if (i + 1) % 1e4 == 0 and epsilon > 0.1:
                epsilon -= 0.1
                pi.set_epsilon_train(epsilon)

            # collect data
            data_collector.collect()

            # update network
            for _ in range(num_batches):
                feed_dict = data_collector.next_batch(batch_size)
                sess.run(train_op, feed_dict=feed_dict)

            print('Elapsed time: {:.1f} min'.format((time.time() - start_time) / 60))

            # test every 1000 training steps
            # tester could share some code with batch!
            if i % 1000 == 0:
                # epsilon 0.05 as in nature paper
                pi.set_epsilon_test(0.05)
                test(env, pi)  # go for act_test of pi, not act
