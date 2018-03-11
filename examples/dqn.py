#!/usr/bin/env python
import tensorflow as tf
import gym
import numpy as np
import time
import logging
logging.basicConfig(level=logging.INFO)

# our lib imports here! It's ok to append path in examples
import sys
sys.path.append('..')
from tianshou.core import losses
import tianshou.data.advantage_estimation as advantage_estimation
import tianshou.core.policy.dqn as policy
import tianshou.core.value_function.action_value as value_function

from tianshou.data.data_buffer.vanilla import VanillaReplayBuffer
from tianshou.data.data_collector import DataCollector
from tianshou.data.tester import test_policy_in_env


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    observation_dim = env.observation_space.shape
    action_dim = env.action_space.n

    ### 1. build network with pure tf
    observation_ph = tf.placeholder(tf.float32, shape=(None,) + observation_dim)

    def my_network():
        net = tf.layers.dense(observation_ph, 32, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 32, activation=tf.nn.tanh)

        action_values = tf.layers.dense(net, action_dim, activation=None)

        return None, action_values  # no policy head

    ### 2. build policy, loss, optimizer
    dqn = value_function.DQN(my_network, observation_placeholder=observation_ph, weight_update=800)
    pi = policy.DQN(dqn)

    dqn_loss = losses.qlearning(dqn)

    total_loss = dqn_loss
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(total_loss, var_list=dqn.trainable_variables)

    ### 3. define data collection
    replay_buffer = VanillaReplayBuffer(capacity=2e4, nstep=1)

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
    # hyper-parameters
    batch_size = 128
    replay_buffer_warmup = 1000
    epsilon_decay_interval = 500
    epsilon = 0.6
    test_interval = 5000

    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # assign actor to pi_old
        pi.sync_weights()  # TODO: rethink and redesign target network interface

        start_time = time.time()
        pi.set_epsilon_train(epsilon)
        data_collector.collect(num_timesteps=replay_buffer_warmup)  # TODO: uniform random warm-up
        for i in range(int(1e8)):  # number of training steps
            # anneal epsilon step-wise
            if (i + 1) % epsilon_decay_interval == 0 and epsilon > 0.1:
                epsilon -= 0.1
                pi.set_epsilon_train(epsilon)

            # collect data
            data_collector.collect(num_timesteps=4)

            # update network
            feed_dict = data_collector.next_batch(batch_size)
            sess.run(train_op, feed_dict=feed_dict)

            # test every 1000 training steps
            # tester could share some code with batch!
            if i % test_interval == 0:
                print('Step {}, elapsed time: {:.1f} min'.format(i, (time.time() - start_time) / 60))
                # epsilon 0.05 as in nature paper
                pi.set_epsilon_test(0.05)
                test_policy_in_env(pi, env, num_timesteps=1000)
