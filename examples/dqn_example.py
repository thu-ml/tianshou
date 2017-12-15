#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import time
import gym

# our lib imports here!
import sys
sys.path.append('..')
import tianshou.core.losses as losses
from tianshou.data.replay import Replay
import tianshou.data.advantage_estimation as advantage_estimation
import tianshou.core.policy as policy


def policy_net(observation, action_dim):
    """
    Constructs the policy network. NOT NEEDED IN THE LIBRARY! this is pure tf

    :param observation: Placeholder for the observation. A tensor of shape (bs, x, y, channels)
    :param action_dim: int. The number of actions.
    :param scope: str. Specifying the scope of the variables.
    """
    net = tf.layers.conv2d(observation, 16, 8, 4, 'valid', activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 32, 4, 2, 'valid', activation=tf.nn.relu)
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 256, activation=tf.nn.relu)

    q_values = tf.layers.dense(net, action_dim)

    return q_values


if __name__ == '__main__':
    env = gym.make('PongNoFrameskip-v4')
    observation_dim = env.observation_space.shape
    action_dim = env.action_space.n

    # 1. build network with pure tf
    observation = tf.placeholder(tf.float32, shape=(None,) + observation_dim) # network input

    with tf.variable_scope('q_net'):
        q_values = policy_net(observation, action_dim)
        train_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) # TODO: better management of TRAINABLE_VARIABLES
    with tf.variable_scope('target_net'):
        q_values_target = policy_net(observation, action_dim)

    # 2. build losses, optimizers
    q_net = policy.DQN(q_values, observation_placeholder=observation) # YongRen: policy.DQN
    target_net = policy.DQN(q_values_target, observation_placeholder=observation)

    action = tf.placeholder(dtype=tf.int32, shape=[None]) # batch of integer actions
    target = tf.placeholder(dtype=tf.float32, shape=[None]) # target value for DQN

    dqn_loss = losses.dqn_loss(action, target, q_net) # TongzhengRen

    total_loss = dqn_loss
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(total_loss, var_list=train_var_list)

    # 3. define data collection
    training_data = Replay(env, q_net, advantage_estimation.qlearning_target(target_net)) #
                                                             # ShihongSong: Replay(env, pi, advantage_estimation.qlearning_target(target_network)), use your ReplayMemory, interact as follows. Simplify your advantage_estimation.dqn to run before YongRen's DQN
    # maybe a dict to manage the elements to be collected

    # 4. start training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        minibatch_count = 0
        collection_count = 0
        while True: # until some stopping criterion met...
            # collect data
            training_data.collect() # ShihongSong
            collection_count += 1
            print('Collected {} times.'.format(collection_count))

            # update network
            data = training_data.next_batch(64) # YouQiaoben, ShihongSong
            # TODO: auto managing of the placeholders? or add this to params of data.Batch
            sess.run(train_op, feed_dict={observation: data['observations'], action: data['actions'], target: data['target']})
            minibatch_count += 1
            print('Trained {} minibatches.'.format(minibatch_count))

            # TODO: assigning pi to pi_old is not implemented yet