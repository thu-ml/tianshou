#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import time
import gym

# our lib imports here!
import sys
sys.path.append('..')
import tianshou.core.losses as losses
from tianshou.data.batch import Batch
import tianshou.data.advantage_estimation as advantage_estimation
import tianshou.core.policy as policy


def policy_net(observation, action_dim, scope=None):
    """
    Constructs the policy network. NOT NEEDED IN THE LIBRARY! this is pure tf

    :param observation: Placeholder for the observation. A tensor of shape (bs, x, y, channels)
    :param action_dim: int. The number of actions.
    :param scope: str. Specifying the scope of the variables.
    """
    # with tf.variable_scope(scope):
    net = tf.layers.conv2d(observation, 16, 8, 4, 'valid', activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 32, 4, 2, 'valid', activation=tf.nn.relu)
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 256, activation=tf.nn.relu)

    act_logits = tf.layers.dense(net, action_dim)

    return act_logits


if __name__ == '__main__': # a clean version with only policy net, no value net
    env = gym.make('PongNoFrameskip-v4')
    observation_dim = env.observation_space.shape
    action_dim = env.action_space.n

    clip_param = 0.2
    num_batches = 2

    # 1. build network with pure tf
    observation = tf.placeholder(tf.float32, shape=(None,) + observation_dim) # network input

    with tf.variable_scope('pi'):
        action_logits = policy_net(observation, action_dim, 'pi')
        train_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) # TODO: better management of TRAINABLE_VARIABLES
    with tf.variable_scope('pi_old'):
        action_logits_old = policy_net(observation, action_dim, 'pi_old')

    # 2. build losses, optimizers
    pi = policy.OnehotCategorical(action_logits, observation_placeholder=observation) # YongRen: policy.Gaussian (could reference the policy in TRPO paper, my code is adapted from zhusuan.distributions) policy.DQN etc.
    # for continuous action space, you may need to change an environment to run
    pi_old = policy.OnehotCategorical(action_logits_old, observation_placeholder=observation)

    action = tf.placeholder(dtype=tf.int32, shape=[None]) # batch of integer actions
    advantage = tf.placeholder(dtype=tf.float32, shape=[None]) # advantage values used in the Gradients

    ppo_loss_clip = losses.ppo_clip(action, advantage, clip_param, pi, pi_old) # TongzhengRen: losses.vpg ... management of placeholders and feed_dict

    total_loss = ppo_loss_clip
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(total_loss, var_list=train_var_list)

    # 3. define data collection
    training_data = Batch(env, pi, advantage_estimation.full_return) # YouQiaoben: finish and polish Batch, advantage_estimation.gae_lambda as in PPO paper
                                                             # ShihongSong: Replay(env, pi, advantage_estimation.target_network), use your ReplayMemory, interact as follows. Simplify your advantage_estimation.dqn to run before YongRen's DQN
    # maybe a dict to manage the elements to be collected

    # 4. start training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        minibatch_count = 0
        collection_count = 0
        while True: # until some stopping criterion met...
            # collect data
            training_data.collect(num_episodes=2) # YouQiaoben, ShihongSong
            collection_count += 1
            print('Collected {} times.'.format(collection_count))

            # update network
            for _ in range(num_batches):
                data = training_data.next_batch(64) # YouQiaoben, ShihongSong
                # TODO: auto managing of the placeholders? or add this to params of data.Batch
                sess.run(train_op, feed_dict={observation: data['observations'], action: data['actions'], advantage: data['returns']})
                minibatch_count += 1
                print('Trained {} minibatches.'.format(minibatch_count))