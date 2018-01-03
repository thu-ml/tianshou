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
from tianshou.data.batch import Batch
import tianshou.data.advantage_estimation as advantage_estimation
import tianshou.core.policy.stochastic as policy  # TODO: fix imports as zhusuan so that only need to import to policy

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize


def policy_net(observation, action_dim, scope=None):
    """
    Constructs the policy network. NOT NEEDED IN THE LIBRARY! this is pure tf

    :param observation: Placeholder for the observation. A tensor of shape (bs, x, y, channels)
    :param action_dim: int. The number of actions.
    :param scope: str. Specifying the scope of the variables.
    """
    # with tf.variable_scope(scope):
    net = tf.layers.dense(observation, 32, activation=tf.nn.tanh)
    net = tf.layers.dense(net, 32, activation=tf.nn.tanh)

    act_logits = tf.layers.dense(net, action_dim, activation=None)

    return act_logits


if __name__ == '__main__': # a clean version with only policy net, no value net
    env = gym.make('CartPole-v0')
    observation_dim = env.observation_space.shape
    action_dim = env.action_space.n

    clip_param = 0.2
    num_batches = 10
    batch_size = 512

    seed = 10
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # 1. build network with pure tf
    observation = tf.placeholder(tf.float32, shape=(None,) + observation_dim) # network input

    with tf.variable_scope('pi'):
        action_logits = policy_net(observation, action_dim, 'pi')
        train_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) # TODO: better management of TRAINABLE_VARIABLES
    with tf.variable_scope('pi_old'):
        action_logits_old = policy_net(observation, action_dim, 'pi_old')
        pi_old_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'pi_old')

    # 2. build losses, optimizers
    pi = policy.OnehotCategorical(action_logits, observation_placeholder=observation) # YongRen: policy.Gaussian (could reference the policy in TRPO paper, my code is adapted from zhusuan.distributions) policy.DQN etc.
    # for continuous action space, you may need to change an environment to run
    pi_old = policy.OnehotCategorical(action_logits_old, observation_placeholder=observation)

    action = tf.placeholder(dtype=tf.int32, shape=(None,)) # batch of integer actions
    advantage = tf.placeholder(dtype=tf.float32, shape=(None,)) # advantage values used in the Gradients

    ppo_loss_clip = losses.ppo_clip(action, advantage, clip_param, pi, pi_old) # TongzhengRen: losses.vpg ... management of placeholders and feed_dict

    total_loss = ppo_loss_clip
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(total_loss, var_list=train_var_list)

    # 3. define data collection
    training_data = Batch(env, pi, advantage_estimation.full_return) # YouQiaoben: finish and polish Batch, advantage_estimation.gae_lambda as in PPO paper
                                                             # ShihongSong: Replay(), see dqn_example.py
    # maybe a dict to manage the elements to be collected

    # 4. start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # sync pi and pi_old
        sess.run([tf.assign(theta_old, theta) for (theta_old, theta) in zip(pi_old_var_list, train_var_list)])

        start_time = time.time()
        for i in range(100): # until some stopping criterion met...
            # collect data
            training_data.collect(num_episodes=50) # YouQiaoben, ShihongSong

            # print current return
            print('Epoch {}:'.format(i))
            training_data.statistics()

            # update network
            for _ in range(num_batches):
                data = training_data.next_batch(batch_size)  # YouQiaoben, ShihongSong
                # TODO: auto managing of the placeholders? or add this to params of data.Batch
                sess.run(train_op, feed_dict={observation: data['observations'], action: data['actions'],
                                              advantage: data['returns']})

            # assigning pi to pi_old
            sess.run([tf.assign(theta_old, theta) for (theta_old, theta) in zip(pi_old_var_list, train_var_list)])

            print('Elapsed time: {:.1f} min'.format((time.time() - start_time) / 60))