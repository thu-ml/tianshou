#!/usr/bin/env python

import tensorflow as tf, numpy as np
import time
import gym

# our lib imports here!
import sys
sys.path.append('..')
import tianshou.core.losses as losses
from tianshou.data.Batch import Batch
import tianshou.data.adv_estimate as adv_estimate
import tianshou.core.policy as policy


def policy_net(obs, act_dim, scope=None):
    """
    Constructs the policy network. NOT NEEDED IN THE LIBRARY! this is pure tf

    :param obs: Placeholder for the observation. A tensor of shape (bs, x, y, channels)
    :param act_dim: int. The number of actions.
    :param scope: str. Specifying the scope of the variables.
    """
    # with tf.variable_scope(scope):
    net = tf.layers.conv2d(obs, 16, 8, 4, 'valid', activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 32, 4, 2, 'valid', activation=tf.nn.relu)
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 256, activation=tf.nn.relu)

    act_logits = tf.layers.dense(net, act_dim)

    return act_logits


if __name__ == '__main__': # a clean version with only policy net, no value net
    env = gym.make('PongNoFrameskip-v4')
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    clip_param = 0.2
    nb_batches = 2

    # 1. build network with pure tf
    obs = tf.placeholder(tf.float32, shape=(None,) + obs_dim) # network input

    with tf.variable_scope('pi'):
        act_logits = policy_net(obs, act_dim, 'pi')
        train_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) # TODO: better management of TRAINABLE_VARIABLES
    with tf.variable_scope('pi_old'):
        act_logits_old = policy_net(obs, act_dim, 'pi_old')

    # 2. build losses, optimizers
    pi = policy.OnehotCategorical(act_logits, obs_placeholder=obs) # YongRen: policy.Gaussian (could reference the policy in TRPO paper, my code is adapted from zhusuan.distributions) policy.DQN etc.
    # for continuous action space, you may need to change an environment to run
    pi_old = policy.OnehotCategorical(act_logits_old, obs_placeholder=obs)

    act = tf.placeholder(dtype=tf.int32, shape=[None]) # batch of integer actions
    Dgrad = tf.placeholder(dtype=tf.float32, shape=[None]) # values used in the Gradients

    ppo_loss_clip = losses.ppo_clip(act, Dgrad, clip_param, pi, pi_old) # TongzhengRen: losses.vpg ... management of placeholders and feed_dict

    total_loss = ppo_loss_clip
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(total_loss, var_list=train_var_list)

    # 3. define data collection
    training_data = Batch(env, pi, adv_estimate.full_return) # YouQiaoben: finish and polish Batch, adv_estimate.gae_lambda as in PPO paper
                                                             # ShihongSong: Replay(env, pi, adv_estimate.target_network), use your ReplayMemory, interact as follows. Simplify your adv_estimate.dqn to run before YongRen's DQN
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
            for _ in range(nb_batches):
                data = training_data.next_batch(64) # YouQiaoben, ShihongSong
                # TODO: auto managing of the placeholders? or add this to params of data.Batch
                sess.run(train_op, feed_dict={obs: data['obs'], act: data['acs'], Dgrad: data['Gts']})
                minibatch_count += 1
                print('Trained {} minibatches.'.format(minibatch_count))