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

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize


# for tutorial purpose, placeholders are explicitly appended with '_ph' suffix
# this example with batch_norm and dropout almost surely cannot improve. it just shows how to use those
# layers and another way of writing networks.

class MyPolicy(object):
    def __init__(self, observation_ph, is_training_ph, keep_prob_ph, action_dim):
        self.observation_ph = observation_ph
        self.is_training_ph = is_training_ph
        self.keep_prob_ph = keep_prob_ph
        self.action_dim = action_dim

    def __call__(self):
        net = tf.layers.dense(self.observation_ph, 32, activation=None)
        net = tf.layers.batch_normalization(net, training=self.is_training_ph)
        net = tf.nn.relu(net)
        net = tf.nn.dropout(net, keep_prob=self.keep_prob_ph)

        net = tf.layers.dense(net, 32, activation=tf.nn.relu)
        net = tf.layers.dropout(net, rate=1 - self.keep_prob_ph)
        action_mean = tf.layers.dense(net, action_dim, activation=None)
        action_logstd = tf.get_variable('action_logstd', shape=(self.action_dim,), dtype=tf.float32)

        return action_mean, action_logstd, None


if __name__ == '__main__':
    env = normalize(CartpoleEnv())
    observation_dim = env.observation_space.shape
    action_dim = env.action_space.flat_dim

    # clip_param = 0.2
    num_batches = 10
    batch_size = 128

    seed = 10
    np.random.seed(seed)
    tf.set_random_seed(seed)

    ### 1. build network with pure tf
    observation_ph = tf.placeholder(tf.float32, shape=(None,) + observation_dim)
    is_training_ph = tf.placeholder(tf.bool, shape=())
    keep_prob_ph = tf.placeholder(tf.float32, shape=())

    my_policy = MyPolicy(observation_ph, is_training_ph, keep_prob_ph, action_dim)

    ### 2. build policy, loss, optimizer
    pi = policy.Normal(my_policy, observation_placeholder=observation_ph, weight_update=0)

    clip_param = tf.placeholder(tf.float32, shape=(), name='ppo_loss_clip_param')
    ppo_loss_clip = losses.ppo_clip(pi, clip_param)

    total_loss = ppo_loss_clip
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(total_loss, var_list=pi.trainable_variables)

    ### 3. define data collection
    training_data = Batch(env, pi, [advantage_estimation.full_return], [pi])

    ### 4. start training
    feed_dict_train = {is_training_ph: True, keep_prob_ph: 0.8}
    feed_dict_test = {is_training_ph: False, keep_prob_ph: 1}

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # assign actor to pi_old
        pi.sync_weights()  # TODO: automate this for policies with target network

        start_time = time.time()
        for i in range(100):
            # collect data
            training_data.collect(num_episodes=20, my_feed_dict=feed_dict_train)

            # print current return
            print('Epoch {}:'.format(i))
            training_data.statistics()

            # manipulate decay_param
            if i < 30:
                feed_dict_train[clip_param] = 0.2
            else:
                feed_dict_train[clip_param] = 0.1

            # update network
            for _ in range(num_batches):
                feed_dict = training_data.next_batch(batch_size)
                feed_dict.update(feed_dict_train)
                sess.run(train_op, feed_dict=feed_dict)

            # assigning actor to pi_old
            pi.update_weights()

            # approximate test mode
            training_data.collect(num_episodes=10, my_feed_dict=feed_dict_test)
            print('After training:')
            training_data.statistics()

            print('Elapsed time: {:.1f} min'.format((time.time() - start_time) / 60))