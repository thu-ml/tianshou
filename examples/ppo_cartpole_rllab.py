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

    def my_policy():
        net = tf.layers.dense(observation_ph, 32, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 32, activation=tf.nn.tanh)

        action_mean = tf.layers.dense(net, action_dim, activation=None)
        action_logstd = tf.get_variable('action_logstd', shape=(action_dim, ))
        # value = tf.layers.dense(net, 1, activation=None)

        return action_mean, action_logstd, None  # None value head

    # TODO: current implementation of passing function or overriding function has to return a value head
    # to allow network sharing between policy and value networks. This makes 'policy' and 'value_function'
    # imbalanced semantically (though they are naturally imbalanced since 'policy' is required to interact
    # with the environment and 'value_function' is not). I have an idea to solve this imbalance, which is
    # not based on passing function or overriding function.

    ### 2. build policy, loss, optimizer
    pi = policy.Normal(my_policy, observation_placeholder=observation_ph, weight_update=0)

    ppo_loss_clip = losses.ppo_clip(pi, clip_param)

    total_loss = ppo_loss_clip
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(total_loss, var_list=pi.trainable_variables)

    ### 3. define data collection
    training_data = Batch(env, pi, [advantage_estimation.full_return], [pi])

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
            training_data.collect(num_episodes=20)

            # print current return
            print('Epoch {}:'.format(i))
            training_data.statistics()

            # update network
            for _ in range(num_batches):
                feed_dict = training_data.next_batch(batch_size)
                sess.run(train_op, feed_dict=feed_dict)

            # assigning actor to pi_old
            pi.update_weights()

            print('Elapsed time: {:.1f} min'.format((time.time() - start_time) / 60))