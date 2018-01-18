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
import tianshou.core.policy as policy
import tianshou.core.value_function.action_value as value_function
import tianshou.core.opt as opt


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    observation_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    clip_param = 0.2
    num_batches = 10
    batch_size = 512

    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)

    ### 1. build network with pure tf
    observation_ph = tf.placeholder(tf.float32, shape=(None,) + observation_dim)
    action_ph = tf.placeholder(tf.float32, shape=(None,) + action_dim)

    def my_network():
        net = tf.layers.dense(observation_ph, 32, activation=tf.nn.relu)
        net = tf.layers.dense(net, 32, activation=tf.nn.relu)
        action = tf.layers.dense(net, action_dim[0], activation=None)

        action_value_input = tf.concat([observation_ph, action_ph], axis=1)
        net = tf.layers.dense(action_value_input, 32, activation=tf.nn.relu)
        net = tf.layers.dense(net, 32, activation=tf.nn.relu)
        action_value = tf.layers.dense(net, 1, activation=None)

        return action, action_value

    ### 2. build policy, loss, optimizer
    actor = policy.Deterministic(my_network, observation_placeholder=observation_ph, weight_update=1e-3)
    critic = value_function.ActionValue(my_network, observation_placeholder=observation_ph,
                                        action_placeholder=action_ph, weight_update=1e-3)

    critic_loss = losses.value_mse(critic)
    critic_optimizer = tf.train.AdamOptimizer(1e-3)
    critic_train_op = critic_optimizer.minimize(critic_loss, var_list=critic.trainable_variables)

    dpg_grads = opt.DPG(actor, critic)  # not sure if it's correct
    actor_optimizer = tf.train.AdamOptimizer(1e-4)
    actor_train_op = actor_optimizer.apply_gradients(dpg_grads)

    ### 3. define data collection
    data_collector = Batch(env, actor, [advantage_estimation.ddpg_return(actor, critic)], [actor, critic])

    ### 4. start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # assign actor to pi_old
        actor.sync_weights()  # TODO: automate this for policies with target network
        critic.sync_weights()

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
                sess.run([actor_train_op, critic_train_op], feed_dict=feed_dict)

            print('Elapsed time: {:.1f} min'.format((time.time() - start_time) / 60))