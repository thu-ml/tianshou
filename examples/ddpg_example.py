#!/usr/bin/env python
from __future__ import absolute_import

import tensorflow as tf
import gym
import numpy as np
import time
import argparse
import logging
logging.basicConfig(level=logging.INFO)

# our lib imports here! It's ok to append path in examples
import sys
sys.path.append('..')
from tianshou.core import losses
import tianshou.data.advantage_estimation as advantage_estimation
import tianshou.core.policy as policy
import tianshou.core.value_function.action_value as value_function
import tianshou.core.opt as opt

from tianshou.data.data_buffer.vanilla import VanillaReplayBuffer
from tianshou.data.data_collector import DataCollector
from tianshou.data.tester import test_policy_in_env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", default=False)
    args = parser.parse_args()

    env = gym.make('Pendulum-v0')
    observation_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    batch_size = 32

    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    ### 1. build network with pure tf
    observation_ph = tf.placeholder(tf.float32, shape=(None,) + observation_dim)
    action_ph = tf.placeholder(tf.float32, shape=(None,) + action_dim)

    def my_network():
        net = tf.layers.dense(observation_ph, 16, activation=tf.nn.relu)
        net = tf.layers.dense(net, 16, activation=tf.nn.relu)
        net = tf.layers.dense(net, 16, activation=tf.nn.relu)
        action = tf.layers.dense(net, action_dim[0], activation=None)

        action_value_input = tf.concat([observation_ph, action_ph], axis=1)
        net = tf.layers.dense(action_value_input, 32, activation=tf.nn.relu)
        net = tf.layers.dense(net, 32, activation=tf.nn.relu)
        net = tf.layers.dense(net, 32, activation=tf.nn.relu)
        action_value = tf.layers.dense(net, 1, activation=None)

        return action, action_value

    ### 2. build policy, loss, optimizer
    actor = policy.Deterministic(my_network, observation_placeholder=observation_ph, weight_update=1e-3)
    critic = value_function.ActionValue(my_network, observation_placeholder=observation_ph,
                                        action_placeholder=action_ph, weight_update=1e-3)

    critic_loss = losses.value_mse(critic)
    critic_optimizer = tf.train.AdamOptimizer(1e-3)
    # clip by norm
    critic_grads, vars = zip(*critic_optimizer.compute_gradients(critic_loss, var_list=critic.trainable_variables))
    critic_grads, _ = tf.clip_by_global_norm(critic_grads, 1.0)
    critic_train_op = critic_optimizer.apply_gradients(zip(critic_grads, vars))

    dpg_grads_vars = opt.DPG(actor, critic)  # check which action to use in dpg
    # clip by norm
    dpg_grads, vars = zip(*dpg_grads_vars)
    dpg_grads, _ = tf.clip_by_global_norm(dpg_grads, 1.0)
    actor_optimizer = tf.train.AdamOptimizer(1e-3)
    actor_train_op = actor_optimizer.apply_gradients(zip(dpg_grads, vars))

    ### 3. define data collection
    data_buffer = VanillaReplayBuffer(capacity=100000, nstep=1)

    process_functions = [advantage_estimation.ddpg_return(actor, critic)]

    data_collector = DataCollector(
        env=env,
        policy=actor,
        data_buffer=data_buffer,
        process_functions=process_functions,
        managed_networks=[actor, critic]
    )

    ### 4. start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # assign actor to pi_old
        actor.sync_weights()
        critic.sync_weights()

        start_time = time.time()
        data_collector.collect(num_timesteps=100)  # warm-up
        for i in range(int(1e8)):
            # collect data
            data_collector.collect(num_timesteps=1, episode_cutoff=200)

            # update network
            feed_dict = data_collector.next_batch(batch_size)
            sess.run(critic_train_op, feed_dict=feed_dict)
            sess.run(actor_train_op, feed_dict=feed_dict)

            # update target networks
            actor.sync_weights()
            critic.sync_weights()

            # test every 1000 training steps
            if i % 1000 == 0:
                print('Step {}, elapsed time: {:.1f} min'.format(i, (time.time() - start_time) / 60))
                test_policy_in_env(actor, env, num_episodes=5, episode_cutoff=200)
