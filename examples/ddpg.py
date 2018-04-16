#!/usr/bin/env python
from __future__ import absolute_import

import tensorflow as tf
import gym
import numpy as np
import time
import argparse

import tianshou as ts


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
    env.seed(seed)

    ### 1. build network with pure tf
    observation_ph = tf.placeholder(tf.float32, shape=(None,) + observation_dim)
    action_ph = tf.placeholder(tf.float32, shape=(None,) + action_dim)

    def my_network():
        net = tf.layers.dense(observation_ph, 32, activation=tf.nn.relu)
        net = tf.layers.dense(net, 32, activation=tf.nn.relu)
        action = tf.layers.dense(net, action_dim[0], activation=None)

        action_value_input = tf.concat([observation_ph, action_ph], axis=1)
        net = tf.layers.dense(action_value_input, 64, activation=tf.nn.relu)
        net = tf.layers.dense(net, 64, activation=tf.nn.relu)
        action_value = tf.layers.dense(net, 1, activation=None)

        return action, action_value

    ### 2. build policy, loss, optimizer
    actor = ts.policy.Deterministic(my_network, observation_placeholder=observation_ph,
                                 has_old_net=True)
    critic = ts.value_function.ActionValue(my_network, observation_placeholder=observation_ph,
                                        action_placeholder=action_ph, has_old_net=True)
    soft_update_op = ts.get_soft_update_op(1e-2, [actor, critic])

    critic_loss = ts.losses.value_mse(critic)
    critic_optimizer = tf.train.AdamOptimizer(1e-3)
    critic_train_op = critic_optimizer.minimize(critic_loss, var_list=list(critic.trainable_variables))

    dpg_grads_vars = ts.opt.DPG(actor, critic)
    actor_optimizer = tf.train.AdamOptimizer(1e-3)
    actor_train_op = actor_optimizer.apply_gradients(dpg_grads_vars)

    ### 3. define data collection
    data_buffer = ts.data.VanillaReplayBuffer(capacity=10000, nstep=1)

    process_functions = [ts.data.advantage_estimation.ddpg_return(actor, critic)]

    data_collector = ts.data.DataCollector(
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
        data_collector.collect(num_timesteps=5000)  # warm-up
        for i in range(int(1e8)):
            # collect data
            data_collector.collect(num_timesteps=1, episode_cutoff=200)

            # train critic
            feed_dict = data_collector.next_batch(batch_size)
            sess.run(critic_train_op, feed_dict=feed_dict)

            # recompute action
            data_collector.denoise_action(feed_dict)

            # train actor
            sess.run(actor_train_op, feed_dict=feed_dict)

            # update target networks
            sess.run(soft_update_op)

            # test every 1000 training steps
            if i % 1000 == 0:
                print('Step {}, elapsed time: {:.1f} min'.format(i, (time.time() - start_time) / 60))
                ts.data.test_policy_in_env(actor, env, num_episodes=5, episode_cutoff=200)
