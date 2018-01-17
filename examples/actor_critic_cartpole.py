#!/usr/bin/env python
from __future__ import absolute_import

import tensorflow as tf
import time
import numpy as np
import gym

# our lib imports here! It's ok to append path in examples
import sys
sys.path.append('..')
from tianshou.core import losses
from tianshou.data.batch import Batch
import tianshou.data.advantage_estimation as advantage_estimation
import tianshou.core.policy.stochastic as policy  # TODO: fix imports as zhusuan so that only need to import to policy
import tianshou.core.value_function.state_value as value_function


# for tutorial purpose, placeholders are explicitly appended with '_ph' suffix

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    observation_dim = env.observation_space.shape
    action_dim = env.action_space.n

    clip_param = 0.2
    num_batches = 10
    batch_size = 128

    seed = 10
    np.random.seed(seed)
    tf.set_random_seed(seed)

    ### 1. build network with pure tf
    observation_ph = tf.placeholder(tf.float32, shape=(None,) + observation_dim)

    def my_network():
        # placeholders defined in this function would be very difficult to manage
        net = tf.layers.dense(observation_ph, 32, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 32, activation=tf.nn.tanh)

        action_logtis = tf.layers.dense(net, action_dim, activation=None)
        value = tf.layers.dense(net, 1, activation=None)

        return action_logtis, value
    # TODO: overriding seems not able to handle shared layers, unless a new class `SharedPolicyValue`
    # maybe the most desired thing is to freely build policy and value function from any tensor?
    # but for now, only the outputs of the network matters

    ### 2. build policy, critic, loss, optimizer
    actor = policy.OnehotCategorical(my_network, observation_placeholder=observation_ph, weight_update=1)
    critic = value_function.StateValue(my_network, observation_placeholder=observation_ph)

    actor_loss = losses.REINFORCE(actor)
    critic_loss = losses.state_value_mse(critic)
    total_loss = actor_loss + critic_loss

    optimizer = tf.train.AdamOptimizer(1e-4)

    # this hack would be unnecessary if we have a `SharedPolicyValue` class, or hack the trainable_variables management
    var_list = list(set(actor.trainable_variables + critic.trainable_variables))

    train_op = optimizer.minimize(total_loss, var_list=var_list)

    ### 3. define data collection
    data_collector = Batch(env, actor,
                           [advantage_estimation.gae_lambda(1, critic), advantage_estimation.nstep_return(1, critic)],
                           [actor, critic])
    # TODO: refactor this, data_collector should be just the top-level abstraction

    ### 4. start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        start_time = time.time()
        for i in range(100):
            # collect data
            data_collector.collect(num_episodes=20)

            # print current return
            print('Epoch {}:'.format(i))
            data_collector.statistics()

            # update network
            for _ in range(num_batches):
                feed_dict = data_collector.next_batch(batch_size)
                sess.run(train_op, feed_dict=feed_dict)

            print('Elapsed time: {:.1f} min'.format((time.time() - start_time) / 60))