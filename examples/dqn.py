import tensorflow as tf
import gym
import numpy as np
import time

import tianshou as ts


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    observation_dim = env.observation_space.shape
    action_dim = env.action_space.n

    # hyper-parameters
    batch_size = 32

    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    ### 1. build network with pure tf
    observation_ph = tf.placeholder(tf.float32, shape=(None,) + observation_dim)

    def my_network():
        net = tf.layers.dense(observation_ph, 32, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 32, activation=tf.nn.tanh)

        action_values = tf.layers.dense(net, action_dim, activation=None)

        return None, action_values  # no policy head

    ### 2. build policy, loss, optimizer
    dqn = ts.value_function.DQN(my_network, observation_placeholder=observation_ph, has_old_net=True)
    pi = ts.policy.DQN(dqn)

    dqn_loss = ts.losses.value_mse(dqn)

    total_loss = dqn_loss
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(total_loss, var_list=list(dqn.trainable_variables))

    ### 3. define data collection
    replay_buffer = ts.data.VanillaReplayBuffer(capacity=2e4, nstep=1)

    process_functions = [ts.data.advantage_estimation.nstep_q_return(1, dqn)]
    managed_networks = [dqn]

    data_collector = ts.data.DataCollector(
        env=env,
        policy=pi,
        data_buffer=replay_buffer,
        process_functions=process_functions,
        managed_networks=managed_networks
    )

    ### 4. start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # sync target network in the beginning
        pi.sync_weights()

        start_time = time.time()
        data_collector.collect(num_timesteps=5000)
        for i in range(int(1e8)):  # number of training steps
            # collect data
            data_collector.collect(num_timesteps=4)

            # update network
            feed_dict = data_collector.next_batch(batch_size)
            sess.run(train_op, feed_dict=feed_dict)

            if i % 5000 == 0:
                print('Step {}, elapsed time: {:.1f} min'.format(i, (time.time() - start_time) / 60))
                # epsilon 0.05 as in nature paper
                pi.set_epsilon_test(0.05)
                ts.data.test_policy_in_env(pi, env, num_timesteps=1000)

            # update target network
            if i % 1000 == 0:
                pi.sync_weights()