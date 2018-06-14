import tensorflow as tf
import gym
import numpy as np
import time

import tianshou as ts


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    observation_dim = env.observation_space.shape
    action_dim = env.action_space.n

    clip_param = 0.2
    num_batches = 10
    batch_size = 512

    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)

    ### 1. build network with pure tf
    observation_ph = tf.placeholder(tf.float32, shape=(None,) + observation_dim)

    def my_policy():
        net = tf.layers.dense(observation_ph, 32, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 32, activation=tf.nn.tanh)

        action_logits = tf.layers.dense(net, action_dim, activation=None)
        action_dist = tf.distributions.Categorical(logits=action_logits)

        return action_dist, None

    ### 2. build policy, loss, optimizer
    pi = ts.policy.Distributional(my_policy, observation_placeholder=observation_ph, has_old_net=True)

    ppo_loss_clip = ts.losses.ppo_clip(pi, clip_param)

    total_loss = ppo_loss_clip
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(total_loss, var_list=list(pi.trainable_variables))

    ### 3. define data collection
    data_buffer = ts.data.BatchSet()

    data_collector = ts.data.DataCollector(
        env=env,
        policy=pi,
        data_buffer=data_buffer,
        process_functions=[ts.data.advantage_estimation.full_return],
        managed_networks=[pi],
    )

    ### 4. start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # assign actor to pi_old
        pi.sync_weights()

        start_time = time.time()
        for i in range(1000):
            # collect data
            data_collector.collect(num_episodes=50)

            # print current return
            print('Epoch {}:'.format(i))
            data_buffer.statistics()

            # update network
            for _ in range(num_batches):
                feed_dict = data_collector.next_batch(batch_size)
                sess.run(train_op, feed_dict=feed_dict)

            # assigning pi_old to be current pi
            pi.sync_weights()

            print('Elapsed time: {:.1f} min'.format((time.time() - start_time) / 60))