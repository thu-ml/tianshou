from __future__ import absolute_import

import gym
import logging
import numpy as np

__all__ = [
    'test_policy_in_env',
]


def test_policy_in_env(policy, env, num_timesteps=0, num_episodes=0,
                       discount_factor=0.99, seed=0, episode_cutoff=None):
    """
    Tests the policy in the environment and record and prints out the performance. This is useful when the policy
    is trained with off-policy algorithms and thus the rewards in the data buffer does not reflect the
    performance of the current policy.

    :param policy: A :class:`tianshou.core.policy`. The current policy being optimized.
    :param env: An environment.
    :param num_timesteps: An int specifying the number of timesteps to test the policy.
        It defaults to 0 and either
        ``num_timesteps`` or ``num_episodes`` could be set but not both.
    :param num_episodes: An int specifying the number of episodes to test the policy.
        It defaults to 0 and either
        ``num_timesteps`` or ``num_episodes`` could be set but not both.
    :param discount_factor: Optional. A float in range :math:`[0, 1]` defaulting to 0.99. The discount
        factor to compute discounted returns.
    :param seed: An non-negative int. The seed to seed the environment as ``env.seed(seed)``.
    :param episode_cutoff: Optional. An int. The maximum number of timesteps in one episode. This is
            useful when the environment has no terminal states or a single episode could be prohibitively long.
            If set than all episodes are forced to stop beyond this number to timesteps.
    """
    assert sum([num_episodes > 0, num_timesteps > 0]) == 1, \
        'One and only one collection number specification permitted!'
    assert seed >= 0

    # make another env as the original is for training data collection
    env_id = env.spec.id
    env_ = gym.make(env_id)
    env.seed(seed)

    # test policy
    returns = []
    undiscounted_returns = []
    current_return = 0.
    current_undiscounted_return = 0.

    if num_episodes > 0:
        returns = [0.] * num_episodes
        undiscounted_returns = [0.] * num_episodes
        for i in range(num_episodes):
            current_return = 0.
            current_undiscounted_return = 0.
            current_discount = 1.
            observation = env_.reset()
            done = False
            step_count = 0
            while not done:
                action = policy.act_test(observation)
                observation, reward, done, _ = env_.step(action)
                current_return += reward * current_discount
                current_undiscounted_return += reward
                current_discount *= discount_factor
                step_count += 1
                if episode_cutoff and step_count >= episode_cutoff:
                    break

            returns[i] = current_return
            undiscounted_returns[i] = current_undiscounted_return

    # run for fix number of timesteps, only the first episode and finished episodes
    # matters when calcuting average return
    if num_timesteps > 0:
        current_discount = 1.
        observation = env_.reset()
        step_count_this_episode = 0
        for _ in range(num_timesteps):
            action = policy.act_test(observation)
            observation, reward, done, _ = env_.step(action)
            current_return += reward * current_discount
            current_undiscounted_return += reward
            current_discount *= discount_factor
            step_count_this_episode += 1
            if episode_cutoff and step_count_this_episode >= episode_cutoff:
                done = True

            if done:
                returns.append(current_return)
                undiscounted_returns.append(current_undiscounted_return)
                current_return = 0.
                current_undiscounted_return = 0.
                current_discount = 1.
                observation = env_.reset()
                step_count_this_episode = 0

    # log
    if returns:  # has at least one finished episode
        mean_return = np.mean(returns)
        mean_undiscounted_return = np.mean(undiscounted_returns)
    else:  # the first episode is too long to finish
        logging.warning('The first test episode is still not finished after {} timesteps. '
                        'Logging its return anyway.'.format(num_timesteps))
        mean_return = current_return
        mean_undiscounted_return = current_undiscounted_return
    print('Mean return: {}'.format(mean_return))
    print('Mean undiscounted return: {}'.format(mean_undiscounted_return))

    # clear scene
    env_.close()
