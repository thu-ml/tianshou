from __future__ import absolute_import

import gym
import logging


def test_policy_in_env(policy, env, num_episodes, num_timesteps=0):

    assert sum([num_episodes > 0, num_timesteps > 0]) == 1, \
        'One and only one collection number specification permitted!'

    # make another env as the original is for training data collection
    env_id = env.spec.id
    env_ = gym.make(env_id)
    # current_observation = env_.reset()

    # test policy
    if num_episodes > 0:
        pass

    if num_timesteps > 0:
        pass

    # log


    # clear scene
    env_.close()
