from typing import Tuple

import gymnasium as gym

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import VectorEnvNormObs
from tianshou.policy import BasePolicy


def get_continuous_env_info(
    env: gym.Env,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], float]:
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError(
            "Only environments with continuous action space are supported here. "
            f"But got env with action space: {env.action_space.__class__}."
        )
    state_shape = env.observation_space.shape or env.observation_space.n
    if not state_shape:
        raise ValueError("Observation space shape is not defined")
    action_shape = env.action_space.shape
    max_action = env.action_space.high[0]
    return state_shape, action_shape, max_action


def get_train_test_collector(
    buffer_size: int,
    policy: BasePolicy,
    train_envs: VectorEnvNormObs,
    test_envs: VectorEnvNormObs,
    start_timesteps: int = 0,
    start_timesteps_random: bool = True,
):
    """Create train and test collectors for the given policy and environments.

    :param buffer_size: size of the replay buffer
    :param policy: policy to use
    :param train_envs: training environments
    :param test_envs: testing environments
    :param start_timesteps: number of steps to collect before training.
        Mainly useful for off-policy algorithms.
    :param start_timesteps_random: if True, collect the initial steps randomly
        (i.e. without using the policy). Otherwise, use the policy.
        Only relevant if start_timesteps > 0.
    :return: train and test collectors
    """
    if len(train_envs) > 1:
        buffer = VectorReplayBuffer(buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    if start_timesteps > 0:
        train_collector.collect(n_step=start_timesteps, random=start_timesteps_random)
    return train_collector, test_collector


def watch_agent(
    n_episode: int, policy: BasePolicy, test_collector: Collector, render=0.0
):
    policy.eval()
    test_collector.reset()
    result = test_collector.collect(n_episode=n_episode, render=render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
