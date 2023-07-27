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
):
    if len(train_envs) > 1:
        buffer = VectorReplayBuffer(buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    return test_collector, train_collector


def watch_agent(
    n_episode: int, policy: BasePolicy, test_collector: Collector, render=0.0
):
    policy.eval()
    test_collector.reset()
    result = test_collector.collect(n_episode=n_episode, render=render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
