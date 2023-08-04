import gymnasium as gym
import warnings
from typing import Literal, Optional, Tuple

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv, VectorEnvNormObs
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


def make_mujoco_env(
    task: str,
    seed: int,
    num_train_envs: int,
    num_test_envs: int,
    obs_norm: bool,
    render_mode: Optional[Literal["human", "rgb_array"]] = None,
):
    """Wrapper function for Mujoco env.

    If EnvPool is installed, it will automatically switch to EnvPool's Mujoco env.

    :return: a tuple of (single env, training envs, test envs).
    """
    try:
        import envpool

        train_envs = env = envpool.make_gymnasium(
            task, num_envs=num_train_envs, seed=seed
        )
        test_envs = envpool.make_gymnasium(task, num_envs=num_test_envs, seed=seed)
    except ImportError:
        warnings.warn(
            "Recommend using envpool (pip install envpool) "
            "to run Mujoco environments more efficiently."
        )
        env = gym.make(task, render_mode=render_mode)
        train_envs = ShmemVectorEnv(
            [lambda: gym.make(task) for _ in range(num_train_envs)]
        )
        test_envs = ShmemVectorEnv(
            [
                lambda: gym.make(task, render_mode=render_mode)
                for _ in range(num_test_envs)
            ]
        )
        # Note: env.seed() has been removed in gymnasium>0.26
        # https://gymnasium.farama.org/content/migration-guide/#seed-and-random-number-generator
        # seeding through numpy is sufficient for mujoco
        train_envs.seed(seed)
        test_envs.seed(seed)
    if obs_norm:
        # obs norm wrapper
        train_envs = VectorEnvNormObs(train_envs)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
    return env, train_envs, test_envs
