import warnings

import gymnasium as gym

from tianshou.env import ShmemVectorEnv, VectorEnvNormObs
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import ContinuousEnvironments, EnvFactory

try:
    import envpool
except ImportError:
    envpool = None


def make_mujoco_env(task: str, seed: int, num_train_envs: int, num_test_envs: int, obs_norm: bool):
    """Wrapper function for Mujoco env.

    If EnvPool is installed, it will automatically switch to EnvPool's Mujoco env.

    :return: a tuple of (single env, training envs, test envs).
    """
    if envpool is not None:
        train_envs = env = envpool.make_gymnasium(task, num_envs=num_train_envs, seed=seed)
        test_envs = envpool.make_gymnasium(task, num_envs=num_test_envs, seed=seed)
    else:
        warnings.warn(
            "Recommend using envpool (pip install envpool) "
            "to run Mujoco environments more efficiently.",
        )
        env = gym.make(task)
        train_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(num_train_envs)])
        test_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)])
        train_envs.seed(seed)
        test_envs.seed(seed)
    if obs_norm:
        # obs norm wrapper
        train_envs = VectorEnvNormObs(train_envs)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
    return env, train_envs, test_envs


class MujocoEnvFactory(EnvFactory):
    def __init__(self, task: str, seed: int, sampling_config: SamplingConfig):
        self.task = task
        self.sampling_config = sampling_config
        self.seed = seed

    def create_envs(self, config=None) -> ContinuousEnvironments:
        env, train_envs, test_envs = make_mujoco_env(
            task=self.task,
            seed=self.seed,
            num_train_envs=self.sampling_config.num_train_envs,
            num_test_envs=self.sampling_config.num_test_envs,
            obs_norm=True,
        )
        return ContinuousEnvironments(env=env, train_envs=train_envs, test_envs=test_envs)
