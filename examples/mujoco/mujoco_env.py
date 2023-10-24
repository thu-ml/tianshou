import logging
import pickle
import warnings

import gymnasium as gym

from tianshou.env import ShmemVectorEnv, VectorEnvNormObs
from tianshou.highlevel.env import ContinuousEnvironments, EnvFactory
from tianshou.highlevel.persistence import Persistence, PersistEvent, RestoreEvent
from tianshou.highlevel.world import World

try:
    import envpool
except ImportError:
    envpool = None

log = logging.getLogger(__name__)


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


class MujocoEnvObsRmsPersistence(Persistence):
    FILENAME = "env_obs_rms.pkl"

    def persist(self, event: PersistEvent, world: World) -> None:
        if event != PersistEvent.PERSIST_POLICY:
            return
        obs_rms = world.envs.train_envs.get_obs_rms()
        path = world.persist_path(self.FILENAME)
        log.info(f"Saving environment obs_rms value to {path}")
        with open(path, "wb") as f:
            pickle.dump(obs_rms, f)

    def restore(self, event: RestoreEvent, world: World):
        if event != RestoreEvent.RESTORE_POLICY:
            return
        path = world.restore_path(self.FILENAME)
        log.info(f"Restoring environment obs_rms value from {path}")
        with open(path, "rb") as f:
            obs_rms = pickle.load(f)
        world.envs.train_envs.set_obs_rms(obs_rms)
        world.envs.test_envs.set_obs_rms(obs_rms)


class MujocoEnvFactory(EnvFactory):
    def __init__(self, task: str, seed: int, obs_norm=True):
        self.task = task
        self.seed = seed
        self.obs_norm = obs_norm

    def create_envs(self, num_training_envs: int, num_test_envs: int) -> ContinuousEnvironments:
        env, train_envs, test_envs = make_mujoco_env(
            task=self.task,
            seed=self.seed,
            num_train_envs=num_training_envs,
            num_test_envs=num_test_envs,
            obs_norm=self.obs_norm,
        )
        envs = ContinuousEnvironments(env=env, train_envs=train_envs, test_envs=test_envs)
        if self.obs_norm:
            envs.set_persistence(MujocoEnvObsRmsPersistence())
        return envs
