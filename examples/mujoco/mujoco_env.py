import logging
import pickle

from tianshou.env import VectorEnvNormObs
from tianshou.highlevel.env import (
    ContinuousEnvironments,
    EnvFactoryRegistered,
    EnvPoolFactory,
    VectorEnvType,
)
from tianshou.highlevel.persistence import Persistence, PersistEvent, RestoreEvent
from tianshou.highlevel.world import World

envpool_is_available = True
try:
    import envpool
except ImportError:
    envpool_is_available = False
    envpool = None

log = logging.getLogger(__name__)


def make_mujoco_env(task: str, seed: int, num_train_envs: int, num_test_envs: int, obs_norm: bool):
    """Wrapper function for Mujoco env.

    If EnvPool is installed, it will automatically switch to EnvPool's Mujoco env.

    :return: a tuple of (single env, training envs, test envs).
    """
    envs = MujocoEnvFactory(task, seed, obs_norm=obs_norm).create_envs(
        num_train_envs,
        num_test_envs,
    )
    return envs.env, envs.train_envs, envs.test_envs


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


class MujocoEnvFactory(EnvFactoryRegistered):
    def __init__(self, task: str, seed: int, obs_norm=True):
        super().__init__(
            task=task,
            seed=seed,
            venv_type=VectorEnvType.SUBPROC_SHARED_MEM,
            envpool_factory=EnvPoolFactory() if envpool_is_available else None,
        )
        self.obs_norm = obs_norm

    def create_envs(self, num_training_envs: int, num_test_envs: int) -> ContinuousEnvironments:
        envs = super().create_envs(num_training_envs, num_test_envs)
        assert isinstance(envs, ContinuousEnvironments)

        # obs norm wrapper
        if self.obs_norm:
            envs.train_envs = VectorEnvNormObs(envs.train_envs)
            envs.test_envs = VectorEnvNormObs(envs.test_envs, update_obs_rms=False)
            envs.test_envs.set_obs_rms(envs.train_envs.get_obs_rms())
            envs.set_persistence(MujocoEnvObsRmsPersistence())

        return envs
