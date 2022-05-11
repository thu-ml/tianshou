"""Env package."""

from tianshou.env.venv_wrappers import VectorEnvNormObs, VectorEnvWrapper
from tianshou.env.venvs import (
    BaseVectorEnv,
    DummyVectorEnv,
    RayVectorEnv,
    ShmemVectorEnv,
    SubprocVectorEnv,
)

try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    pass

__all__ = [
    "BaseVectorEnv",
    "DummyVectorEnv",
    "SubprocVectorEnv",
    "ShmemVectorEnv",
    "RayVectorEnv",
    "VectorEnvWrapper",
    "VectorEnvNormObs",
    "PettingZooEnv",
]
