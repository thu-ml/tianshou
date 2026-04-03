"""Backward compatibility shim.

The mujoco environment helpers have been moved to ``tianshou.env.mujoco``.
This module re-exports them so that existing code using
``from examples.mujoco.mujoco_env import ...`` or
``from mujoco_env import ...`` continues to work.
"""

from tianshou.env.mujoco import MujocoEnvFactory, MujocoEnvObsRmsPersistence, make_mujoco_env

__all__ = [
    "MujocoEnvFactory",
    "MujocoEnvObsRmsPersistence",
    "make_mujoco_env",
]
