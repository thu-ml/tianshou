from typing import Any, List, Optional, Tuple, Union

import numpy as np

from tianshou.env.utils import gym_new_venv_step_type
from tianshou.env.venvs import GYM_RESERVED_KEYS, BaseVectorEnv
from tianshou.utils import RunningMeanStd


class VectorEnvWrapper(BaseVectorEnv):
    """Base class for vectorized environments wrapper."""

    def __init__(self, venv: BaseVectorEnv) -> None:
        self.venv = venv
        self.is_async = venv.is_async

    def __len__(self) -> int:
        return len(self.venv)

    def __getattribute__(self, key: str) -> Any:
        if key in GYM_RESERVED_KEYS:  # reserved keys in gym.Env
            return getattr(self.venv, key)
        else:
            return super().__getattribute__(key)

    def get_env_attr(
        self,
        key: str,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> List[Any]:
        return self.venv.get_env_attr(key, id)

    def set_env_attr(
        self,
        key: str,
        value: Any,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> None:
        return self.venv.set_env_attr(key, value, id)

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Union[dict, List[dict]]]:
        return self.venv.reset(id, **kwargs)

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> gym_new_venv_step_type:
        return self.venv.step(action, id)

    def seed(
        self,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> List[Optional[List[int]]]:
        return self.venv.seed(seed)

    def render(self, **kwargs: Any) -> List[Any]:
        return self.venv.render(**kwargs)

    def close(self) -> None:
        self.venv.close()


class VectorEnvNormObs(VectorEnvWrapper):
    """An observation normalization wrapper for vectorized environments.

    :param bool update_obs_rms: whether to update obs_rms. Default to True.
    """

    def __init__(
        self,
        venv: BaseVectorEnv,
        update_obs_rms: bool = True,
    ) -> None:
        super().__init__(venv)
        # initialize observation running mean/std
        self.update_obs_rms = update_obs_rms
        self.obs_rms = RunningMeanStd()

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Union[dict, List[dict]]]:
        obs, info = self.venv.reset(id, **kwargs)

        if isinstance(obs, tuple):  # type: ignore
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )

        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs)
        obs = self._norm_obs(obs)
        return obs, info

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> gym_new_venv_step_type:
        step_results = self.venv.step(action, id)
        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(step_results[0])
        return (self._norm_obs(step_results[0]), *step_results[1:])

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms:
            return self.obs_rms.norm(obs)  # type: ignore
        return obs

    def set_obs_rms(self, obs_rms: RunningMeanStd) -> None:
        """Set with given observation running mean/std."""
        self.obs_rms = obs_rms

    def get_obs_rms(self) -> RunningMeanStd:
        """Return observation running mean/std."""
        return self.obs_rms
