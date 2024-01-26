from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from gymnasium import spaces


@dataclass(kw_only=True)
class ActionSpaceInfo:
    action_shape: int | Sequence[int]
    min_action: float
    max_action: float

    @property
    def action_dim(self) -> int:
        if isinstance(self.action_shape, int):
            return self.action_shape
        elif isinstance(self.action_shape, Sequence) and self.action_shape:
            return int(np.prod(self.action_shape))
        else:
            raise ValueError("Invalid action_shape: {self.action_shape}.")

    @classmethod
    def from_space(cls, space: spaces.Space) -> Self:
        if isinstance(space, spaces.Box):
            return cls(
                action_shape=space.shape,
                min_action=float(np.min(space.low)),
                max_action=float(np.max(space.high)),
            )
        elif isinstance(space, spaces.Discrete):
            return cls(
                action_shape=int(space.n),
                min_action=float(space.start),
                max_action=float(space.start + space.n - 1),
            )
        else:
            raise ValueError(
                f"Unsupported space type: {space.__class__}. Currently supported types are Discrete and Box.",
            )


@dataclass(kw_only=True)
class ObservationSpaceInfo:
    obs_shape: int | Sequence[int]

    @property
    def obs_dim(self) -> int:
        if isinstance(self.obs_shape, int):
            return self.obs_shape
        elif isinstance(self.obs_shape, Sequence) and self.obs_shape:
            return int(np.prod(self.obs_shape))
        else:
            raise ValueError("Invalid obs_shape: {self.obs_shape}.")

    @classmethod
    def from_space(cls, space: spaces.Space) -> Self:
        if isinstance(space, spaces.Box):
            return cls(
                obs_shape=space.shape,
            )
        elif isinstance(space, spaces.Discrete):
            return cls(
                obs_shape=int(space.n),
            )
        else:
            raise ValueError(
                f"Unsupported space type: {space.__class__}. Currently supported types are Discrete and Box.",
            )


@dataclass(kw_only=True)
class SpaceInfo:
    action_info: ActionSpaceInfo
    observation_info: ObservationSpaceInfo

    @classmethod
    def from_env(cls, action_space: spaces.Space, observation_space: spaces.Space) -> Self:
        action_info = ActionSpaceInfo.from_space(action_space)
        observation_info = ObservationSpaceInfo.from_space(observation_space)

        return cls(
            action_info=action_info,
            observation_info=observation_info,
        )
