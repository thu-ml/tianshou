from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from gymnasium import spaces


@dataclass(kw_only=True)
class ActionSpaceInfo:
    action_shape: int | Sequence[int]
    action_dim: int = field(init=False)
    min_action: float
    max_action: float

    def __post_init__(self) -> None:
        if isinstance(self.action_shape, int):
            self.action_dim = self.action_shape
        else:
            self.action_dim = int(self.action_shape[0])

    @classmethod
    def from_space(cls, space: spaces.Space) -> "ActionSpaceInfo":
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
    obs_dim: int = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.obs_shape, int):
            self.obs_dim = self.obs_shape
        else:
            self.obs_dim = int(self.obs_shape[0])

    @classmethod
    def from_space(cls, space: spaces.Space) -> "ObservationSpaceInfo":
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
    def from_env(cls, action_space: spaces.Space, observation_space: spaces.Space) -> "SpaceInfo":
        action_info = ActionSpaceInfo.from_space(action_space)
        observation_info = ObservationSpaceInfo.from_space(observation_space)

        return cls(
            action_info=action_info,
            observation_info=observation_info,
        )
