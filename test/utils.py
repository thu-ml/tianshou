from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from tianshou.data.collector import CollectStats


@dataclass
class SpaceInfo:
    action_shape: int | Sequence[int]
    state_shape: int | Sequence[int]
    action_dim: int
    state_dim: int
    min_action: float
    max_action: float


def get_space_info(
    space: spaces.Space,
) -> SpaceInfo:
    if isinstance(space, spaces.Box):
        return SpaceInfo(
            action_shape=space.shape,
            state_shape=space.shape,
            action_dim=space.shape[0],
            state_dim=space.shape[0],
            min_action=float(np.min(space.low)),
            max_action=float(np.max(space.high)),
        )
    elif isinstance(space, spaces.Discrete):
        return SpaceInfo(
            action_shape=int(space.n),
            state_shape=int(space.n),
            action_dim=int(space.n),
            state_dim=int(space.n),
            min_action=float(space.start),
            max_action=float(space.start + space.n - 1),
        )
    else:
        raise NotImplementedError("Unsupported space type")


def print_final_stats(collect_stats: CollectStats) -> None:
    if collect_stats.returns_stat is not None and collect_stats.lens_stat is not None:
        print(
            f"Final reward: {collect_stats.returns_stat.mean}, length: {collect_stats.lens_stat.mean}",
        )
    else:
        print("Final stats rollout not available.")
