from collections.abc import Sequence

import numpy as np
from gymnasium import spaces


def get_spaces_info(
    act_space: spaces.Space,
    obs_space: spaces.Space,
) -> tuple:
    action_shape, state_shape, action_dim, state_dim, min_action, max_action = (
        None,
        None,
        None,
        None,
        None,
        None,
    )

    action_space: spaces.Box | spaces.Discrete
    observation_space: spaces.Discrete | spaces.Box
    state_shape: int | Sequence[int]
    action_shape: int | Sequence[int]

    if isinstance(act_space, spaces.Box | spaces.Discrete):
        action_space = act_space
    else:
        raise NotImplementedError("Observation space is not of type `Box` or `Discrete`.")

    if isinstance(obs_space, spaces.Box | spaces.Discrete):
        observation_space = obs_space
    else:
        raise NotImplementedError("Observation space is not of type `Box` or `Discrete`.")

    if isinstance(observation_space, spaces.Discrete):
        state_shape = int(observation_space.n)
        state_dim = state_shape
    elif isinstance(observation_space, spaces.Box):
        state_shape = observation_space.shape
        state_dim = state_shape[0]
    else:
        raise NotImplementedError("Observation space is not of type `Box` or `Discrete`.")

    if isinstance(action_space, spaces.Box):
        action_shape = action_space.shape
        max_action = float(np.max(action_space.high))
        min_action = float(np.min(action_space.low))
        action_dim = action_shape[0]
    elif isinstance(action_space, spaces.Discrete):
        action_shape = int(action_space.n)
        max_action = float(action_space.start + action_space.n - 1)
        min_action = float(action_space.start)
        action_dim = action_shape
    else:
        raise NotImplementedError("Action space is not of type `Box`.")

    return (
        action_space,
        observation_space,
        action_shape,
        state_shape,
        action_dim,
        state_dim,
        min_action,
        max_action,
    )
