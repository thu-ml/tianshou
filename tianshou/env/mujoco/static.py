import numpy as np


def halfcheetah_terminal_fn(
    obs: np.ndarray,
    act: np.ndarray,
    next_obs: np.ndarray,
) -> np.ndarray:
    """Terminal condition function for HalfCheetah."""
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    return done


def hopper_terminal_fn(
    obs: np.ndarray,
    act: np.ndarray,
    next_obs: np.ndarray,
) -> np.ndarray:
    """Terminal condition function for Hopper."""
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = \
        np.isfinite(next_obs).all(axis=-1) * \
        np.abs(next_obs[:, 1:] < 100).all(axis=-1) * \
        (height > .7) * \
        (np.abs(angle) < .2)

    done = ~not_done
    return done


def walker2d_terminal_fn(
    obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray
) -> np.ndarray:
    """Terminal condition function for Walker2d."""
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = \
        (height > 0.8) * \
        (height < 2.0) * \
        (angle > -1.0) * \
        (angle < 1.0)

    done = ~not_done
    return done


TERMINAL_FUNCTIONS = {
    'HalfCheetah': halfcheetah_terminal_fn,
    'Hopper': hopper_terminal_fn,
    'Walker2d': walker2d_terminal_fn,
}
