import gym
import numpy as np


class ContinuousToDiscrete(gym.ActionWrapper):
    """Gym environment wrapper to take discrete action in a continous environment.

    :param gym.Environment env: gym envirionment with continous action space.
    :param int action_per_branch: number of discrete actions in each dimension
        of the action space.
    """

    def __init__(self, env: gym.Env, action_per_branch: int) -> None:
        super().__init__(env)
        act_space: gym.spaces.Box = env.action_space  # type: ignore
        low, high = act_space.low, act_space.high
        num_branches = act_space.shape[0]
        self.action_space = gym.spaces.MultiDiscrete(
            [action_per_branch] * num_branches
        )
        mesh = []
        for lo, hi in zip(low, high):
            mesh.append(np.linspace(lo, hi, action_per_branch))
        self.mesh = np.array(mesh)

    def action(self, act: np.ndarray) -> np.ndarray:
        # modify act
        return np.array([self.mesh[i][a] for i, a in enumerate(act)])
