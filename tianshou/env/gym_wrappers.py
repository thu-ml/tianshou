import gym
import numpy as np


class DiscreteToContinuous(gym.ActionWrapper):
    """Gym environment wrapper to take discrete action in a continous environment

    Args:
        env (gym.Environment): gym envirionment with continous action space
        action_per_branch (int): number of discrete actions in each dimension of the action space
    """

    def __init__(self, env, action_per_branch):
        super().__init__(env)
        self.action_per_branch = action_per_branch
        low = self.action_space.low
        high = self.action_space.high
        self.mesh = []
        for l, h in zip(low, high):
            self.mesh.append(np.linspace(l, h, action_per_branch))

    def action(self, act):
        # modify act
        act = np.array([self.mesh[i][a] for i, a in enumerate(act)])
        return act
