import numpy as np
from mcts import MCTS
import matplotlib.pyplot as plt

class TestEnv:
    def __init__(self, max_step=5):
        self.max_step = max_step
        self.reward = {i:np.random.uniform() for i in range(2**max_step)}
        self.best = max(self.reward.items(), key=lambda x:x[1])
        print("The best arm is {} with expected reward {}".format(self.best[0],self.best[1]))

    def step_forward(self, state, action):
        if action != 0 and action != 1:
            raise ValueError("Action must be 0 or 1!")
        if state[0] >= 2**state[1] or state[1] >= self.max_step:
            raise ValueError("Invalid State!")
        # print("Operate action {} at state {}, timestep {}".format(action, state[0], state[1]))
        state[0] = state[0] + 2**state[1]*action
        state[1] = state[1] + 1
        return state

    def evaluator(self, state):
        if state[1] == self.max_step:
            reward = int(np.random.uniform() > self.reward[state[0]])
            is_terminated = True
        else:
            reward = 0
            is_terminated = False
        return reward, is_terminated

if __name__=="__main__":
    env = TestEnv(1)
    evaluator = lambda state: env.evaluator(state)
    mcts = MCTS(env, evaluator, [0,0], 2, np.ones([2])/2, max_step=1e4)
