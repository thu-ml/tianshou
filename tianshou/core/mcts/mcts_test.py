import numpy as np
from mcts import MCTS
import matplotlib.pyplot as plt

class TestEnv:
    def __init__(self, max_step=5):
        self.max_step = max_step
        self.reward = {i:np.random.uniform() for i in range(2**max_step)}
        # self.reward = {0:0.8, 1:0.2, 2:0.4, 3:0.6}
        self.best = max(self.reward.items(), key=lambda x:x[1])
        # print("The best arm is {} with expected reward {}".format(self.best[0],self.best[1]))
        print(self.reward)

    def step_forward(self, state, action):
        if action != 0 and action != 1:
            raise ValueError("Action must be 0 or 1! Your action is {}".format(action))
        if state[0] >= 2**state[1] or state[1] >= self.max_step:
            raise ValueError("Invalid State! Your state is {}".format(state))
        # print("Operate action {} at state {}, timestep {}".format(action, state[0], state[1]))
        new_state = [0,0]
        new_state[0] = state[0] + 2**state[1]*action
        new_state[1] = state[1] + 1
        if new_state[1] == self.max_step:
            reward = int(np.random.uniform() < self.reward[state[0]])
            is_terminated = True
        else:
            reward = 0
            is_terminated = False
        return new_state, reward, is_terminated

if __name__=="__main__":
    env = TestEnv(3)
    evaluator = lambda state: env.step_forward(state, action)
    mcts = MCTS(env, evaluator, [0,0], 2, np.array([0.5,0.5]), max_step=1e4)
