import numpy as np
from mcts import MCTS
from evaluator import rollout_policy


class TestEnv:
    def __init__(self, max_step=5):
        self.max_step = max_step
        self.reward = {i: np.random.uniform() for i in range(2 ** max_step)}
        # self.reward = {0:1, 1:0}
        self.best = max(self.reward.items(), key=lambda x: x[1])
        print(self.reward)
        # print("The best arm is {} with expected reward {}".format(self.best[0],self.best[1]))

    def step_forward(self, state, action):
        if action != 0 and action != 1:
            raise ValueError("Action must be 0 or 1! Your action is {}".format(action))
        if state[0] >= 2 ** state[1] or state[1] > self.max_step:
            raise ValueError("Invalid State! Your state is {}".format(state))
        # print("Operate action {} at state {}, timestep {}".format(action, state[0], state[1]))
        if state[1] == self.max_step:
            new_state = None
            reward = 0
        else:
            num = state[0] + 2 ** state[1] * action
            step = state[1] + 1
            new_state = [num, step]
            if step == self.max_step:
                reward = int(np.random.uniform() < self.reward[num])
            else:
                reward = 0
        return new_state, reward


if __name__ == "__main__":
    env = TestEnv(2)
    rollout = rollout_policy(env, 2)
    evaluator = lambda state: rollout(state)
    mcts = MCTS(env, evaluator, [0, 0], 2, np.array([0.5, 0.5]), max_step=1e4)
