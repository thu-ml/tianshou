import numpy as np

class TestEnv:
    def __init__(self, max_step=5):
        self.step = 0
        self.state = 0
        self.max_step = max_step
        self.reward = {i:np.random.uniform() for i in range(2**max_step)}
        self.best = max(self.reward.items(), key=lambda x:x[1])
        print("The best arm is {} with expected reward {}".format(self.best[0],self.best[1]))

    def step_forward(self, action):
        print("Operate action {} at timestep {}".format(action, self.step))
        self.state = self.state + 2**self.step*action
        self.step = self.step + 1
        if self.step == self.max_step:
            reward = int(np.random.uniform() > self.reward[self.state])
            print("Get reward {}".format(reward))
        else:
            reward = 0
        return [self.state, reward]

if __name__=="__main__":
    env = TestEnv(1)
    env.step_forward(1)