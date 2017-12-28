import numpy as np


class evaluator(object):
    def __init__(self, env, action_num):
        self.env = env
        self.action_num = action_num

    def __call__(self, state):
        raise NotImplementedError("Need to implement the evaluator")


class rollout_policy(evaluator):
    def __init__(self, env, action_num):
        super(rollout_policy, self).__init__(env, action_num)
        self.is_terminated = False

    def __call__(self, state):
        # TODO: prior for rollout policy
        total_reward = 0.
        color = state[1]
        action = np.random.randint(0, self.action_num)
        state, reward = self.env.simulate_step_forward(state, action)
        total_reward += reward
        while state is not None:
            action = np.random.randint(0, self.action_num)
            state, reward = self.env.simulate_step_forward(state, action)
            total_reward += reward
        return np.ones([self.action_num])/self.action_num, total_reward * color
