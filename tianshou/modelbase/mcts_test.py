import numpy as np
from mcts import MCTS

class Evaluator(object):
    def __init__(self, env, action_num):
        self.env = env
        self.action_num = action_num
        self.is_terminated = False

    def __call__(self, state):
        return np.ones([self.action_num]), 0
        '''
        # TODO: prior for rollout policy
        total_reward = 0.
        action = np.random.randint(0, self.action_num)
        state, reward = self.env.simulate_step_forward(state, action)
        total_reward += reward
        while state is not None:
            action = np.random.randint(0, self.action_num)
            state, reward = self.env.simulate_step_forward(state, action)
            total_reward += reward
        return np.ones([self.action_num])/self.action_num, total_reward
        '''

class TestEnv:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.leaf_node_number = 2 ** max_depth
        self.half_leaf_node_num = 2 ** (max_depth - 1)
        self.reward = {i: np.random.uniform() for i in range(self.leaf_node_number)}
        # failed when cput == 0.01 failed when max_step = 100, success when max_step = 1000
        #self.reward = {0: 0.024294218657006095, 1: 0.14179687382318884, 2: 0.16441451451991795, 3: 0.019687232249671616}
        # failed when cput == 0.1 no matter how large the max_step is, because they are too close
        #self.reward = {0: 0.0035243152138864087, 1: 0.8803679420277104, 2: 0.23595215356625954, 3: 0.6520143090430304}
        # failed when cput == 1, max_step == 100, 400 failed, max_step == 200 or 300 success, it shocks
        #self.reward = {0: 0.8740523271103036, 1: 0.04428927517208081, 2: 0.8643087058089024, 3: 0.02011970879109981}
        # failed when cput == 0.01, no matter how large the max_step is
        #self.reward = {0: 0.06594878953409933, 1: 0.49721536292003776, 2: 0.26740756411293487, 3: 0.36596764079057453}
        # another failure
        #self.reward = {0: 0.44351562749061946, 1: 0.051276175250161815, 2: 0.6484666942496972, 3: 0.4799672332733024}
        self.max_reward = [0, 0]
        for i in range(self.half_leaf_node_num):
            self.max_reward[0] = max(self.max_reward[0], self.reward[i])
            self.max_reward[1] = max(self.max_reward[1], self.reward[self.half_leaf_node_num + i])
        # self.reward = {0:1, 1:0}
        self.best = max(self.reward.items(), key=lambda x: x[1])
        print(self.reward)
        #print("The best arm is {} with expected reward {}".format(self.best[0],self.best[1]))

    def simulate_is_valid(self, state, act):
        return True

    def simulate_step_forward(self, state, action):
        #print("simulate_step_forward - Operate action {} at state {}, depth {}".format(action, state[0], state[1]))
        num, depth = state
        if action not in range(0, 2):
            raise ValueError("Action must be 0 or 1! Your action is {}".format(action))
        next_num = num * 2 + action
        next_depth = depth + 1
        next_state = [next_num, next_depth]

        if next_depth > self.max_depth:
            return None, 0
        elif next_depth == self.max_depth:
            #print("\tsimulator reach the terminal state : ", next_state[0])
            return next_state, self.reward[next_num - self.leaf_node_number]
        else:
            return next_state, 0

    def simulate_hashable_conversion(self, state):
        return state[0] # the number of the state

if __name__ == "__main__":
    env = TestEnv(max_depth=2)
    evaluator = lambda state: Evaluator(env, 2)(state)
    root_number = 1
    root_depth = 0
    root_node = [root_number, root_depth]
    mcts = MCTS(env, evaluator, root_node, action_num=2)
    mcts.search(max_step=50000)
    print("The root ucb is {}.\nThe max reward vec is {}".format(mcts.root.ucb, env.max_reward))
    if (np.argmax(mcts.root.ucb) == np.argmax(env.max_reward)):
        print("MCTS find the right action successfully!")
    else:
        print("MCTS failed !!!, it select action {} on root node.\n".format(np.argmax(mcts.root.ucb)))

# will cause problem
# {0: 0.8740523271103036, 1: 0.04428927517208081, 2: 0.8643087058089024, 3: 0.02011970879109981}
# initial root prior :  [1. 1.]

