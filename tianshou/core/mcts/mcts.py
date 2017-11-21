import numpy as np
import math
import time

c_puct = 5


class MCTSNode(object):
    def __init__(self, parent, action, state, action_num, prior):
        self.parent = parent
        self.action = action
        self.children = {}
        self.state = state
        self.action_num = action_num
        self.prior = prior

    def selection(self, simulator):
        raise NotImplementedError("Need to implement function selection")

    def backpropagation(self, action):
        raise NotImplementedError("Need to implement function backpropagation")

    def simulation(self, state, evaluator):
        raise NotImplementedError("Need to implement function simulation")


class UCTNode(MCTSNode):
    def __init__(self, parent, action, state, action_num, prior):
        super(UCTNode, self).__init__(parent, action, state, action_num, prior)
        self.Q = np.zeros([action_num])
        self.W = np.zeros([action_num])
        self.N = np.zeros([action_num])
        self.ucb = self.Q + c_puct * self.prior * math.sqrt(np.sum(self.N)) / (self.N + 1)

    def selection(self, simulator):
        action = np.argmax(self.ucb)
        if action in self.children.keys():
            return self.children[action].selection(simulator)
        else:
            self.children[action] = ActionNode(self, action)
            return self.children[action].selection(simulator)

    def backpropagation(self, action):
        action = int(action)
        self.N[action] += 1
        self.W[action] += self.children[action].reward
        for i in range(self.action_num):
            if self.N[i] != 0:
                self.Q[i] = (self.W[i] + 0.) / self.N[i]
        self.ucb = self.Q + c_puct * self.prior * math.sqrt(np.sum(self.N)) / (self.N + 1.)
        if self.parent is not None:
            self.parent.backpropagation(self.children[action].reward)

    def simulation(self, evaluator, state):
        value = evaluator(state)
        return value


class TSNode(MCTSNode):
    def __init__(self, parent, action, state, action_num, prior, method="Gaussian"):
        super(TSNode, self).__init__(parent, action, state, action_num, prior)
        if method == "Beta":
            self.alpha = np.ones([action_num])
            self.beta = np.ones([action_num])
        if method == "Gaussian":
            self.mu = np.zeros([action_num])
            self.sigma = np.zeros([action_num])


class ActionNode:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = {}
        self.next_state = None
        self.reward = 0

    def selection(self, simulator):
        self.next_state, self.reward = simulator.step_forward(self.parent.state, self.action)
        if self.next_state is not None:
            if self.next_state in self.children.keys():
                return self.children[self.next_state].selection(simulator)
            else:
                return self.parent, self.action
        else:
            return self.parent, self.action

    def expansion(self, action_num):
        # TODO: Let users/evaluator give the prior
        if self.next_state is not None:
            prior = np.ones([action_num]) / action_num
            self.children[self.next_state] = UCTNode(self, self.action, self.next_state, action_num, prior)
            return True
        else:
            return False

    def backpropagation(self, value):
        self.reward += value
        self.parent.backpropagation(self.action)


class MCTS:
    def __init__(self, simulator, evaluator, root, action_num, prior, method="UCT", max_step=None, max_time=None):
        self.simulator = simulator
        self.evaluator = evaluator
        self.action_num = action_num
        if method == "UCT":
            self.root = UCTNode(None, None, root, action_num, prior)
        if method == "TS":
            self.root = TSNode(None, None, root, action_num, prior)
        if max_step is not None:
            self.step = 0
            self.max_step = max_step
        if max_time is not None:
            self.start_time = time.time()
            self.max_time = max_time
        if max_step is None and max_time is None:
            raise ValueError("Need a stop criteria!")
        while (max_step is not None and self.step < self.max_step or max_step is None) \
                and (max_time is not None and time.time() - self.start_time < self.max_time or max_time is None):
            print("Q={}".format(self.root.Q))
            print("N={}".format(self.root.N))
            print("W={}".format(self.root.W))
            print("UCB={}".format(self.root.ucb))
            print("\n")
            self.expand()
            if max_step is not None:
                self.step += 1

    def expand(self):
        node, new_action = self.root.selection(self.simulator)
        success = node.children[new_action].expansion(self.action_num)
        if success:
            value = node.simulation(self.evaluator, node.children[new_action].next_state)
            node.children[new_action].backpropagation(value + 0.)
        else:
            node.children[new_action].backpropagation(0.)


if __name__ == "__main__":
    pass
