import numpy as np
import math
import time

c_puct = 1


class MCTSNode(object):
    def __init__(self, parent, action, state, action_num, prior):
        self.parent = parent
        self.action = action
        self.children = {}
        self.state = state
        self.action_num = action_num
        self.prior = prior

    def selection(self):
        raise NotImplementedError("Need to implement function selection")

    def backpropagation(self, action, value):
        raise NotImplementedError("Need to implement function backpropagation")

    def expansion(self, simulator, action):
        raise NotImplementedError("Need to implement function expansion")

    def simulation(self, state, evaluator):
        raise NotImplementedError("Need to implement function simulation")


class UCTNode(MCTSNode):
    def __init__(self, parent, action, state, action_num, prior):
        super(UCTNode, self).__init__(parent, action, state, action_num, prior)
        self.Q = np.zeros([action_num])
        self.W = np.zeros([action_num])
        self.N = np.zeros([action_num])
        self.ucb = self.Q + c_puct * self.prior * math.sqrt(np.sum(self.N)) / (self.N + 1)
        self.is_terminated = False

    def selection(self):
        if not self.is_terminated:
            action = np.argmax(self.ucb)
            if action in self.children.keys():
                node, action = self.children[action].selection()
            else:
                node = self
        else:
            action = None
            node = self
        return node, action

    def backpropagation(self, action, value):
        if action is None:
            if self.parent is not None:
                self.parent.backpropagation(self.action, value)
        else:
            self.N[action] += 1
            self.W[action] += value
            for i in range(self.action_num):
                if self.N[i] != 0:
                    self.Q[i] = (self.W[i] + 0.)/self.N[i]
            self.ucb = self.Q + c_puct * self.prior * math.sqrt(np.sum(self.N)) / (self.N + 1.)
            if self.parent is not None:
                self.parent.backpropagation(self.action, value)

    def expansion(self, simulator, action):
        next_state, is_terminated = simulator.step_forward(self.state, action)
        # TODO: Let users/evaluator give the prior
        prior = np.ones([self.action_num]) / self.action_num
        self.children[action] = UCTNode(self, action, next_state, self.action_num, prior)
        self.children[action].is_terminated = is_terminated

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
        self.value = {}


class MCTS:
    def __init__(self, simulator, evaluator, root, action_num, prior, method="UCT", max_step=None, max_time=None):
        self.simulator = simulator
        self.evaluator = evaluator
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
            print(self.root.Q)
            self.expand()
            if max_step is not None:
                self.step += 1

    def expand(self):
        node, new_action = self.root.selection()
        if new_action is None:
            value = node.simulation(self.evaluator, node.state)
            node.backpropagation(new_action, value)
        else:
            node.expansion(self.simulator, new_action)
            value = node.simulation(self.evaluator, node.children[new_action].state)
            node.backpropagation(new_action, value)


if __name__=="__main__":
    pass