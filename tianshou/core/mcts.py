import numpy as np
import math
import json

js = json.load("state_mask.json")
action_num = 2
c_puct = 5.

class MCTSNode:
    def __init__(self, parent, action, state, action_num, prior):
        self.parent = parent
        self.action = action
        self.children = {}
        self.state = state
        self.action_num = action_num
        self.prior = prior

    def select_leaf(self):
        raise NotImplementedError("Need to implement function select_leaf")

    def backup_value(self, action, value):
        raise NotImplementedError("Need to implement function backup_value")

    def expand(self, action):
        raise NotImplementedError("Need to implement function expand")

    def iteration(self):
        raise NotImplementedError("Need to implement function iteration")


class UCTNode(MCTSNode):
    def __init__(self, parent, action, state, action_num, prior):
        super(UCTNode, self).__init__(parent, action, state, action_num, prior)
        self.Q = np.zeros([action_num])
        self.W = np.zeros([action_num])
        self.N = np.zeros([action_num])
        self.ucb = self.Q + c_puct * self.prior * math.sqrt(np.sum(self.N)) / (self.N + 1)

    def select_leaf(self):
        action = np.argmax(self.ucb)
        if action in self.children.keys():
            self.children[action].select_leaf()
        else:
            # TODO: apply the action and evalate next state
            # state, value = self.env.step_forward(self.state, action)
            # self.children[action] = MCTSNode(self.env, self, action, state, prior)
            # self.backup_value(action, value)
            state, value = self.expand(action)
            self.children[action] = UCTNode(self.env, self, action, state, prior)

    def backup_value(self, action, value):
        self.N[action] += 1
        self.W[action] += 1
        self.Q = self.W / self.N
        self.ucb = self.Q + c_puct * self.prior * math.sqrt(np.sum(self.N)) / (self.N + 1)
        self.parent.backup_value(self.parent.action, value)

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

