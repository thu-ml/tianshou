import numpy as np
import math
import time

c_puct = 5

class MCTSNode(object):
    def __init__(self, parent, action, state, action_num, prior, inverse=False):
        self.parent = parent
        self.action = action
        self.children = {}
        self.state = state
        self.action_num = action_num
        self.prior = np.array(prior).reshape(-1)
        self.inverse = inverse

    def selection(self, simulator):
        raise NotImplementedError("Need to implement function selection")

    def backpropagation(self, action):
        raise NotImplementedError("Need to implement function backpropagation")

    def valid_mask(self, simulator):
        pass

class UCTNode(MCTSNode):
    def __init__(self, parent, action, state, action_num, prior, mcts, inverse=False):
        super(UCTNode, self).__init__(parent, action, state, action_num, prior, inverse)
        self.Q = np.random.uniform(-1, 1, action_num) * (1e-6)
        self.W = np.zeros([action_num])
        self.N = np.zeros([action_num])
        self.c_puct = c_puct
        self.ucb = self.Q + self.c_puct * self.prior * math.sqrt(np.sum(self.N)) / (self.N + 1)
        self.mask = None
        self.elapse_time = 0
        self.mcts = mcts

    def selection(self, simulator):
        head = time.time()
        self.valid_mask(simulator)
        self.mcts.valid_mask_time += time.time() - head
        action = np.argmax(self.ucb)
        if action in self.children.keys():
            self.mcts.state_selection_time += time.time() - head
            return self.children[action].selection(simulator)
        else:
            self.children[action] = ActionNode(self, action, mcts=self.mcts)
            self.mcts.state_selection_time += time.time() - head
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
            if self.inverse:
                self.parent.backpropagation(-self.children[action].reward)
            else:
                self.parent.backpropagation(self.children[action].reward)

    def valid_mask(self, simulator):
        # let all invalid actions be illegal in mcts
        if not hasattr(simulator, 'simulate_get_mask'):
            pass
        else:
            if self.mask is None:
                self.mask = simulator.simulate_get_mask(self.state, range(self.action_num))
            self.ucb[self.mask] = -float("Inf")

# Code reserved for Thompson Sampling
class TSNode(MCTSNode):
    def __init__(self, parent, action, state, action_num, prior, method="Gaussian", inverse=False):
        super(TSNode, self).__init__(parent, action, state, action_num, prior, inverse)
        if method == "Beta":
            self.alpha = np.ones([action_num])
            self.beta = np.ones([action_num])
        if method == "Gaussian":
            self.mu = np.zeros([action_num])
            self.sigma = np.zeros([action_num])


class ActionNode(object):
    def __init__(self, parent, action, mcts):
        self.parent = parent
        self.action = action
        self.children = {}
        self.next_state = None
        self.next_state_hashable = None
        self.state_type = None
        self.reward = 0
        self.mcts = mcts

    def selection(self, simulator):
        head = time.time()
        self.next_state, self.reward = simulator.simulate_step_forward(self.parent.state, self.action)
        self.mcts.simulate_sf_time += time.time() - head
        if self.next_state is None: # next_state is None means that self.parent.state is the terminate state
            self.mcts.action_selection_time += time.time() - head
            return self
        head = time.time()
        self.next_state_hashable = simulator.simulate_hashable_conversion(self.next_state)
        self.mcts.hash_time += time.time() - head
        if self.next_state_hashable in self.children.keys(): # next state has already visited before
            self.mcts.action_selection_time += time.time() - head
            return self.children[self.next_state_hashable].selection(simulator)
        else: # next state is a new state never seen before
            self.mcts.action_selection_time += time.time() - head
            return self

    def expansion(self, prior, action_num):
        self.children[self.next_state_hashable] = UCTNode(self, self.action, self.next_state, action_num, prior,
                                                          mcts=self.mcts, inverse=self.parent.inverse)

    def backpropagation(self, value):
        self.reward += value
        self.parent.backpropagation(self.action)

class MCTS(object):
    def __init__(self, simulator, evaluator, start_state, action_num, method="UCT",
                 role="unknown", debug=False, inverse=False, epsilon=0.25):
        self.simulator = simulator
        self.evaluator = evaluator
        self.role = role
        self.debug = debug
        self.epsilon = epsilon
        prior, _ = self.evaluator(start_state)
        prior = (1 - self.epsilon) * prior + self.epsilon * np.random.dirichlet(1.0/action_num * np.ones([action_num]))
        self.action_num = action_num
        if method == "":
            self.root = start_state
        if method == "UCT":
            self.root = UCTNode(None, None, start_state, action_num, prior, mcts=self, inverse=inverse)
        if method == "TS":
            self.root = TSNode(None, None, start_state, action_num, prior, inverse=inverse)
        self.inverse = inverse

        # time spend on each step
        self.selection_time = 0
        self.expansion_time = 0
        self.backpropagation_time = 0
        self.action_selection_time = 0
        self.state_selection_time = 0
        self.simulate_sf_time = 0
        self.valid_mask_time = 0
        self.hash_time = 0

    def search(self, max_step=None, max_time=None):
        step = 0
        start_time = time.time()
        if max_step is None:
            max_step = int("Inf")
        if max_time is None:
            max_time = float("Inf")
        if max_step is None and max_time is None:
            raise ValueError("Need a stop criteria!")

        while step < max_step and time.time() - start_time < max_step:
            sel_time, exp_time, back_time = self._expand()
            self.selection_time += sel_time
            self.expansion_time += exp_time
            self.backpropagation_time += back_time
            step += 1
        if self.debug:
            file = open("mcts_profiling.log", "a")
            file.write("[" + str(self.role) + "]"
                       + " sel " + '%.3f' % self.selection_time + "  "
                       + " sel_sta " + '%.3f' % self.state_selection_time + "  "
                       + " valid " + '%.3f' % self.valid_mask_time + "  "
                       + " sel_act " + '%.3f' % self.action_selection_time + "  "
                       + " hash " + '%.3f' % self.hash_time + "  "
                       + " step forward " + '%.3f' % self.simulate_sf_time + "  "
                       + " expansion  " + '%.3f' % self.expansion_time + "  "
                       + " backprop " + '%.3f' % self.backpropagation_time + "  "
                       + "\n")
            file.close()

    def _expand(self):
        t0 = time.time()
        next_action = self.root.selection(self.simulator)
        t1 = time.time()
        # next_action.next_state is None means the parent state node of next_action is a terminate node
        if next_action.next_state is not None:
            prior, value = self.evaluator(next_action.next_state)
            next_action.expansion(prior, self.action_num)
        else:
            value = 0
        t2 = time.time()
        if self.inverse:
            next_action.backpropagation(-value + 0.)
        else:
            next_action.backpropagation(value + 0.)
        t3 = time.time()
        return t1 - t0, t2 - t1, t3 - t2

if __name__ == "__main__":
    pass
