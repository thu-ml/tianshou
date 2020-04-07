import numpy as np
import math
import time

c_puct = 0.01


class MCTSNode(object):
    def __init__(self, parent, action, state, action_num, prior,
                 inverse=False):
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
    def __init__(self, parent, action, state, action_num, prior, mcts,
                 inverse=False):
        super(UCTNode, self).__init__(parent, action, state, action_num, prior,
                                      inverse)
        # self.Q = np.finfo(np.float32).eps.item()
        self.Q = np.random.uniform(0, 1, action_num) * (1e-6)
        self.W = np.zeros([action_num])
        self.N = np.zeros([action_num])
        self.c_puct = c_puct
        self.ucb = self.Q + self.c_puct * self.prior * math.sqrt(
            np.sum(self.N)) / (self.N + 1)
        self.mask = None
        self.elapse_time = 0
        self.mcts = mcts

    def selection(self, simulator):
        head = time.time()
        self.valid_mask(simulator)
        self.mcts.valid_mask_time += time.time() - head
        action = np.argmax(self.ucb)
        # print("\tselection node {} ucb {}".format(self.state[0], self.ucb))
        if action in self.children.keys():
            # print("\told action : ", action)
            self.mcts.state_selection_time += time.time() - head
            return self.children[action].selection(simulator)
        else:  # new action is the expansion step
            # print("\tnew action : ", action)
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
        self.ucb = self.Q + c_puct * self.prior * math.sqrt(np.sum(self.N)) / (
                    self.N + 1.)
        # print("\tnode {} UCB {}".format(self.state[0], self.ucb))
        # print("\tN {} W {} Q {}".format(self.N, self.W, self.Q))
        # print("\tU(s,a)  : {}".format(
        #    c_puct * self.prior * math.sqrt(np.sum(self.N)) / (self.N + 1.)))
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
                self.mask = simulator.simulate_get_mask(self.state,
                                                        range(self.action_num))
            self.ucb[self.mask] = -float("Inf")


class ActionNode(object):
    def __init__(self, parent, action, mcts):
        self.parent = parent
        self.action = action
        self.children = {}
        self.next_state = None
        self.next_state_hashvalue = None
        self.state_type = None
        self.reward = 0
        self.mcts = mcts

    def selection(self, simulator):
        head = time.time()
        self.next_state, self.reward = simulator.simulate_step_forward(
            self.parent.state, self.action)
        self.mcts.simulate_sf_time += time.time() - head
        # next_state is None means that self.parent.state is terminate state
        if self.next_state is None:
            self.mcts.action_selection_time += time.time() - head
            return self
        head = time.time()
        self.next_state_hashvalue = simulator.simulate_hashable_conversion(
            self.next_state)
        self.mcts.hash_time += time.time() - head
        if self.next_state_hashvalue in self.children.keys():
            # next state has already visited before
            self.mcts.action_selection_time += time.time() - head
            return self.children[self.next_state_hashvalue].\
                selection(simulator)
        else:
            # next state is a new state never seen before,
            # so return to _expand for expansion
            self.mcts.action_selection_time += time.time() - head
            return self

    def expansion(self, prior, action_num):
        # print("\t================ expansion state : ", self.next_state[0])
        self.children[self.next_state_hashvalue] = \
            UCTNode(self, self.action,
                    self.next_state,
                    action_num, prior,
                    mcts=self.mcts,
                    inverse=self.parent.inverse)

    def backpropagation(self, value):
        self.reward += value
        self.parent.backpropagation(self.action)


class MCTS(object):
    def __init__(self, simulator, evaluator, start_state, action_num,
                 method="UCT",
                 role="unknown", debug=False, inverse=False, epsilon=0.25):
        self.simulator = simulator  # simulator is a class
        self.evaluator = evaluator  # evaluator is a callable function
        self.role = role
        self.debug = debug
        self.epsilon = epsilon
        # the initialization of the evaluator will call simulate_step_forward
        prior, _ = self.evaluator(start_state)
        # prior = (1 - self.epsilon) * prior + self.epsilon *
        # np.random.dirichlet(1.0/action_num * np.ones([action_num]))
        prior = np.ones([action_num])
        print("initial root prior : ", prior)
        self.action_num = action_num
        if method == "":
            self.root = start_state
        if method == "UCT":
            self.root = UCTNode(None, None, start_state, action_num, prior,
                                mcts=self, inverse=inverse)
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
        if max_step is None and max_time is None:
            raise ValueError("Need a stop criteria!")

        while step < max_step and time.time() - start_time < max_step:
            # print("round : ", step)
            sel_time, exp_time, back_time = self._expand()
            self.selection_time += sel_time
            self.expansion_time += exp_time
            self.backpropagation_time += back_time
            step += 1
        if self.debug:
            file = open("mcts_profiling.log", "a")
            file.write("[" + str(self.role) + "]"
                       + " sel "
                       + '%.3f' % self.selection_time + "  "
                       + " sel_sta "
                       + '%.3f' % self.state_selection_time + "  "
                       + " valid "
                       + '%.3f' % self.valid_mask_time + "  "
                       + " sel_act "
                       + '%.3f' % self.action_selection_time + "  "
                       + " hash "
                       + '%.3f' % self.hash_time + "  "
                       + " step forward "
                       + '%.3f' % self.simulate_sf_time + "  "
                       + " expansion  "
                       + '%.3f' % self.expansion_time + "  "
                       + " backprop "
                       + '%.3f' % self.backpropagation_time + "  "
                       + "\n")
            file.close()

    def _expand(self):
        t0 = time.time()
        # 0. selection
        next_action = self.root.selection(self.simulator)
        t1 = time.time()
        # 1. expansion and simulation
        # next_action.next_state is None means the parent state node of
        # next_action is a terminate node, which means the selection step
        # reaches the terminal node.
        if next_action.next_state is not None:
            # evaluator returns the policy and value for new expansion state
            policy, value = self.evaluator(next_action.next_state)
            next_action.expansion(policy, self.action_num)
        else:
            # print("\t------- terminal node : ", next_action.parent.state[0])
            value = 0
        t2 = time.time()
        # 2. backpropagation
        if self.inverse:
            next_action.backpropagation(-value + 0.)
        else:
            next_action.backpropagation(value + 0.)
        t3 = time.time()
        return t1 - t0, t2 - t1, t3 - t2


if __name__ == "__main__":
    pass
