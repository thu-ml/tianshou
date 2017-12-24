# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# $File: mcts_virtual_loss.py
# $Date: Sun Dec 24 16:4740 2017 +0800
# Original file: mcts.py
# $Author: renyong15 © <mails.tsinghua.edu.cn>
#

"""
    This is an implementation of the MCTS with virtual loss.
    Due to the limitation of Python design mechanism, we implements the virtual loss in a mini-batch
    manner.
"""

from __future__ import absolute_import

import numpy as np
import math
import time
import sys,os
from .utils import list2tuple, tuple2list


class MCTSNodeVirtualLoss(object):
    """
        MCTS abstract class with virtual loss. Currently we only support UCT node.
        Role of the Parameters can be found in Readme.md.
    """
    def __init__(self, 
                 parent, 
                 action, 
                 state, 
                 action_num, 
                 prior, 
                 inverse = False):
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

class UCTNodeVirtualLoss(MCTSNodeVirtualLoss):
    """
        UCT node (state node) with virtual loss.
        Role of the Parameters can be found in Readme.md.
        :param c_puct balance between exploration and exploition,
    """
    def __init__(self, 
                 parent, 
                 action, 
                 state, 
                 action_num, 
                 prior, 
                 inverse=False, 
                 c_puct = 5):
        super(UCTNodeVirtualLoss, self).__init__(parent, action, state, action_num, prior, inverse)
        self.Q = np.zeros([action_num])
        self.W = np.zeros([action_num])
        self.N = np.zeros([action_num])
        self.virtual_loss = np.zeros([action_num])
        self.c_puct = c_puct
        #### modified by adding virtual loss
        #self.ucb = self.Q + c_puct * self.prior * math.sqrt(np.sum(self.N)) / (self.N + 1)

        self.mask = None

    def selection(self, 
                  simulator):
        self.valid_mask(simulator)
        self.Q = np.zeros([self.action_num])
        N_not_zero = (self.N + self.virtual_loss) > 0
        self.Q[N_not_zero] = (self.W[N_not_zero] + 0.)/ (self.virtual_loss[N_not_zero] + self.N[N_not_zero])
        self.ucb = self.Q + self.c_puct * self.prior * math.sqrt(np.sum(self.N + self.virtual_loss)) /\
                   (self.N + self.virtual_loss + 1)
        action = np.argmax(self.ucb)
        self.virtual_loss[action] += 1

        if action in self.children.keys():
            return self.children[action].selection(simulator)
        else:
            self.children[action] = ActionNodeVirtualLoss(self, action)
            return self.children[action].selection(simulator)

    def remove_virtual_loss(self):
        ### if not virtual_loss for every action is zero
        if np.sum(self.virtual_loss > 0) > 0:
            self.virtual_loss = np.zeros([self.action_num])
            if self.parent:
                self.parent.remove_virtual_loss()

    def backpropagation(self, action):
        action = int(action)
        self.N[action] += 1
        self.W[action] += self.children[action].reward

        ## do not need to  compute Q and ucb immediately since it will be modified by virtual loss
        ## just comment out and leaving for comparision
        #for i in range(self.action_num):
        #    if self.N[i] != 0:
        #        self.Q[i] = (self.W[i] + 0.) / self.N[i]
        #self.ucb = self.Q + c_puct * self.prior * math.sqrt(np.sum(self.N)) / (self.N + 1.)

        if self.parent is not None:
            if self.inverse:
                self.parent.backpropagation(-self.children[action].reward)
            else:
                self.parent.backpropagation(self.children[action].reward)

    def valid_mask(self, simulator):
        if self.mask is None:
            start_time = time.time()
            self.mask = []
            for act in range(self.action_num - 1):
                if not simulator.simulate_is_valid(self.state, act):
                    self.mask.append(act)
                    self.ucb[act] = -float("Inf")
        else:
            self.ucb[self.mask] = -float("Inf")



class ActionNodeVirtualLoss(object):
    """
        Action node with virtual loss.
    """
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = {}
        self.next_state = None
        self.origin_state = None
        self.state_type = None
        self.reward = 0

    def remove_virtual_loss(self):
        self.parent.remove_virtual_loss()

    def type_conversion_to_tuple(self):
        if type(self.next_state) is np.ndarray:
            self.next_state = self.next_state.tolist()
        if type(self.next_state) is list:
            self.next_state = list2tuple(self.next_state)

    def type_conversion_to_origin(self):
        if self.state_type is np.ndarray:
            self.next_state = np.array(self.next_state)
        if self.state_type is list:
            self.next_state = tuple2list(self.next_state)

    def selection(self, simulator):
        self.next_state, self.reward = simulator.step_forward(self.parent.state, self.action)
        self.origin_state = self.next_state
        self.state_type = type(self.next_state)
        self.type_conversion_to_tuple()
        if self.next_state is not None:
            if self.next_state in self.children.keys():
                return self.children[self.next_state].selection(simulator)
            else:
                return self.parent, self.action
        else:
            return self.parent, self.action

    def expansion(self, action, state, action_num, prior, inverse ):
        if state is not None:
            self.children[state] = UCTNodeVirtualLoss(self, action, state, action_num, prior, inverse)


    def backpropagation(self, value):
        self.reward += value
        self.parent.backpropagation(self.action)


class MCTSVirtualLoss(object):
    """
        MCTS class with virtual loss 
    """
    def __init__(self, simulator, evaluator, root, action_num, batch_size = 1, method = "UCT", inverse = False):
        self.simulator = simulator
        self.evaluator = evaluator
        prior, _ = self.evaluator(root)
        self.action_num = action_num
        self.batch_size = batch_size

        if method == "":
            self.root = root
        elif method == "UCT":
            self.root = UCTNodeVirtualLoss(None, None, root, action_num, prior, inverse)
        elif method == "TS":
            self.root = TSNodeVirtualLoss(None, None, root, action_num, prior, inverse=inverse)
        else:
            raise ValueError("Need a root type")

        self.inverse = inverse


    def do_search(self, max_step=None, max_time=None):
        """
            Expand the MCTS tree with stop crierion either by max_step or max_time
        
            :param max_step search maximum minibath rounds. ONE step is ONE minibatch
            :param max_time search maximum seconds
        """
        if max_step is not None:
            self.step = 0
            self.max_step = max_step
        if max_time is not None:
            self.start_time = time.time()
            self.max_time = max_time
        if max_step is None and max_time is None:
            raise ValueError("Need a stop criteria!")

        self.select_time = []
        self.evaluate_time = []
        self.bp_time = []
        while (max_step is not None and self.step < self.max_step or max_step is None) \
                and (max_time is not None and time.time() - self.start_time < self.max_time or max_time is None):
            self._expand()
            if max_step is not None:
                self.step += 1

    def _expand(self):
        """
            Core logic method for MCTS tree to expand nodes.
            Steps to expand node:
            1. Select final action node with virtual loss and collect them in to a minibatch.
               (i.e. root->action->state->action...->action)
            2. Remove the virtual loss
            3. Evaluate the whole minibatch using evaluator 
            4. Expand new nodes and perform back propogation.
        """
        ## minibatch with virtual loss
        nodes = []
        new_actions = []
        next_states = []

        for i in range(self.batch_size):
            node, new_action = self.root.selection(self.simulator)
            nodes.append(node)
            new_actions.append(new_action)
            next_states.append(node.children[new_action].next_state)

        for node in nodes:
            node.remove_virtual_loss()

        assert(np.sum(self.root.virtual_loss > 0) == 0)
        #### compute value in batch manner unless the evaluator do not support it
        try:
            priors, values = self.evaluator(next_states)
        except:
            priors = []
            values = []
            for i in range(self.batch_size):
                if next_states[i] is not None:
                    prior, value = self.evaluator(next_states[i])
                    priors.append(prior)
                    values.append(value)
                else:
                    priors.append(0.)
                    values.append(0.)

        #### for now next_state == origin_state
        #### may have problem here. What if we reached the same next_state with same parent and action pair
        for i in range(self.batch_size):
            nodes[i].children[new_actions[i]].expansion(new_actions[i],
                                                        next_states[i],
                                                        self.action_num,
                                                        priors[i],
                                                        nodes[i].inverse)

        for i in range(self.batch_size):
            nodes[i].children[new_actions[i]].backpropagation(values[i] + 0.)


##### TODO 
class TSNodeVirtualLoss(MCTSNodeVirtualLoss):
    def __init__(self, parent, action, state, action_num, prior, method="Gaussian", inverse=False):
        super(TSNodeVirtualLoss, self).__init__(parent, action, state, action_num, prior, inverse)
        if method == "Beta":
            self.alpha = np.ones([action_num])
            self.beta = np.ones([action_num])
        if method == "Gaussian":
            self.mu = np.zeros([action_num])
            self.sigma = np.zeros([action_num])

if __name__ == "__main__":
    mcts_virtual_loss = MCTSNodeVirtualLoss(None, None, 10, 1, 'UCT')
