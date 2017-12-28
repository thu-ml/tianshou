#!/usr/bin/env python
import numpy as np
import ZOGame
import Evaluator
from mcts import MCTS

temp = 1


class Agent:
    def __init__(self, size, color):
        self.size = size
        self.color = color
        self.simulator = ZOGame.ZOTree(self.size)
        self.evaluator = Evaluator.rollout_policy(self.simulator, 2)

    def gen_move(self, seq):
        if len(seq) >= 2 * self.size:
            raise ValueError("Game is terminated.")
        mcts = MCTS(self.simulator, self.evaluator, [seq, self.color], 2, inverse=True)
        mcts.search(max_step=50)
        N = mcts.root.N
        N = np.power(N, 1.0 / temp)
        prob = N / np.sum(N)
        print("prob: {}".format(prob))
        action = int(np.random.binomial(1, prob[1]))
        return action
