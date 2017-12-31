#!/usr/bin/env python
import numpy as np
import ZOGame
import Evaluator
from mcts import MCTS




class Agent:
    def __init__(self, size, color, searching_step, temp):
        self.size = size
        self.color = color
        self.searching_step = searching_step
        self.temp = temp
        self.simulator = ZOGame.ZOTree(self.size)
        self.evaluator = Evaluator.rollout_policy(self.simulator, 2)

    def gen_move(self, seq):
        if len(seq) >= 2 * self.size:
            raise ValueError("Game is terminated.")
        mcts = MCTS(self.simulator, self.evaluator, [seq, self.color], 2, inverse=True)
        mcts.search(max_step=self.searching_step)
        N = mcts.root.N
        N = np.power(N, 1.0 / self.temp)
        prob = N / np.sum(N)
        action = int(np.random.binomial(1, prob[1]))
        return action
