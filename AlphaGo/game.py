# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# $File: game.py
# $Date: Fri Nov 17 15:0745 2017 +0800
# $Author: renyong15 © <mails.tsinghua.edu.cn>
#

import numpy as np
import utils
import Network
from strategy import strategy
from collections import deque


class Game:
    def __init__(self, size=19, komi=6.5):
        self.size = size
        self.komi = 6.5
        self.board = [utils.EMPTY] * (self.size * self.size)
        self.strategy = strategy(Network.forward)
        self.history = deque(maxlen=8)
        for i in range(8):
            self.history.append(self.board)

    def _flatten(self, vertex):
        x, y = vertex
        return (x - 1) * self.size + (y - 1)

    def clear(self):
        self.board = [utils.EMPTY] * (self.size * self.size)

    def set_size(self, n):
        self.size = n
        self.clear()

    def set_komi(self, k):
        self.komi = k

    def do_move(self, color, vertex):
        if vertex == utils.PASS:
            return True

        id_ = self._flatten(vertex)
        if self.board[id_] == utils.EMPTY:
            self.board[id_] = color
            self.history.append(self.board)
            return True
        else:
            return False

    def step_forward(self, state, action):
        if state[0, 0, 0, -1] == 1:
            color = 1
        else:
            color = -1
        if action == 361:
            vertex = (0, 0)
        else:
            vertex = (action / 19 + 1, action % 19)
        self.do_move(color, vertex)
        new_state = np.concatenate([state[:, :, :, 1:8], self.board == 1, state[:, :, :, 9:16], 1 - state[:, :, :, -1]],
                                   axis=3)
        return new_state, 0

    def gen_move(self, color):
        move = self.strategy.gen_move(self.history, color)
        return move
        # return utils.PASS

