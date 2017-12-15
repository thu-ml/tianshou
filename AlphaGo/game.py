# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# $File: game.py
# $Date: Fri Dec 01 01:3738 2017 +0800
# $Author: renyong15 © <mails.tsinghua.edu.cn>
#
from __future__ import print_function
import utils
import copy
import tensorflow as tf
import numpy as np
import sys
import go
import network_small
import strategy
from collections import deque
from tianshou.core.mcts.mcts import MCTS

import Network
#from strategy import strategy

class Game:
    '''
    Load the real game and trained weights.
    
    TODO : Maybe merge with the engine class in future, 
    currently leave it untouched for interacting with Go UI.
    '''
    def __init__(self, size=9, komi=6.5, checkpoint_path=None):
        self.size = size
        self.komi = komi
        self.board = [utils.EMPTY] * (self.size * self.size)
        self.history = []
        self.past = deque(maxlen=8)
        for _ in range(8):
            self.past.append(self.board)

        self.executor = go.Go(game=self)
        #self.strategy = strategy(checkpoint_path)

        self.simulator = strategy.GoEnv()
        self.net = network_small.Network()
        self.sess = self.net.forward(checkpoint_path)
        self.evaluator = lambda state: self.sess.run([tf.nn.softmax(self.net.p), self.net.v],
                                                     feed_dict={self.net.x: state, self.net.is_training: False})

    def _flatten(self, vertex):
        x, y = vertex
        return (y - 1) * self.size + (x - 1)

    def _deflatten(self, idx):
        x = idx % self.size + 1
        y = idx // self.size + 1
        return (x,y)

    def clear(self):
        self.board = [utils.EMPTY] * (self.size * self.size)
        self.history = []
        for _ in range(8):
            self.past.append(self.board)

    def set_size(self, n):
        self.size = n
        self.clear()

    def set_komi(self, k):
        self.komi = k

    def data_process(self, history, color):
        state = np.zeros([1, self.simulator.size, self.simulator.size, 17])
        for i in range(8):
            state[0, :, :, i] = np.array(np.array(history[i]) == np.ones(self.simulator.size ** 2)).reshape(self.simulator.size, self.simulator.size)
            state[0, :, :, i + 8] = np.array(np.array(history[i]) == -np.ones(self.simulator.size ** 2)).reshape(self.simulator.size, self.simulator.size)
        if color == utils.BLACK:
            state[0, :, :, 16] = np.ones([self.simulator.size, self.simulator.size])
        if color == utils.WHITE:
            state[0, :, :, 16] = np.zeros([self.simulator.size, self.simulator.size])
        return state

    def strategy_gen_move(self, history, color):
        self.simulator.history = copy.copy(history)
        self.simulator.board = copy.copy(history[-1])
        state = self.data_process(self.simulator.history, color)
        mcts = MCTS(self.simulator, self.evaluator, state, self.simulator.size ** 2 + 1, inverse=True, max_step=10)
        temp = 1
        prob = mcts.root.N ** temp / np.sum(mcts.root.N ** temp)
        choice = np.random.choice(self.simulator.size ** 2 + 1, 1, p=prob).tolist()[0]
        if choice == self.simulator.size ** 2:
            move = utils.PASS
        else:
            move = (choice % self.simulator.size + 1, choice / self.simulator.size + 1)
        return move, prob

    def do_move(self, color, vertex):
        if vertex == utils.PASS:
            return True
        res = self.executor.do_move(color, vertex)
        return res

    def gen_move(self, color):
        # move = self.strategy.gen_move(color)
        # return move
        move, self.prob = self.strategy_gen_move(self.past, color)
        self.do_move(color, move)
        return move

    def status2symbol(self, s):
        pool = {utils.WHITE: 'O', utils.EMPTY: '.', utils.BLACK: 'X', utils.FILL: 'F', utils.UNKNOWN: '?'}
        return pool[s]

    def show_board(self):
        row = [i for i in range(1, 20)]
        col = ' abcdefghijklmnopqrstuvwxyz'
        print(' ', end='')
        for j in range(self.size + 1):
            print(col[j], end='  ')
        print('')
        for i in range(self.size):
            print(row[i], end='  ')
            if row[i] < 10:
                print(' ', end='')
            for j in range(self.size):
                print(self.status2symbol(self.board[self._flatten((j + 1, i + 1))]), end='  ')
            print('')
        sys.stdout.flush()

if __name__ == "__main__":
    g = Game()
    g.show_board()
