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
        self.board = [utils.EMPTY] * (self.size ** 2)
        self.history = []
        self.latest_boards = deque(maxlen=8)
        for _ in range(8):
            self.latest_boards.append(self.board)

        self.executor = go.Go(game=self)
        #self.strategy = strategy(checkpoint_path)

        self.simulator = strategy.GoEnv(game=self)
        self.net = network_small.Network()
        self.sess = self.net.forward(checkpoint_path)
        self.evaluator = lambda state: self.sess.run([tf.nn.softmax(self.net.p), self.net.v],
                                                     feed_dict={self.net.x: state, self.net.is_training: False})

    def _flatten(self, vertex):
        x, y = vertex
        return (x - 1) * self.size + (y - 1)

    def _deflatten(self, idx):
        x = idx // self.size + 1
        y = idx % self.size + 1
        return (x, y)

    def clear(self):
        self.board = [utils.EMPTY] * (self.size ** 2)
        self.history = []
        for _ in range(8):
            self.latest_boards.append(self.board)

    def set_size(self, n):
        self.size = n
        self.clear()

    def set_komi(self, k):
        self.komi = k

    def generate_nn_input(self, latest_boards, color):
        state = np.zeros([1, self.size, self.size, 17])
        for i in range(8):
            state[0, :, :, i] = np.array(np.array(latest_boards[i]) == np.ones(self.size ** 2)).reshape(self.size, self.size)
            state[0, :, :, i + 8] = np.array(np.array(latest_boards[i]) == -np.ones(self.size ** 2)).reshape(self.size, self.size)
        if color == utils.BLACK:
            state[0, :, :, 16] = np.ones([self.size, self.size])
        if color == utils.WHITE:
            state[0, :, :, 16] = np.zeros([self.size, self.size])
        return state

    def think(self, latest_boards, color):
        # TODO : using copy is right, or should we change to deepcopy?
        self.simulator.simulate_latest_boards = copy.copy(latest_boards)
        self.simulator.simulate_board = copy.copy(latest_boards[-1])
        nn_input = self.generate_nn_input(self.simulator.simulate_latest_boards, color)
        mcts = MCTS(self.simulator, self.evaluator, nn_input, self.size ** 2 + 1, inverse=True, max_step=1)
        temp = 1
        prob = mcts.root.N ** temp / np.sum(mcts.root.N ** temp)
        choice = np.random.choice(self.size ** 2 + 1, 1, p=prob).tolist()[0]
        if choice == self.size ** 2:
            move = utils.PASS
        else:
            move = self._deflatten(choice)
        return move, prob

    def play_move(self, color, vertex):
        # this function can be called directly to play the opponent's move
        if vertex == utils.PASS:
            return True
        res = self.executor.executor_do_move(color, vertex)
        return res

    def think_play_move(self, color):
        # although we dont need to return self.prob, however it is needed for neural network training
        move, self.prob = self.think(self.latest_boards, color)
        # play the move immediately
        self.play_move(color, move)
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
    #file = open("debug.txt", "a")
    #file.write("mcts check\n")
    #file.close()
