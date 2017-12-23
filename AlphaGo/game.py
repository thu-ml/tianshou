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
import sys, os
import model
from collections import deque
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from tianshou.core.mcts.mcts import MCTS

import go
import reversi

class Game:
    '''
    Load the real game and trained weights.
    
    TODO : Maybe merge with the engine class in future, 
    currently leave it untouched for interacting with Go UI.
    '''
    def __init__(self, name="go", checkpoint_path=None):
        self.name = name
        if self.name == "go":
            self.size = 9
            self.komi = 3.75
            self.board = [utils.EMPTY] * (self.size ** 2)
            self.history = []
            self.history_length = 8
            self.latest_boards = deque(maxlen=8)
            for _ in range(8):
                self.latest_boards.append(self.board)
            self.game_engine = go.Go(size=self.size, komi=self.komi)
        elif self.name == "reversi":
            self.size = 8
            self.history_length = 1
            self.game_engine = reversi.Reversi()
            self.board = self.game_engine.get_board()
        else:
            raise ValueError(name + " is an unknown game...")

        self.evaluator = model.ResNet(self.size, self.size ** 2 + 1, history_length=self.history_length)

    def clear(self):
        self.board = [utils.EMPTY] * (self.size ** 2)
        self.history = []
        for _ in range(self.history_length):
            self.latest_boards.append(self.board)

    def set_size(self, n):
        self.size = n
        self.clear()

    def set_komi(self, k):
        self.komi = k

    def think(self, latest_boards, color):
        mcts = MCTS(self.game_engine, self.evaluator, [latest_boards, color], self.size ** 2 + 1, inverse=True)
        mcts.search(max_step=20)
        temp = 1
        prob = mcts.root.N ** temp / np.sum(mcts.root.N ** temp)
        choice = np.random.choice(self.size ** 2 + 1, 1, p=prob).tolist()[0]
        if choice == self.size ** 2:
            move = utils.PASS
        else:
            move = self.game_engine._deflatten(choice)
        return move, prob

    def play_move(self, color, vertex):
        # this function can be called directly to play the opponent's move
        if vertex == utils.PASS:
            return True
        # TODO this implementation is not very elegant
        if self.name == "go":
            res = self.game_engine.executor_do_move(self.history, self.latest_boards, self.board, color, vertex)
        elif self.name == "reversi":
            res = self.game_engine.executor_do_move(self.board, color, vertex)
        return res

    def think_play_move(self, color):
        # although we don't need to return self.prob, however it is needed for neural network training
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
    g.think_play_move(1)
    #file = open("debug.txt", "a")
    #file.write("mcts check\n")
    #file.close()
