# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# $File: game.py
# $Date: Fri Nov 17 15:0745 2017 +0800
# $Author: renyong15 © <mails.tsinghua.edu.cn>
#

import utils


class Game:
    def __init__(self, size=19, komi=6.5):
        self.size = size
        self.komi = 6.5
        self.board = [utils.EMPTY] * (self.size * self.size)
        self.strategy = None

    def _flatten(self, vertex):
        x,y = vertex
        return (x-1) * self.size + (y-1)
        

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
            return True
        else:
            return False

    def gen_move(self, color):
        move = self.strategy.gen_move(color)
        return move
        #return utils.PASS


    
