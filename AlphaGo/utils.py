# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# $File: utils.py
# $Date: Mon Nov 27 18:2755 2017 +0800
# $Author: renyong15 © <mails.tsinghua.edu.cn>
#

class Data(object):
    def __init__(self):
        self.boards = []
        self.probs = []
        self.winner = 0

    def reset(self):
        self.__init__()

WHITE = -1
EMPTY = 0
BLACK = +1
FILL = +2
KO = +3
UNKNOWN = +4

PASS = (0,0)
RESIGN = "resign"


def another_color(color):
    return color * -1
