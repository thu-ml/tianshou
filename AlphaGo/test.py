# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# $File: test.py
# $Date: Fri Dec 01 01:3722 2017 +0800
# $Author: renyong15 © <mails.tsinghua.edu.cn>
#

from game import Game
from engine import GTPEngine
import utils

g = Game()
e = GTPEngine(game_obj=g)

e.run_cmd("genmove BLACK")
g.show_board()
e.run_cmd("genmove WHITE")
g.show_board()
e.run_cmd("genmove BLACK")
g.show_board()
e.run_cmd("genmove WHITE")
g.show_board()
e.run_cmd("genmove BLACK")
g.show_board()
e.run_cmd("genmove WHITE")
g.show_board()
e.run_cmd("genmove BLACK")
g.show_board()
e.run_cmd("genmove WHITE")
g.show_board()
e.run_cmd("genmove BLACK")
g.show_board()
e.run_cmd("genmove WHITE")
g.show_board()
