# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# $File: test.py
# $Date: Fri Nov 17 13:5600 2017 +0800
# $Author: renyong15 © <mails.tsinghua.edu.cn>
#

from game import Game
from engine import GTPEngine



g = Game()
e = GTPEngine(game_obj = g)
res = e.run_cmd('1 protocol_version')
print(e.known_commands)
print(res)

res = e.run_cmd('2 name')
print(res)

res = e.run_cmd('3 known_command quit')
print(res)

res = e.run_cmd('4 unknown_command quitagain')
print(res)

res = e.run_cmd('5 list_commands')
print(res)

res = e.run_cmd('6 komi 6')
print(res)

res = e.run_cmd('7 play BLACK C3')
print(res)

res = e.run_cmd('8 genmove BLACK')
print(res)


