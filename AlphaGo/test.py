# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# $File: test.py
# $Date: Tue Nov 28 14:4717 2017 +0800
# $Author: renyong15 © <mails.tsinghua.edu.cn>
#

from game import Game
from engine import GTPEngine
import utils

g = Game()
e = GTPEngine(game_obj=g)
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

res = e.run_cmd('7 play BLACK D4')
print(res)

# res = e.run_cmd('play BLACK C4')
# res = e.run_cmd('play BLACK C5')
# res = e.run_cmd('play BLACK C6')
# res = e.run_cmd('play BLACK D3')
# print(res)


res = e.run_cmd('8 genmove WHITE')
print(res)
g.show_board()

# res = e.run_cmd('8 genmove BLACK')
# print(res)
# g.show_board()
#
# res = e.run_cmd('8 genmove WHITE')
# print(res)
# g.show_board()
#
# res = e.run_cmd('8 genmove BLACK')
# print(res)
# g.show_board()
#
# res = e.run_cmd('8 genmove WHITE')
# print(res)
# g.show_board()
# #g.show_board()
# print(g.check_valid((10, 9)))
# print(g.executor._neighbor((1,1)))
# print(g.do_move(utils.WHITE, (4, 6)))
# #g.show_board()
#
#
# res = e.run_cmd('play BLACK L10')
# res = e.run_cmd('play BLACK L11')
# res = e.run_cmd('play BLACK L12')
# res = e.run_cmd('play BLACK L13')
# res = e.run_cmd('play BLACK L14')
# res = e.run_cmd('play BLACK m15')
# res = e.run_cmd('play BLACK m9')
# res = e.run_cmd('play BLACK C9')
# res = e.run_cmd('play BLACK D9')
# res = e.run_cmd('play BLACK E9')
# res = e.run_cmd('play BLACK F9')
# res = e.run_cmd('play BLACK G9')
# res = e.run_cmd('play BLACK H9')
# res = e.run_cmd('play BLACK I9')
#
# res = e.run_cmd('play BLACK N9')
# res = e.run_cmd('play BLACK N15')
# res = e.run_cmd('play BLACK O10')
# res = e.run_cmd('play BLACK O11')
# res = e.run_cmd('play BLACK O12')
# res = e.run_cmd('play BLACK O13')
# res = e.run_cmd('play BLACK O14')
# res = e.run_cmd('play BLACK M12')
#
# res = e.run_cmd('play WHITE M10')
# res = e.run_cmd('play WHITE M11')
# res = e.run_cmd('play WHITE N10')
# res = e.run_cmd('play WHITE N11')
#
# res = e.run_cmd('play WHITE M13')
# res = e.run_cmd('play WHITE M14')
# res = e.run_cmd('play WHITE N13')
# res = e.run_cmd('play WHITE N14')
# print(res)
#
# res = e.run_cmd('play BLACK N12')
# print(res)
# #g.show_board()
#
# res = e.run_cmd('play BLACK P16')
# res = e.run_cmd('play BLACK P17')
# res = e.run_cmd('play BLACK P18')
# res = e.run_cmd('play BLACK P19')
# res = e.run_cmd('play BLACK Q16')
# res = e.run_cmd('play BLACK R16')
# res = e.run_cmd('play BLACK S16')
#
# res = e.run_cmd('play WHITE S18')
# res = e.run_cmd('play WHITE S17')
# res = e.run_cmd('play WHITE Q19')
# res = e.run_cmd('play WHITE Q18')
# res = e.run_cmd('play WHITE Q17')
# res = e.run_cmd('play WHITE R18')
# res = e.run_cmd('play WHITE R17')
# res = e.run_cmd('play BLACK S19')
# print(res)
# #g.show_board()
#
# res = e.run_cmd('play WHITE R19')
# g.show_board()
#
# res = e.run_cmd('play BLACK S19')
# print(res)
# g.show_board()
#
# res = e.run_cmd('play BLACK S19')
# print(res)
#
#
# res = e.run_cmd('play BLACK E17')
# res = e.run_cmd('play BLACK F16')
# res = e.run_cmd('play BLACK F18')
# res = e.run_cmd('play BLACK G17')
# res = e.run_cmd('play WHITE G16')
# res = e.run_cmd('play WHITE G18')
# res = e.run_cmd('play WHITE H17')
# g.show_board()
#
# res = e.run_cmd('play WHITE F17')
# g.show_board()
#
# res = e.run_cmd('play BLACK G17')
# print(res)
# g.show_board()
#
# res = e.run_cmd('play BLACK G19')
# res = e.run_cmd('play BLACK G17')
# g.show_board()



