from game import Game
from engine import GTPEngine
import re

g = Game()


g.show_board()
e = GTPEngine(game_obj=g)

num = 0
black_pass = False
white_pass = False
while not (black_pass and white_pass):
    if num % 2 == 0:
        res = e.run_cmd(str(num) + " genmove BLACK")
        num += 1
        print(res)
        if re.search("pass", res) is not None:
            black_pass = True
        else:
            black_pass = False
    else:
        res = e.run_cmd(str(num) + " genmove WHITE")
        num += 1
        print(res)
        if re.search("pass", res) is not None:
            white_pass = True
        else:
            white_pass = False
    g.show_board()
