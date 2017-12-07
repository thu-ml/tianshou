from game import Game
from engine import GTPEngine
import re

g = Game()
pattern = "[A-Z]{1}[0-9]{1}"

g.show_board()
e = GTPEngine(game_obj=g)

num = 0
black_pass = False
white_pass = False
while not (black_pass and white_pass):
    if num % 2 == 0:
        res = e.run_cmd(str(num) + " genmove BLACK")
        num += 1
        # print(res)
        match = re.search(pattern, res)
        if match is not None:
            print(match.group())
        else:
            print("pass")
        if re.search("pass", res) is not None:
            black_pass = True
        else:
            black_pass = False
    else:
        res = e.run_cmd(str(num) + " genmove WHITE")
        num += 1
        match = re.search(pattern, res)
        if match is not None:
            print(match.group())
        else:
            print("pass")
        if re.search("pass", res) is not None:
            white_pass = True
        else:
            white_pass = False
    g.show_board()
