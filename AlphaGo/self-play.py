from game import Game
from engine import GTPEngine

g = Game()


g.show_board()
e = GTPEngine(game_obj=g)

num = 0
black_pass = False
white_pass = False
while not (black_pass and white_pass):
    if num % 2 == 0:
        res=e.run_cmd("genmove BLACK")[0]
        num += 1
        print(res)
        if res == (0,0):
            black_pass = True
    else:
        res = e.run_cmd("genmove WHITE")[0]
        num += 1
        print(res)
        if res == (0, 0):
            white_pass = True
    g.show_board()
