from game import Game
from engine import GTPEngine
import utils



g = Game()
e = GTPEngine(game_obj = g)

res = e.run_cmd('8 genmove BLACK')
print(res)
