import sys
from game import Game
from engine import GTPEngine
# import utils

game = Game()
engine = GTPEngine(game_obj=game, name='tianshou')
cmd = raw_input

while not engine.disconnect:
    command = cmd()
    result = engine.run_cmd(command)
    sys.stdout.write(result)
    sys.stdout.flush()
