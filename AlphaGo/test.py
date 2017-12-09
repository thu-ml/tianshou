import sys
from game import Game
from engine import GTPEngine
# import utils
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/")
args = parser.parse_args()

game = Game(checkpoint_path=args.checkpoint_path)
engine = GTPEngine(game_obj=game, name='tianshou', version=0)

while not engine.disconnect:
    command = sys.stdin.readline()
    result = engine.run_cmd(command)
    sys.stdout.write(result)
    sys.stdout.flush()
