from game import Game
from engine import GTPEngine
import re
import numpy as np
import os
from collections import deque
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str, default='./part1')
args = parser.parse_args()

if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

game = Game()
engine = GTPEngine(game_obj=game)
history = deque(maxlen=8)
for i in range(8):
    history.append(game.board)
state = []
prob = []
winner = []
pattern = "[A-Z]{1}[0-9]{1}"
game.show_board()


def history2state(history, color):
    state = np.zeros([1, game.size, game.size, 17])
    for i in range(8):
        state[0, :, :, i] = np.array(np.array(history[i]) == np.ones(game.size ** 2)).reshape(game.size, game.size)
        state[0, :, :, i + 8] = np.array(np.array(history[i]) == -np.ones(game.size ** 2)).reshape(game.size, game.size)
    if color == utils.BLACK:
        state[0, :, :, 16] = np.ones([game.size, game.size])
    if color == utils.WHITE:
        state[0, :, :, 16] = np.zeros([game.size, game.size])
    return state


num = 0
game_num = 0
black_pass = False
white_pass = False
while True:
    print("Start game {}".format(game_num))
    while not (black_pass and white_pass) and num < game.size ** 2 * 2:
        if num % 2 == 0:
            color = utils.BLACK
            new_state = history2state(history, color)
            state.append(new_state)
            result = engine.run_cmd(str(num) + " genmove BLACK")
            num += 1
            match = re.search(pattern, result)
            if match is not None:
                print(match.group())
            else:
                print("pass")
            if re.search("pass", result) is not None:
                black_pass = True
            else:
                black_pass = False
        else:
            color = utils.WHITE
            new_state = history2state(history, color)
            state.append(new_state)
            result = engine.run_cmd(str(num) + " genmove WHITE")
            num += 1
            match = re.search(pattern, result)
            if match is not None:
                print(match.group())
            else:
                print("pass")
            if re.search("pass", result) is not None:
                white_pass = True
            else:
                white_pass = False
        game.show_board()
        prob.append(np.array(game.prob).reshape(-1, game.size ** 2 + 1))
    print("Finished")
    print("\n")
    score = game.executor.get_score(True)
    if score > 0:
        winner = utils.BLACK
    else:
        winner = utils.WHITE
    state = np.concatenate(state, axis=0)
    prob = np.concatenate(prob, axis=0)
    winner = np.ones([num, 1]) * winner
    assert state.shape[0] == prob.shape[0]
    assert state.shape[0] == winner.shape[0]
    np.savez(args.result_path + "/game" + str(game_num), state=state, prob=prob, winner=winner)
    state = []
    prob = []
    winner = []
    num = 0
    black_pass = False
    white_pass = False
    engine.run_cmd(str(num) + " clear_board")
    history.clear()
    for _ in range(8):
        history.append(game.board)
    game_num += 1
