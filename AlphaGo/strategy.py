import numpy as np
import utils
from collections import deque
from tianshou.core.mcts.mcts import MCTS


class GoEnv:
    def __init__(self, size=19, komi=6.5):
        self.size = size
        self.komi = 6.5
        self.board = [utils.EMPTY] * (self.size * self.size)
        self.history = deque(maxlen=8)

    def _flatten(self, vertex):
        x, y = vertex
        return (x - 1) * self.size + (y - 1)

    def do_move(self, color, vertex):
        if vertex == utils.PASS:
            return True

        id_ = self._flatten(vertex)
        if self.board[id_] == utils.EMPTY:
            self.board[id_] = color
            self.history.append(self.board)
            return True
        else:
            return False

    def step_forward(self, state, action):
        # print(state)
        if state[0, 0, 0, -1] == 1:
            color = 1
        else:
            color = -1
        if action == 361:
            vertex = (0, 0)
        else:
            vertex = (action / 19 + 1, action % 19)
        self.do_move(color, vertex)
        new_state = np.concatenate(
            [state[:, :, :, 1:8], (np.array(self.board) == 1).reshape(1, 19, 19, 1),
             state[:, :, :, 9:16], (np.array(self.board) == -1).reshape(1, 19, 19, 1),
             np.array(1 - state[:, :, :, -1]).reshape(1, 19, 19, 1)],
            axis=3)
        return new_state, 0


class strategy(object):
    def __init__(self, evaluator):
        self.simulator = GoEnv()
        self.evaluator = evaluator

    def data_process(self, history, color):
        state = np.zeros([1, 19, 19, 17])
        for i in range(8):
            state[0, :, :, i] = history[i] == 1
            state[0, :, :, i + 8] = history[i] == -1
        if color == 1:
            state[0, :, :, 16] = np.ones([19, 19])
        if color == -1:
            state[0, :, :, 16] = np.zeros([19, 19])
        return state

    def gen_move(self, history, color):
        self.simulator.history = history
        self.simulator.board = history[-1]
        state = self.data_process(history, color)
        prior = self.evaluator(state)[0]
        mcts = MCTS(self.simulator, self.evaluator, state, 362, prior, inverse=True, max_step=20)
        temp = 1
        p = mcts.root.N ** temp / np.sum(mcts.root.N ** temp)
        choice = np.random.choice(362, 1, p=p).tolist()[0]
        if choice == 361:
            move = (0, 0)
        else:
            move = (choice / 19 + 1, choice % 19 + 1)
        return move
