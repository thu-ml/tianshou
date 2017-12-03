import numpy as np
import utils
import time
import Network
import tensorflow as tf
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
    
    def _bfs(self, vertex, color, block, status, alive_break):
        block.append(vertex)
        status[self._flatten(vertex)] = True
        nei = self._neighbor(vertex)
        for n in nei:
            if not status[self._flatten(n)]:
                if self.board[self._flatten(n)] == color:
                    self._bfs(n, color, block, status, alive_break)

    def _find_block(self, vertex, alive_break=False):
        block = []
        status = [False] * (self.size * self.size)
        color = self.board[self._flatten(vertex)]
        self._bfs(vertex, color, block, status, alive_break)

        for b in block:
            for n in self._neighbor(b):
                if self.board[self._flatten(n)] == utils.EMPTY:
                    return False, block
        return True, block

    def _is_qi(self, color, vertex):
        nei = self._neighbor(vertex)
        for n in nei:
            if self.board[self._flatten(n)] == utils.EMPTY:
                return True

        self.board[self._flatten(vertex)] = color
        for n in nei:
            if self.board[self._flatten(n)] == utils.another_color(color):
                can_kill, block = self._find_block(n)
                if can_kill:
                    self.board[self._flatten(vertex)] = utils.EMPTY
                    return True

        ### can not suicide
        can_kill, block = self._find_block(vertex)
        if can_kill:
            self.board[self._flatten(vertex)] = utils.EMPTY
            return False

        self.board[self._flatten(vertex)] = utils.EMPTY
        return True

    def _check_global_isomorphous(self, color, vertex):
        ##backup
        _board = copy.copy(self.board)
        self.board[self._flatten(vertex)] = color
        self._process_board(color, vertex)
        if self.board in self.history:
            res = True
        else:
            res = False

        self.board = _board
        return res

    def _in_board(self, vertex):
        x, y = vertex
        if x < 1 or x > self.size: return False
        if y < 1 or y > self.size: return False
        return True

    def _neighbor(self, vertex):
        x, y = vertex
        nei = []
        for d in DELTA:
            _x = x + d[0]
            _y = y + d[1]
            if self._in_board((_x, _y)):
                nei.append((_x, _y))
        return nei

    def _process_board(self, color, vertex):
        nei = self._neighbor(vertex)
        for n in nei:
            if self.board[self._flatten(n)] == utils.another_color(color):
                can_kill, block = self._find_block(n, alive_break=True)
                if can_kill:
                    for b in block:
                        self.board[self._flatten(b)] = utils.EMPTY

    def is_valid(self, color, vertex):
        ### in board
        if not self._in_board(vertex):
            return False

        ### already have stone
        if not self.board[self._flatten(vertex)] == utils.EMPTY:
            return False

        ### check if it is qi
        if not self._is_qi(color, vertex):
            return False

        if self._check_global_isomorphous(color, vertex):
            return False

        return True

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
    def __init__(self):
        self.simulator = GoEnv()
        self.net = Network.Network()
        self.sess = self.net.forward()
        self.evaluator = lambda state: self.sess.run([tf.nn.softmax(self.net.p), self.net.v],
                                                     feed_dict={self.net.x: state, self.net.is_training: False})

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
        mcts = MCTS(self.simulator, self.evaluator, state, 362, prior, inverse=True, max_step=100)
        temp = 1
        p = mcts.root.N ** temp / np.sum(mcts.root.N ** temp)
        choice = np.random.choice(362, 1, p=p).tolist()[0]
        if choice == 361:
            move = (0, 0)
        else:
            move = (choice / 19 + 1, choice % 19 + 1)
        return move
