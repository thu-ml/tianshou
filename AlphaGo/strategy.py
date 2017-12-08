import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import numpy as np
import utils
import time
import copy
import network_small
import tensorflow as tf
from collections import deque
from tianshou.core.mcts.mcts import MCTS

DELTA = [[1, 0], [-1, 0], [0, -1], [0, 1]]
CORNER_OFFSET = [[-1, -1], [-1, 1], [1, 1], [1, -1]]


class GoEnv:
    def __init__(self, size=9, komi=6.5):
        self.size = size
        self.komi = komi
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

        ### avoid suicide
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

    def _corner(self, vertex):
        x, y = vertex
        corner = []
        for d in CORNER_OFFSET:
            _x = x + d[0]
            _y = y + d[1]
            if self._in_board((_x, _y)):
                corner.append((_x, _y))
        return corner

    def _process_board(self, color, vertex):
        nei = self._neighbor(vertex)
        for n in nei:
            if self.board[self._flatten(n)] == utils.another_color(color):
                can_kill, block = self._find_block(n, alive_break=True)
                if can_kill:
                    for b in block:
                        self.board[self._flatten(b)] = utils.EMPTY

    def _find_group(self, start):
        color = self.board[self._flatten(start)]
        # print ("color : ", color)
        chain = set()
        frontier = [start]
        while frontier:
            current = frontier.pop()
            # print ("current : ", current)
            chain.add(current)
            for n in self._neighbor(current):
                # print n, self._flatten(n), self.board[self._flatten(n)],
                if self.board[self._flatten(n)] == color and not n in chain:
                    frontier.append(n)
        return chain

    def _is_eye(self, color, vertex):
        nei = self._neighbor(vertex)
        cor = self._corner(vertex)
        ncolor = {color == self.board[self._flatten(n)] for n in nei}
        if False in ncolor:
            # print "not all neighbors are in same color with us"
            return False
        if set(nei) < self._find_group(nei[0]):
            # print "all neighbors are in same group and same color with us"
            return True
        else:
            opponent_number = [self.board[self._flatten(c)] for c in cor].count(-color)
            opponent_propotion = float(opponent_number) / float(len(cor))
            if opponent_propotion < 0.5:
                # print "few opponents, real eye"
                return True
            else:
                # print "many opponents, fake eye"
                return False

    # def is_valid(self, color, vertex):
    def is_valid(self, state, action):
        # state is the play board, the shape is [1, 9, 9, 17]
        if action == self.size * self.size:
            vertex = (0, 0)
        else:
            vertex = (action / self.size + 1, action % self.size + 1)
        if state[0, 0, 0, -1] == utils.BLACK:
            color = utils.BLACK
        else:
            color = utils.WHITE
        self.history.clear()
        for i in range(8):
            self.history.append((state[:, :, :, i] - state[:, :, :, i + 8]).reshape(-1).tolist())
        self.board = copy.copy(self.history[-1])
        ### in board
        if not self._in_board(vertex):
            return False

        ### already have stone
        if not self.board[self._flatten(vertex)] == utils.EMPTY:
            # print(np.array(self.board).reshape(9, 9))
            # print(vertex)
            return False

        ### check if it is qi
        if not self._is_qi(color, vertex):
            return False

        ### check if it is an eye of yourself
        ### assumptions : notice that this judgement requires that the state is an endgame
        if self._is_eye(color, vertex):
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
            return True
        else:
            return False

    def step_forward(self, state, action):
        if state[0, 0, 0, -1] == 1:
            color = utils.BLACK
        else:
            color = utils.WHITE
        if action == self.size ** 2:
            vertex = utils.PASS
        else:
            vertex = (action % self.size + 1, action / self.size + 1)
        # print(vertex)
        # print(self.board)
        self.board = (state[:, :, :, 7] - state[:, :, :, 15]).reshape(-1).tolist()
        self.do_move(color, vertex)
        new_state = np.concatenate(
            [state[:, :, :, 1:8], (np.array(self.board) == utils.BLACK).reshape(1, self.size, self.size, 1),
             state[:, :, :, 9:16], (np.array(self.board) == utils.WHITE).reshape(1, self.size, self.size, 1),
             np.array(1 - state[:, :, :, -1]).reshape(1, self.size, self.size, 1)],
            axis=3)
        return new_state, 0


class strategy(object):
    def __init__(self):
        self.simulator = GoEnv()
        self.net = network_small.Network()
        self.sess = self.net.forward()
        self.evaluator = lambda state: self.sess.run([tf.nn.softmax(self.net.p), self.net.v],
                                                     feed_dict={self.net.x: state, self.net.is_training: False})

    def data_process(self, history, color):
        state = np.zeros([1, self.simulator.size, self.simulator.size, 17])
        for i in range(8):
            state[0, :, :, i] = np.array(np.array(history[i]) == np.ones(self.simulator.size ** 2)).reshape(self.simulator.size, self.simulator.size)
            state[0, :, :, i + 8] = np.array(np.array(history[i]) == -np.ones(self.simulator.size ** 2)).reshape(self.simulator.size, self.simulator.size)
        if color == utils.BLACK:
            state[0, :, :, 16] = np.ones([self.simulator.size, self.simulator.size])
        if color == utils.WHITE:
            state[0, :, :, 16] = np.zeros([self.simulator.size, self.simulator.size])
        return state

    def gen_move(self, history, color):
        self.simulator.history = copy.copy(history)
        self.simulator.board = copy.copy(history[-1])
        state = self.data_process(self.simulator.history, color)
        mcts = MCTS(self.simulator, self.evaluator, state, self.simulator.size ** 2 + 1, inverse=True, max_step=100)
        temp = 1
        prob = mcts.root.N ** temp / np.sum(mcts.root.N ** temp)
        choice = np.random.choice(self.simulator.size ** 2 + 1, 1, p=prob).tolist()[0]
        if choice == self.simulator.size ** 2:
            move = utils.PASS
        else:
            move = (choice % self.simulator.size + 1, choice / self.simulator.size + 1)
        return move, prob
