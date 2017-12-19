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
    def __init__(self, **kwargs):
        self.game = kwargs['game']
        self.simulate_board = [utils.EMPTY] * (self.game.size ** 2)
        self.simulate_latest_boards = deque(maxlen=8)

    def simulate_flatten(self, vertex):
        x, y = vertex
        return (x - 1) * self.game.size + (y - 1)

    def _find_group(self, start):
        color = self.simulate_board[self.simulate_flatten(start)]
        # print ("color : ", color)
        chain = set()
        frontier = [start]
        has_liberty = False
        while frontier:
            current = frontier.pop()
            # print ("current : ", current)
            chain.add(current)
            for n in self._neighbor(current):
                # print n, self._flatten(n), self.board[self._flatten(n)],
                if self.simulate_board[self.simulate_flatten(n)] == color and not n in chain:
                    frontier.append(n)
                if self.simulate_board[self.simulate_flatten(n)] == utils.EMPTY:
                    has_liberty = True
        return has_liberty, chain

    def _is_suicide(self, color, vertex):
        self.simulate_board[self.simulate_flatten(vertex)] = color # assume that we already take this move
        suicide = False

        has_liberty, group = self._find_group(vertex)
        if not has_liberty:
            suicide = True # no liberty, suicide
            for n in self._neighbor(vertex):
                if self.simulate_board[self.simulate_flatten(n)] == utils.another_color(color):
                    opponent_liberty, group = self._find_group(n)
                    if not opponent_liberty:
                        suicide = False # this move is able to take opponent's stone, not suicide

        self.simulate_board[self.simulate_flatten(vertex)] = utils.EMPTY # undo this move
        return suicide

    def _check_global_isomorphous(self, color, vertex):
        ##backup
        _board = copy.copy(self.simulate_board)
        self.simulate_board[self.simulate_flatten(vertex)] = color
        self._process_board(color, vertex)
        if self.simulate_board in self.game.history:
            res = True
        else:
            res = False

        self.simulate_board = _board
        return res

    def _in_board(self, vertex):
        x, y = vertex
        if x < 1 or x > self.game.size: return False
        if y < 1 or y > self.game.size: return False
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
            if self.simulate_board[self.simulate_flatten(n)] == utils.another_color(color):
                has_liberty, group = self._find_group(n)
                if not has_liberty:
                    for b in group:
                        self.simulate_board[self.simulate_flatten(b)] = utils.EMPTY

    def _is_eye(self, color, vertex):
        nei = self._neighbor(vertex)
        cor = self._corner(vertex)
        ncolor = {color == self.simulate_board[self.simulate_flatten(n)] for n in nei}
        if False in ncolor:
            # print "not all neighbors are in same color with us"
            return False
        _, group = self._find_group(nei[0])
        if set(nei) < group:
            # print "all neighbors are in same group and same color with us"
            return True
        else:
            opponent_number = [self.simulate_board[self.simulate_flatten(c)] for c in cor].count(-color)
            opponent_propotion = float(opponent_number) / float(len(cor))
            if opponent_propotion < 0.5:
                # print "few opponents, real eye"
                return True
            else:
                # print "many opponents, fake eye"
                return False

    def knowledge_prunning(self, color, vertex):
        ### check if it is an eye of yourself
        ### assumptions : notice that this judgement requires that the state is an endgame
        if self._is_eye(color, vertex):
            return False
        return True

    def simulate_is_valid(self, state, action):
        # State is the play board, the shape is [1, self.game.size, self.game.size, 17].
        # Action is an index
        # We need to transfer the (state, action) pair into (color, vertex) pair to simulate the move
        if action == self.game.size ** 2:
            vertex = (0, 0)
        else:
            vertex = (action / self.game.size + 1, action % self.game.size + 1)
        if state[0, 0, 0, -1] == utils.BLACK:
            color = utils.BLACK
        else:
            color = utils.WHITE
        self.simulate_latest_boards.clear()
        for i in range(8):
            self.simulate_latest_boards.append((state[:, :, :, i] - state[:, :, :, i + 8]).reshape(-1).tolist())
        self.simulate_board = copy.copy(self.simulate_latest_boards[-1])

        ### in board
        if not self._in_board(vertex):
            return False

        ### already have stone
        if not self.simulate_board[self.simulate_flatten(vertex)] == utils.EMPTY:
            # print(np.array(self.board).reshape(9, 9))
            # print(vertex)
            return False

        ### check if it is suicide
        if self._is_suicide(color, vertex):
            return False

        ### forbid global isomorphous
        if self._check_global_isomorphous(color, vertex):
            return False

        if not self.knowledge_prunning(color, vertex):
            return False

        return True

    def simulate_do_move(self, color, vertex):
        if vertex == utils.PASS:
            return True

        id_ = self.simulate_flatten(vertex)
        if self.simulate_board[id_] == utils.EMPTY:
            self.simulate_board[id_] = color
            return True
        else:
            return False

    def step_forward(self, state, action):
        if state[0, 0, 0, -1] == 1:
            color = utils.BLACK
        else:
            color = utils.WHITE
        if action == self.game.size ** 2:
            vertex = utils.PASS
        else:
            vertex = (action % self.game.size + 1, action / self.game.size + 1)
        # print(vertex)
        # print(self.board)
        self.simulate_board = (state[:, :, :, 7] - state[:, :, :, 15]).reshape(-1).tolist()
        self.simulate_do_move(color, vertex)
        new_state = np.concatenate(
            [state[:, :, :, 1:8], (np.array(self.simulate_board) == utils.BLACK).reshape(1, self.game.size, self.game.size, 1),
             state[:, :, :, 9:16], (np.array(self.simulate_board) == utils.WHITE).reshape(1, self.game.size, self.game.size, 1),
             np.array(1 - state[:, :, :, -1]).reshape(1, self.game.size, self.game.size, 1)],
            axis=3)
        return new_state, 0
