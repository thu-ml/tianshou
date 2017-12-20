from __future__ import print_function
import utils
import copy
import numpy as np
from collections import deque

'''
Settings of the Go game.

(1, 1) is considered as the upper left corner of the board,
(size, 1) is the lower left
'''

NEIGHBOR_OFFSET = [[1, 0], [-1, 0], [0, -1], [0, 1]]
CORNER_OFFSET = [[-1, -1], [-1, 1], [1, 1], [1, -1]]

class Go:
    def __init__(self, **kwargs):
        self.game = kwargs['game']

    def _in_board(self, vertex):
        x, y = vertex
        if x < 1 or x > self.game.size: return False
        if y < 1 or y > self.game.size: return False
        return True

    def _neighbor(self, vertex):
        x, y = vertex
        nei = []
        for d in NEIGHBOR_OFFSET:
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

    def _find_group(self, current_board, vertex):
        color = current_board[self.game._flatten(vertex)]
        # print ("color : ", color)
        chain = set()
        frontier = [vertex]
        has_liberty = False
        while frontier:
            current = frontier.pop()
            # print ("current : ", current)
            chain.add(current)
            for n in self._neighbor(current):
                if current_board[self.game._flatten(n)] == color and not n in chain:
                    frontier.append(n)
                if current_board[self.game._flatten(n)] == utils.EMPTY:
                    has_liberty = True
        return has_liberty, chain

    def _is_suicide(self, current_board, color, vertex):
        current_board[self.game._flatten(vertex)] = color # assume that we already take this move
        suicide = False

        has_liberty, group = self._find_group(current_board, vertex)
        if not has_liberty:
            suicide = True # no liberty, suicide
            for n in self._neighbor(vertex):
                if current_board[self.game._flatten(n)] == utils.another_color(color):
                    opponent_liberty, group = self._find_group(current_board, n)
                    if not opponent_liberty:
                        suicide = False # this move is able to take opponent's stone, not suicide

        current_board[self.game._flatten(vertex)] = utils.EMPTY # undo this move
        return suicide

    def _process_board(self, current_board, color, vertex):
        nei = self._neighbor(vertex)
        for n in nei:
            if current_board[self.game._flatten(n)] == utils.another_color(color):
                has_liberty, group = self._find_group(current_board, n)
                if not has_liberty:
                    for b in group:
                        current_board[self.game._flatten(b)] = utils.EMPTY

    def _check_global_isomorphous(self, history_boards, current_board, color, vertex):
        repeat = False
        next_board = copy.copy(current_board)
        next_board[self.game._flatten(vertex)] = color
        self._process_board(next_board, color, vertex)
        if next_board in history_boards:
            repeat = True
        return repeat

    def _is_eye(self, current_board, color, vertex):
        nei = self._neighbor(vertex)
        cor = self._corner(vertex)
        ncolor = {color == current_board[self.game._flatten(n)] for n in nei}
        if False in ncolor:
            # print "not all neighbors are in same color with us"
            return False
        _, group = self._find_group(current_board, nei[0])
        if set(nei) < group:
            # print "all neighbors are in same group and same color with us"
            return True
        else:
            opponent_number = [current_board[self.game._flatten(c)] for c in cor].count(-color)
            opponent_propotion = float(opponent_number) / float(len(cor))
            if opponent_propotion < 0.5:
                # print "few opponents, real eye"
                return True
            else:
                # print "many opponents, fake eye"
                return False

    def _knowledge_prunning(self, current_board, color, vertex):
        #  forbid some stupid selfplay using human knowledge
        if self._is_eye(current_board, color, vertex):
            return False
            # forbid position on its own eye.
        #if self._is_game_finish(current_board, color) and vertex == utils.PASS
        #    return False
        # forbid pass if the game is not finished.
        return True


    def _is_game_finished(self, current_board, color):
        '''
        for each empty position, if it has both BLACK and WHITE neighbors, the game is still not finished
        :return: return the game is finished
        '''
        board = copy.deepcopy(current_board)
        empty_idx = [i for i, x in enumerate(board) if x == utils.EMPTY]  # find all empty idx
        for idx in empty_idx:
            neighbor_idx = self._neighbor(self.game.deflatten(idx))
        if len(neighbor_idx) > 1:
            first_idx = neighbor_idx[0]
        for other_idx in neighbor_idx[1:]:
            if self.game.board[self.game.flatten(other_idx)] != self.game.board[self.game.flatten(first_idx)]:
                return False

        return True

    def _action2vertex(self, action):
        if action == self.game.size ** 2:
            vertex = (0, 0)
        else:
            vertex = self.game._deflatten(action)
        return vertex

    def _is_valid(self, history_boards, current_board, color, vertex):
        ### in board
        if not self._in_board(vertex):
            return False

        ### already have stone
        if not current_board[self.game._flatten(vertex)] == utils.EMPTY:
            return False

        ### check if it is suicide
        if self._is_suicide(current_board, color, vertex):
            return False

        ### forbid global isomorphous
        if self._check_global_isomorphous(history_boards, current_board, color, vertex):
            return False

        return True

    def simulate_is_valid(self, state, action):
        history_boards, color = state
        vertex = self._action2vertex(action)
        current_board = history_boards[-1]

        if not self._is_valid(history_boards, current_board, color, vertex):
            return False

        if not self._knowledge_prunning(current_board, color, vertex):
            return False
        return True

    def simulate_is_valid_list(self, state, action_set):
        # find all the invalid actions
        invalid_action_list = []
        for action_candidate in action_set[:-1]:
            # go through all the actions excluding pass
            if not self.simulate_is_valid(state, action_candidate):
                invalid_action_list.append(action_candidate)
        if len(invalid_action_list) < len(action_set) - 1:
            invalid_action_list.append(action_set[-1])
            # forbid pass, if we have other choices
        return invalid_action_list

    def _do_move(self, board, color, vertex):
        if vertex == utils.PASS:
            return board
        else:
            id_ = self.game._flatten(vertex)
            board[id_] = color
            return board

    def simulate_step_forward(self, state, action):
        # initialize the simulate_board from state
        history_boards, color = state
        vertex = self._action2vertex(action)
        new_board = self._do_move(copy.copy(history_boards[-1]), color, vertex)
        history_boards.append(new_board)
        new_color = -color
        return [history_boards, new_color], 0

    def executor_do_move(self, color, vertex):
        if not self._is_valid(self.game.history, self.game.board, color, vertex):
            return False
        self.game.board[self.game._flatten(vertex)] = color
        self._process_board(self.game.board, color, vertex)
        self.game.history.append(copy.copy(self.game.board))
        self.game.latest_boards.append(copy.copy(self.game.board))
        return True

    def _find_empty(self):
        idx = [i for i,x in enumerate(self.game.board) if x == utils.EMPTY ][0]
        return self.game._deflatten(idx)

    def _find_boarder(self, vertex):
        _, group = self._find_group(self.game.board, vertex)
        border = []
        for b in group:
            for n in self._neighbor(b):
                if not (n in group):
                    border.append(n)
        return border

    def _add_nearby_stones(self, neighbor_vertex_set, start_vertex_x, start_vertex_y, x_diff, y_diff, num_step):
        '''
        add the nearby stones around the input vertex
        :param neighbor_vertex_set: input list
        :param start_vertex_x: x axis of the input vertex
        :param start_vertex_y: y axis of the input vertex
        :param x_diff: add x axis
        :param y_diff: add y axis
        :param num_step: number of steps to be added
        :return:
        '''
        for step in xrange(num_step):
            new_neighbor_vertex = (start_vertex_x, start_vertex_y)
            if self._in_board(new_neighbor_vertex):
                neighbor_vertex_set.append((start_vertex_x, start_vertex_y))
            start_vertex_x += x_diff
            start_vertex_y += y_diff

    def _predict_from_nearby(self, vertex, neighbor_step=3):
        '''
        step: the nearby 3 steps is considered
        :vertex: position to be estimated
        :neighbor_step: how many steps nearby
        :return: the nearby positions of the input position
            currently the nearby 3*3 grid is returned, altogether 4*8 points involved
        '''
        for step in range(1, neighbor_step + 1): # check the stones within the steps in range
            neighbor_vertex_set = []
            self._add_nearby_stones(neighbor_vertex_set, vertex[0] - step, vertex[1], 1, 1, neighbor_step)
            self._add_nearby_stones(neighbor_vertex_set, vertex[0], vertex[1] + step, 1, -1, neighbor_step)
            self._add_nearby_stones(neighbor_vertex_set, vertex[0] + step, vertex[1], -1, -1, neighbor_step)
            self._add_nearby_stones(neighbor_vertex_set, vertex[0], vertex[1] -  step, -1, 1, neighbor_step)
            color_estimate = 0
            for neighbor_vertex in neighbor_vertex_set:
                color_estimate += self.game.board[self.game._flatten(neighbor_vertex)]
            if color_estimate > 0:
                return utils.BLACK
            elif color_estimate < 0:
                return utils.WHITE

    def executor_get_score(self, is_unknown_estimation=False):
        '''
            is_unknown_estimation: whether use nearby stone to predict the unknown
            return score from BLACK perspective.
        '''
        _board = copy.copy(self.game.board)
        while utils.EMPTY in self.game.board:
            vertex = self._find_empty()
            boarder = self._find_boarder(vertex)
            boarder_color = set(map(lambda v: self.game.board[self.game._flatten(v)], boarder))
            if boarder_color == {utils.BLACK}:
                self.game.board[self.game._flatten(vertex)] = utils.BLACK
            elif boarder_color == {utils.WHITE}:
                self.game.board[self.game._flatten(vertex)] = utils.WHITE
            elif is_unknown_estimation:
                self.game.board[self.game._flatten(vertex)] = self._predict_from_nearby(vertex)
            else:
                self.game.board[self.game._flatten(vertex)] =utils.UNKNOWN
        score = 0
        for i in self.game.board:
            if i == utils.BLACK:
                score += 1
            elif i == utils.WHITE:
                score -= 1
        score -= self.game.komi

        self.game.board = _board
        return score

