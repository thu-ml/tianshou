from __future__ import print_function
import utils
import copy
import numpy as np
from collections import deque
import time
'''
Settings of the Go game.

(1, 1) is considered as the upper left corner of the board,
(size, 1) is the lower left
'''

NEIGHBOR_OFFSET = [[1, 0], [-1, 0], [0, -1], [0, 1]]
CORNER_OFFSET = [[-1, -1], [-1, 1], [1, 1], [1, -1]]

class Go:
    def __init__(self, **kwargs):
        self.size = kwargs['size']
        self.komi = kwargs['komi']

    def _flatten(self, vertex):
        x, y = vertex
        return (x - 1) * self.size + (y - 1)

    def _deflatten(self, idx):
        x = idx // self.size + 1
        y = idx % self.size + 1
        return (x, y)

    def _in_board(self, vertex):
        x, y = vertex
        if x < 1 or x > self.size: return False
        if y < 1 or y > self.size: return False
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

    def _neighbor_color(self, current_board, vertex, color):
        # return neighbors which are listed in different colors
        color_neighbor = []  # 1)neighbors in the same color
        reverse_color_neighbor = []  # 2)neighbors in the reverse color
        empty_neighbor = []  # 2)empty neighbors
        reverse_color = utils.BLACK if color == utils.WHITE else utils.WHITE
        for n in self._neighbor(vertex):
            if current_board[self._flatten(n)] == color:
                color_neighbor.append(self._flatten(n))
            elif current_board[self._flatten(n)] == utils.EMPTY:
                empty_neighbor.append(self._flatten(n))
            elif current_board[self._flatten(n)] == reverse_color:
                reverse_color_neighbor.append(self._flatten(n))
            else:
                raise ValueError("board have other positions excluding BLACK, WHITE and EMPTY")
        return color_neighbor, reverse_color_neighbor, empty_neighbor

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
        color = current_board[self._flatten(vertex)]
        chain = set()
        frontier = [vertex]
        has_liberty = False
        while frontier:
            current = frontier.pop()
            chain.add(current)
            for n in self._neighbor(current):
                if current_board[self._flatten(n)] == color and not n in chain:
                    frontier.append(n)
                if current_board[self._flatten(n)] == utils.EMPTY:
                    has_liberty = True
        return has_liberty, chain

    def _find_ancestor(self, group_ancestors, idx):
        r = idx
        while group_ancestors[r] != r:
            r = group_ancestors[r]
        group_ancestors[idx] = r
        return r

    def _is_suicide(self, current_board, group_ancestors, liberty, color, vertex):
        color_neighbor, reverse_color_neighbor, empty_neighbor = self._neighbor_color(current_board, vertex, color)
        if empty_neighbor:
            return False  # neighbors have empty spaces
        elif color_neighbor:  # neighbors have same color, they have liberties
            for idx in color_neighbor:
                if len(liberty[self._find_ancestor(group_ancestors, idx)]) > 1:
                    return False
        else:  # neighbors have reverse color, they have only one liberty
            for idx in reverse_color_neighbor:
                if len(liberty[self._find_ancestor(group_ancestors, idx)]) == 1:
                    return False
        return True

    def _process_board(self, current_board, color, vertex):
        nei = self._neighbor(vertex)
        for n in nei:
            if current_board[self._flatten(n)] == utils.another_color(color):
                has_liberty, group = self._find_group(current_board, n)
                if not has_liberty:
                    for b in group:
                        current_board[self._flatten(b)] = utils.EMPTY

    def _check_global_isomorphous(self, history_hashtable, current_board, color, vertex):
        repeat = False
        next_board = copy.deepcopy(current_board)
        next_board[self._flatten(vertex)] = color
        self._process_board(next_board, color, vertex)
        if tuple(next_board) in history_hashtable:
            repeat = True
        del next_board
        return repeat

    def _is_eye(self, current_board, color, vertex):
        # return is this position is an real eye of color
        color_neighbor, reverse_color_neighbor, empty_neighbor = self._neighbor_color(current_board, vertex, color)
        if reverse_color_neighbor or empty_neighbor:  # not an eye
            return False
        cor = self._corner(vertex)
        opponent_number = [current_board[self._flatten(c)] for c in cor].count(-color)
        opponent_propotion = float(opponent_number) / float(len(cor))
        # opponent_propotion<0.5 fake eye
        return True if opponent_propotion < 0.5 else False

    def _knowledge_prunning(self, current_board, color, vertex):
        #  forbid some stupid selfplay using human knowledge
        if self._is_eye(current_board, color, vertex):
            return False
            # forbid position on its own eye.
        return True

    def _action2vertex(self, action):
        if action == self.size ** 2:
            vertex = (0, 0)
        else:
            vertex = self._deflatten(action)
        return vertex

    def _rule_check(self, history_hashtable, current_board, group_ancestors, liberty, color, vertex, is_thinking=True):
        ### in board
        if not self._in_board(vertex):
            if not is_thinking:
                raise ValueError("Target point not in board, Current Board: {}, color: {}, vertex : {}".format(current_board, color, vertex))
            else:
                return False

        ### already have stone
        if not current_board[self._flatten(vertex)] == utils.EMPTY:
            if not is_thinking:
                raise ValueError("Target point already has a stone, Current Board: {}, color: {}, vertex : {}".format(current_board, color, vertex))
            else:
                return False

        ### check if it is suicide
        if self._is_suicide(current_board, group_ancestors, liberty, color, vertex):
            if not is_thinking:
                raise ValueError("Target point causes suicide, Current Board: {}, color: {}, vertex : {}".format(current_board, color, vertex))
            else:
                return False

        ### forbid global isomorphous
        if self._check_global_isomorphous(history_hashtable, current_board, color, vertex):
            if not is_thinking:
                raise ValueError("Target point causes global isomorphous, Current Board: {}, color: {}, vertex : {}".format(current_board, color, vertex))
            else:
                return False

        return True

    def _is_valid(self, state, action, history_hashtable, group_ancestors, liberty):
        history_boards, color = state
        vertex = self._action2vertex(action)
        current_board = history_boards[-1]

        if not self._rule_check(history_hashtable, current_board, group_ancestors, liberty, color, vertex):
            return False

        if not self._knowledge_prunning(current_board, color, vertex):
            return False
        return True

    def _get_groups(self, board):
        group_ancestors = {}  # key: idx, value: ancestor idx
        liberty = {}  # key: ancestor idx, value: set of liberty
        for idx, color in enumerate(board):
            if color and idx not in group_ancestors:
                # build group
                group_ancestors[idx] = idx
                color_neighbor, _, empty_neighbor = \
                    self._neighbor_color(board, self._deflatten(idx), color)
                liberty[idx] = set(empty_neighbor)
                group_list = copy.deepcopy(color_neighbor)
                while group_list:
                    add_idx = group_list.pop()
                    if add_idx not in group_ancestors:
                        group_ancestors[add_idx] = idx
                        color_neighbor_add, _, empty_neighbor_add = \
                            self._neighbor_color(board, self._deflatten(add_idx), color)
                        group_list += color_neighbor_add
                        liberty[idx] |= set(empty_neighbor_add)
        return group_ancestors, liberty

    def simulate_get_mask(self, state, action_set):
        # find all the invalid actions
        invalid_action_mask = []
        history_boards, color = state
        group_ancestors, liberty = self._get_groups(history_boards[-1])
        history_hashtable = set()
        for board in history_boards:
            history_hashtable.add(tuple(board))
        for action_candidate in action_set[:-1]:
            # go through all the actions excluding pass
            if not self._is_valid(state, action_candidate, history_hashtable, group_ancestors, liberty):
                invalid_action_mask.append(action_candidate)
        if len(invalid_action_mask) < len(action_set) - 1:
            invalid_action_mask.append(action_set[-1])
            # forbid pass, if we have other choices
            # TODO: In fact we should not do this. In some extreme cases, we should permit pass.
        del history_hashtable
        del group_ancestors
        del liberty
        # del stones
        return invalid_action_mask

    def _do_move(self, board, color, vertex):
        if vertex == utils.PASS:
            return board
        else:
            id_ = self._flatten(vertex)
            board[id_] = color
            return board

    def simulate_step_forward(self, state, action):
        # initialize the simulate_board from state
        history_boards, color = copy.deepcopy(state)
        if history_boards[-1] == history_boards[-2] and action is utils.PASS:
            return None, 2 * (float(self.simple_executor_get_score(history_boards[-1]) > 0)-0.5) * color
        else:
            vertex = self._action2vertex(action)
            new_board = self._do_move(copy.deepcopy(history_boards[-1]), color, vertex)
            history_boards.append(new_board)
            new_color = -color
            return [history_boards, new_color], 0

    def simulate_hashable_conversion(self, state):
        # since go is MDP, we only need the last board for hashing
        return tuple(state[0][-1])

    def _join_group(self, idx, idx_list, empty_neighbor, group_ancestors, liberty, stones):
        # idx joins its neighbors id_list
        # empty_neighbor: empty neighbors of idx
        color_ancestor = set()
        for color_idx in idx_list:
            color_ancestor.add(self._find_ancestor(group_ancestors, color_idx))
        joined_ancestor = color_ancestor.pop()
        liberty[joined_ancestor] |= set(empty_neighbor)
        stones[joined_ancestor].add(idx)
        group_ancestors[idx] = joined_ancestor
        # add other groups
        for color_idx in color_ancestor:
            liberty[joined_ancestor] |= liberty[color_idx]
            stones[joined_ancestor] |= stones[color_idx]
            del liberty[color_idx]
            for stone in stones[color_idx]:
                group_ancestors[stone] = joined_ancestor
            del stones[color_idx]
        liberty[joined_ancestor].remove(idx)

    def _add_captured_liberty(self, board, liberty, group_ancestors, stones):
        for captured_stone in stones:
            color_neighbor, reverse_color_neighbor, empty_neighbor = \
                self._neighbor_color(board, self._deflatten(captured_stone), board[captured_stone])
            assert not empty_neighbor  # make sure no empty spaces
            for reverse_color_idx in reverse_color_neighbor:
                reverse_color_idx_ancestor = self._find_ancestor(group_ancestors, reverse_color_idx)
                liberty[reverse_color_idx_ancestor].add(captured_stone)

    def _remove_liberty(self, idx, reverse_color_neighbor, current_board, group_ancestors, liberty, stones):
        # reverse_color_neighbor: stones near idx in the reverse color
        reverse_color_ancestor = set()
        for reverse_idx in reverse_color_neighbor:
            reverse_color_ancestor.add(self._find_ancestor(group_ancestors, reverse_idx))
        for reverse_color_ancestor_idx in reverse_color_ancestor:
            if len(liberty[reverse_color_ancestor_idx]) == 1:
                # capture this group if no liberty left
                self._add_captured_liberty(current_board, liberty, group_ancestors, stones[reverse_color_ancestor_idx])
                for captured_stone in stones[reverse_color_ancestor_idx]:
                    current_board[captured_stone] = utils.EMPTY
                    del group_ancestors[captured_stone]
                del liberty[reverse_color_ancestor_idx]
                del stones[reverse_color_ancestor_idx]
            else:
                # remove this liberty
                liberty[reverse_color_ancestor_idx].remove(idx)

    def executor_do_move(self, history, history_hashtable, latest_boards, current_board, group_ancestors, liberty, stones, color, vertex):
        #print("===")
        #print(color, vertex)
        #print(group_ancestors, liberty, stones)
        if not self._rule_check(history_hashtable, current_board, group_ancestors, liberty, color, vertex):
            return False
        idx = self._flatten(vertex)
        current_board[idx] = color
        color_neighbor, reverse_color_neighbor, empty_neighbor = self._neighbor_color(current_board, vertex, color)
        if color_neighbor:  # join nearby groups
            self._join_group(idx, color_neighbor, empty_neighbor, group_ancestors, liberty, stones)
        else:  # build a new group
            group_ancestors[idx] = idx
            liberty[idx] = set(empty_neighbor)
            stones[idx] = {idx}
        if reverse_color_neighbor:  # remove liberty for nearby reverse color
            self._remove_liberty(idx, reverse_color_neighbor, current_board, group_ancestors, liberty, stones)
        history.append(copy.deepcopy(current_board))
        latest_boards.append(copy.deepcopy(current_board))
        history_hashtable.add(copy.deepcopy(tuple(current_board)))
        return True

    def _find_empty(self, current_board):
        idx = [i for i,x in enumerate(current_board) if x == utils.EMPTY ][0]
        return self._deflatten(idx)

    def _find_boarder(self, current_board, vertex):
        _, group = self._find_group(current_board, vertex)
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

    def _predict_from_nearby(self, current_board, vertex, neighbor_step=3):
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
                color_estimate += current_board[self._flatten(neighbor_vertex)]
            if color_estimate > 0:
                return utils.BLACK
            elif color_estimate < 0:
                return utils.WHITE

    def executor_get_score(self, current_board):
        #return score from BLACK perspective.
        _board = copy.deepcopy(current_board)
        while utils.EMPTY in _board:
            vertex = self._find_empty(_board)
            boarder = self._find_boarder(_board, vertex)
            boarder_color = set(map(lambda v: _board[self._flatten(v)], boarder))
            if boarder_color == {utils.BLACK}:
                _board[self._flatten(vertex)] = utils.BLACK
            elif boarder_color == {utils.WHITE}:
                _board[self._flatten(vertex)] = utils.WHITE
            else:
                _board[self._flatten(vertex)] = self._predict_from_nearby(_board, vertex)
        score = 0
        for i in _board:
            if i == utils.BLACK:
                score += 1
            elif i == utils.WHITE:
                score -= 1
        score -= self.komi

        return score


    def simple_executor_get_score(self, current_board):
        '''
            can only be used for the empty group only have one single stone
            return score from BLACK perspective.
        '''
        score = 0
        for idx, color in enumerate(current_board):
            if color == utils.EMPTY:
                neighbors = self._neighbor(self._deflatten(idx))
                color = current_board[self._flatten(neighbors[0])]
            if color == utils.BLACK:
                score += 1
            elif color == utils.WHITE:
                score -= 1
        score -= self.komi
        return score


if __name__ == "__main__":
    go = Go(size=9, komi=3.75)
    endgame = [
        1, 0, 1, 0, 1, 1, -1, 0, -1,
        1, 1, 1, 1, 1, 1, -1, -1, -1,
        0, 1, 1, 1, 1, -1, 0, -1, 0,
        1, 1, 1, 1, 1, -1, -1, -1, -1,
        1, -1, 1, -1, 1, 1, -1, -1, -1,
        -1, -1, -1, -1, -1, 1, -1, 0, -1,
        1, 1, 1, -1, -1, -1, -1, -1, -1,
        1, 0, 1, 1, 1, 1, 1, -1, 0,
        1, 1, 0, 1, -1, -1, -1, -1, -1
    ]

    '''
    time0 = time.time()
    score = go.executor_get_score(endgame)
    time1 = time.time()
    print(score, time1 - time0)
    score = go.new_executor_get_score(endgame)
    time2 = time.time()
    print(score, time2 - time1)
    '''
    '''
    ### do unit test for Go class
    pure_test = [
        0, 1, 0, 1, 0, 1, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 0,
        1, 1, 1, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 0, 1, 1, 0, 0,
        1, 1, 1, 0, 1, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0
    ]

    pt_qry = [(1, 1), (1, 5), (3, 3), (4, 7), (7, 2), (8, 6)]
    pt_ans = [True, True, True, True, True, True]

    opponent_test = [
        0, 1, 0, 1, 0, 1, 0,-1, 1,
        1,-1, 0,-1, 1,-1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1,-1, 0, 1,-1, 1, 0, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 0,
        -1,1, 1, 0, 1, 1, 1, 0, 0,
        0, 1,-1, 0,-1,-1,-1, 0, 0,
        1, 0, 1, 0,-1, 0,-1, 0, 0,
        0, 1, 0, 0,-1,-1,-1, 0, 0
    ]
    ot_qry = [(1, 1), (1, 5), (2, 9), (5, 2), (5, 6), (8, 6), (8, 2)]
    ot_ans = [False, False, False, False, False, False, True]

    go = Go(size=9, komi=3.75)
    for i in range(6):
        print (go._is_eye(pure_test, utils.BLACK, pt_qry[i]))
    print("Test of pure eye\n")

    for i in range(7):
        print (go._is_eye(opponent_test, utils.BLACK, ot_qry[i]))
    print("Test of eye surrend by opponents\n")
    '''
