import numpy as np
import copy
'''
Settings of the Reversi game.

(1, 1) is considered as the upper left corner of the board,
(size, 1) is the lower left
'''


class Reversi:
    def __init__(self, **kwargs):
        self.size = kwargs['size']

    def _deflatten(self, idx):
        x = idx // self.size + 1
        y = idx % self.size + 1
        return (x, y)

    def _flatten(self, vertex):
        x, y = vertex
        if (x == 0) and (y == 0):
            return self.size ** 2
        return (x - 1) * self.size + (y - 1)

    def get_board(self):
        board = np.zeros([self.size, self.size], dtype=np.int32)
        board[self.size / 2 - 1, self.size / 2 - 1] = -1
        board[self.size / 2, self.size / 2] = -1
        board[self.size / 2 - 1, self.size / 2] = 1
        board[self.size / 2, self.size / 2 - 1] = 1
        return board

    def _find_correct_moves(self, board, color, is_next=False):
        moves = []
        if is_next:
            new_color = 0 - color
        else:
            new_color = color
        for i in range(self.size ** 2):
            x, y = self._deflatten(i)
            valid = self._is_valid(board, x - 1, y - 1, new_color)
            if valid:
                moves.append(i)
        return moves

    def _one_direction_valid(self, board, x, y, color):
        if (x >= 0) and (x < self.size):
            if (y >= 0) and (y < self.size):
                if board[x, y] == color:
                    return True
        return False

    def _is_valid(self, board, x, y, color):
        if board[x, y]:
            return False
        for x_direction in [-1, 0, 1]:
            for y_direction in [-1, 0, 1]:
                new_x = x
                new_y = y
                flag = 0
                while True:
                    new_x += x_direction
                    new_y += y_direction
                    if self._one_direction_valid(board, new_x, new_y, 0 - color):
                        flag = 1
                    else:
                        break
                if self._one_direction_valid(board, new_x, new_y, color) and flag:
                    return True
        return False

    def simulate_get_mask(self, state, action_set):
        history_boards, color = copy.deepcopy(state)
        board = copy.deepcopy(history_boards[-1])
        valid_moves = self._find_correct_moves(board, color)
        if not len(valid_moves):
            invalid_action_mask = action_set[0:-1]
        else:
            invalid_action_mask = []
            for action in action_set:
                if action not in valid_moves:
                    invalid_action_mask.append(action)
        return invalid_action_mask

    def simulate_step_forward(self, state, action):
        history_boards, color = copy.deepcopy(state)
        board = copy.deepcopy(history_boards[-1])
        if action == self.size ** 2:
            valid_moves = self._find_correct_moves(board, color, is_next=True)
            if not len(valid_moves):
                winner = self._get_winner(board)
                return None, winner * color
            else:
                return [history_boards, 0 - color], 0
        new_board = self._step(board, color, action)
        history_boards.append(new_board)
        return [history_boards, 0 - color], 0

    def _get_winner(self, board):
        black_num, white_num = self._number_of_black_and_white(board)
        black_win = black_num - white_num
        if black_win > 0:
            winner = 1
        elif black_win < 0:
            winner = -1
        else:
            winner = 0
        return winner

    def _number_of_black_and_white(self, board):
        black_num = 0
        white_num = 0
        board_list = np.reshape(board, self.size ** 2)
        for i in range(len(board_list)):
            if board_list[i] == 1:
                black_num += 1
            elif board_list[i] == -1:
                white_num += 1
        return black_num, white_num

    def _step(self, board, color, action):
        if action < 0 or action > self.size ** 2 - 1:
            raise ValueError("Action not in the range of [0,63]!")
        if action is None:
            raise ValueError("Action is None!")
        x, y = self._deflatten(action)
        new_board = self._flip(board, x - 1, y - 1, color)
        return new_board

    def _flip(self, board, x, y, color):
        valid = 0
        board[x, y] = color
        for x_direction in [-1, 0, 1]:
            for y_direction in [-1, 0, 1]:
                new_x = x
                new_y = y
                flag = 0
                while True:
                    new_x += x_direction
                    new_y += y_direction
                    if self._one_direction_valid(board, new_x, new_y, 0 - color):
                        flag = 1
                    else:
                        break
                if self._one_direction_valid(board, new_x, new_y, color) and flag:
                    valid = 1
                    flip_x = x
                    flip_y = y
                    while True:
                        flip_x += x_direction
                        flip_y += y_direction
                        if self._one_direction_valid(board, flip_x, flip_y, 0 - color):
                            board[flip_x, flip_y] = color
                        else:
                            break
        if valid:
            return board
        else:
            raise ValueError("Invalid action")

    def executor_do_move(self, history, latest_boards, board, color, vertex):
        board = np.reshape(board, (self.size, self.size))
        color = color
        action = self._flatten(vertex)
        if action == self.size ** 2:
            valid_moves = self._find_correct_moves(board, color, is_next=True)
            if not len(valid_moves):
                return False
            else:
                return True
        else:
            new_board = self._step(board, color, action)
            history.append(new_board)
            latest_boards.append(new_board)
            return True

    def executor_get_score(self, board):
        board = board
        winner = self._get_winner(board)
        return winner


if __name__ == "__main__":
    reversi = Reversi()
    # board = reversi.get_board()
    # print(board)
    # state, value = reversi.simulate_step_forward([board, -1], 20)
    # print(state[0])
    # print("board")
    # print(board)
    # r = reversi.executor_get_score(board)
    # print(r)

