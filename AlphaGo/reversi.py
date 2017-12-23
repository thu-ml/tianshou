import numpy as np
'''
Settings of the Reversi game.

(1, 1) is considered as the upper left corner of the board,
(size, 1) is the lower left
'''


class Reversi:
    def __init__(self, black=None, white=None):
        self.board = None  # 8 * 8 board with 1 for black, -1 for white and 0 for blank
        self.color = None  # 1 for black and -1 for white
        self.action = None   # number in 0~63
        self.winner = None
        self.black_win = None
        self.size = 8

    def _deflatten(self, idx):
        x = idx // self.size + 1
        y = idx % self.size + 1
        return (x, y)

    def _flatten(self, vertex):
        x, y = vertex
        if (x == 0) and (y == 0):
            return 64
        return (x - 1) * self.size + (y - 1)

    def get_board(self, board=None):
        self.board = board or np.zeros([8,8])
        self.board[3, 3] = -1
        self.board[4, 4] = -1
        self.board[3, 4] = 1
        self.board[4, 3] = 1
        return self.board

    def _find_correct_moves(self, is_next=False):
        moves = []
        if is_next:
            color = 0 - self.color
        else:
            color = self.color
        for i in range(64):
            x, y = self._deflatten(i)
            valid = self._is_valid(x - 1, y - 1, color)
            if valid:
                moves.append(i)
        return moves

    def _one_direction_valid(self, x, y, color):
        if (x >= 0) and (x < self.size):
            if (y >= 0) and (y < self.size):
                if self.board[x, y] == color:
                    return True
        return False

    def _is_valid(self, x, y, color):
        if self.board[x, y]:
            return False
        for x_direction in [-1, 0, 1]:
            for y_direction in [-1, 0, 1]:
                new_x = x
                new_y = y
                flag = 0
                while True:
                    new_x += x_direction
                    new_y += y_direction
                    if self._one_direction_valid(new_x, new_y, 0 - color):
                        flag = 1
                    else:
                        break
                if self._one_direction_valid(new_x, new_y, color) and flag:
                    return True
        return False

    def simulate_get_mask(self, state, action_set):
        history_boards, color = state
        self.board = np.reshape(history_boards[-1], (self.size, self.size))
        self.color = color
        valid_moves = self._find_correct_moves()
        print(valid_moves)
        if not len(valid_moves):
            invalid_action_mask = action_set[0:-1]
        else:
            invalid_action_mask = []
            for action in action_set:
                if action not in valid_moves:
                    invalid_action_mask.append(action)
        return invalid_action_mask

    def simulate_step_forward(self, state, action):
        self.board = state[0].copy()
        self.board = np.reshape(self.board, (self.size, self.size))
        self.color = state[1]
        self.action = action
        if self.action == 64:
            valid_moves = self._find_correct_moves(is_next=True)
            if not len(valid_moves):
                self._game_over()
                return None, self.winner * self.color
            else:
                return [self.board, 0 - self.color], 0
        self._step()
        return [self.board, 0 - self.color], 0

    def _game_over(self):
        black_num, white_num = self._number_of_black_and_white()
        self.black_win = black_num - white_num
        if self.black_win > 0:
            self.winner = 1
        elif self.black_win < 0:
            self.winner = -1
        else:
            self.winner = 0

    def _number_of_black_and_white(self):
        black_num = 0
        white_num = 0
        board_list = np.reshape(self.board, self.size ** 2)
        for i in range(len(board_list)):
            if board_list[i] == 1:
                black_num += 1
            elif board_list[i] == -1:
                white_num += 1
        return black_num, white_num

    def _step(self):
        if self.action < 0 or self.action > 63:
            raise ValueError("Action not in the range of [0,63]!")
        if self.action is None:
            raise ValueError("Action is None!")
        x, y = self._deflatten(self.action)
        valid = self._flip(x -1, y - 1)
        if not valid:
            raise ValueError("Illegal action!")

    def _flip(self, x, y):
        valid = 0
        self.board[x, y] = self.color
        for x_direction in [-1, 0, 1]:
            for y_direction in [-1, 0, 1]:
                new_x = x
                new_y = y
                flag = 0
                while True:
                    new_x += x_direction
                    new_y += y_direction
                    if self._one_direction_valid(new_x, new_y, 0 - self.color):
                        flag = 1
                    else:
                        break
                if self._one_direction_valid(new_x, new_y, self.color) and flag:
                    valid = 1
                    flip_x = x
                    flip_y = y
                    while True:
                        flip_x += x_direction
                        flip_y += y_direction
                        if self._one_direction_valid(flip_x, flip_y, 0 - self.color):
                            self.board[flip_x, flip_y] = self.color
                        else:
                            break
        if valid:
            return True
        else:
            return False

    def executor_do_move(self, history, latest_boards, board, color, vertex):
        self.board = np.reshape(board, (self.size, self.size))
        self.color = color
        self.action = self._flatten(vertex)
        if self.action == 64:
            valid_moves = self._find_correct_moves(is_next=True)
            if not len(valid_moves):
                return False
            else:
                return True
        else:
            self._step()
            return True

    def executor_get_score(self, board):
        self.board = board
        self._game_over()
        if self.black_win is not None:
            return self.black_win
        else:
            raise ValueError("Game not finished!")


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

