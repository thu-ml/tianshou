from __future__ import print_function
import numpy as np

'''
Settings of the Go game.

(1, 1) is considered as the upper left corner of the board,
(size, 1) is the lower left
'''


def find_correct_moves(own, enemy):
    """return legal moves"""
    left_right_mask = 0x7e7e7e7e7e7e7e7e  # Both most left-right edge are 0, else 1
    top_bottom_mask = 0x00ffffffffffff00  # Both most top-bottom edge are 0, else 1
    mask = left_right_mask & top_bottom_mask
    mobility = 0
    mobility |= search_offset_left(own, enemy, left_right_mask, 1)  # Left
    mobility |= search_offset_left(own, enemy, mask, 9)  # Left Top
    mobility |= search_offset_left(own, enemy, top_bottom_mask, 8)  # Top
    mobility |= search_offset_left(own, enemy, mask, 7)  # Top Right
    mobility |= search_offset_right(own, enemy, left_right_mask, 1)  # Right
    mobility |= search_offset_right(own, enemy, mask, 9)  # Bottom Right
    mobility |= search_offset_right(own, enemy, top_bottom_mask, 8)  # Bottom
    mobility |= search_offset_right(own, enemy, mask, 7)  # Left bottom
    return mobility

def calc_flip(pos, own, enemy):
    """return flip stones of enemy by bitboard when I place stone at pos.

    :param pos: 0~63
    :param own: bitboard (0=top left, 63=bottom right)
    :param enemy: bitboard
    :return: flip stones of enemy when I place stone at pos.
    """
    f1 = _calc_flip_half(pos, own, enemy)
    f2 = _calc_flip_half(63 - pos, rotate180(own), rotate180(enemy))
    return f1 | rotate180(f2)


def _calc_flip_half(pos, own, enemy):
    el = [enemy, enemy & 0x7e7e7e7e7e7e7e7e, enemy & 0x7e7e7e7e7e7e7e7e, enemy & 0x7e7e7e7e7e7e7e7e]
    masks = [0x0101010101010100, 0x00000000000000fe, 0x0002040810204080, 0x8040201008040200]
    masks = [b64(m << pos) for m in masks]
    flipped = 0
    for e, mask in zip(el, masks):
        outflank = mask & ((e | ~mask) + 1) & own
        flipped |= (outflank - (outflank != 0)) & mask
    return flipped


def search_offset_left(own, enemy, mask, offset):
    e = enemy & mask
    blank = ~(own | enemy)
    t = e & (own >> offset)
    t |= e & (t >> offset)
    t |= e & (t >> offset)
    t |= e & (t >> offset)
    t |= e & (t >> offset)
    t |= e & (t >> offset)  # Up to six stones can be turned at once
    return blank & (t >> offset)  # Only the blank squares can be started


def search_offset_right(own, enemy, mask, offset):
    e = enemy & mask
    blank = ~(own | enemy)
    t = e & (own << offset)
    t |= e & (t << offset)
    t |= e & (t << offset)
    t |= e & (t << offset)
    t |= e & (t << offset)
    t |= e & (t << offset)  # Up to six stones can be turned at once
    return blank & (t << offset)  # Only the blank squares can be started


def flip_vertical(x):
    k1 = 0x00FF00FF00FF00FF
    k2 = 0x0000FFFF0000FFFF
    x = ((x >> 8) & k1) | ((x & k1) << 8)
    x = ((x >> 16) & k2) | ((x & k2) << 16)
    x = (x >> 32) | b64(x << 32)
    return x


def b64(x):
    return x & 0xFFFFFFFFFFFFFFFF


def bit_count(x):
    return bin(x).count('1')


def bit_to_array(x, size):
    """bit_to_array(0b0010, 4) -> array([0, 1, 0, 0])"""
    return np.array(list(reversed((("0" * size) + bin(x)[2:])[-size:])), dtype=np.uint8)


def flip_diag_a1h8(x):
    k1 = 0x5500550055005500
    k2 = 0x3333000033330000
    k4 = 0x0f0f0f0f00000000
    t = k4 & (x ^ b64(x << 28))
    x ^= t ^ (t >> 28)
    t = k2 & (x ^ b64(x << 14))
    x ^= t ^ (t >> 14)
    t = k1 & (x ^ b64(x << 7))
    x ^= t ^ (t >> 7)
    return x


def rotate90(x):
    return flip_diag_a1h8(flip_vertical(x))


def rotate180(x):
    return rotate90(rotate90(x))


class Reversi:
    def __init__(self, black=None, white=None):
        self.black = black or (0b00001000 << 24 | 0b00010000 << 32)
        self.white = white or (0b00010000 << 24 | 0b00001000 << 32)
        self.board = None  # 8 * 8 board with 1 for black, -1 for white and 0 for blank
        self.color = None  # 1 for black and -1 for white
        self.action = None   # number in 0~63
        # self.winner = None
        self.black_win = None

    def get_board(self, black=None, white=None):
        self.black = black or (0b00001000 << 24 | 0b00010000 << 32)
        self.white = white or (0b00010000 << 24 | 0b00001000 << 32)
        self.board = self.bitboard2board() 	
        return self.board

    def simulate_get_mask(self, state, action_set):
        history_boards, color = state
        board = history_boards[-1]
        self.board = board
        self.color = color
        self.board2bitboard()
        own, enemy = self.get_own_and_enemy()
        mobility = find_correct_moves(own, enemy)
        valid_moves = bit_to_array(mobility, 64)
        valid_moves = np.argwhere(valid_moves)
        valid_moves = list(np.reshape(valid_moves, len(valid_moves)))
        # TODO it seems that the pass move is not considered
        invalid_action_mask = []
        for action in action_set:
            if action not in valid_moves:
                invalid_action_mask.append(action)
        return invalid_action_mask

    def simulate_step_forward(self, state, action):
        self.board = state[0]
        self.color = state[1]
        self.board2bitboard()
        self.action = action
        step_forward = self.step()
        if step_forward:
            new_board = self.bitboard2board()
            return [new_board, 0 - self.color], 0

    def executor_do_move(self, board, color, vertex):
        self.board = board
        self.color = color
        self.board2bitboard()
        self.vertex2action(vertex)
        step_forward = self.step()
        if step_forward:
            new_board = self.bitboard2board()
        for i in range(64):
        	board[i] = new_board[i]

    def executor_get_score(self, board):
        self.board = board
        self._game_over()
        if self.black_win is not None:
            return self.black_win
        else:
            ValueError("Game not finished!")

    def board2bitboard(self):
        count = 1
        if self.board is None:
            ValueError("None board!")
        self.black = 0
        self.white = 0
        for i in range(64):
            if self.board[i] == 1:
                self.black |= count
            elif self.board[i] == -1:
                self.white |= count
            count *= 2

    def vertex2action(self, vertex):
        x, y = vertex
        if x == 0 and y == 0:
            self.action = None
        else:
            self.action = 8 * (x - 1) + y - 1

    def bitboard2board(self):
        board = []
        black = bit_to_array(self.black, 64)
        white = bit_to_array(self.white, 64)
        for i in range(64):
            if black[i]:
                board.append(1)
            elif white[i]:
                board.append(-1)
            else:
                board.append(0)
        return board

    def step(self):
        if self.action < 0 or self.action > 63:
            ValueError("Wrong action!")
        if self.action is None:
            return False

        own, enemy = self.get_own_and_enemy()

        flipped = calc_flip(self.action, own, enemy)
        if bit_count(flipped) == 0:
            self.illegal_move_to_lose(self.action)
            return False
        own ^= flipped
        own |= 1 << self.action
        enemy ^= flipped

        self.set_own_and_enemy(own, enemy)
        return True

    def _game_over(self):
        # self.done = True
        '''
        if self.winner is None:
            black_num, white_num = self.number_of_black_and_white
            if black_num > white_num:
                self.winner = 1
            elif black_num < white_num:
                self.winner = -1
            else:
                self.winner = 0
        '''
        if self.black_win is None:
        	black_num, white_num = self.number_of_black_and_white
        	self.black_win = black_num - white_num

    def illegal_move_to_lose(self, action):
        self._game_over()

    def get_own_and_enemy(self):
        if self.color == 1:
            own, enemy = self.black, self.white
        elif self.color == -1:
            own, enemy = self.white, self.black
        else:
            own, enemy = None, None
        return own, enemy

    def set_own_and_enemy(self, own, enemy):
        if self.color == 1:
            self.black, self.white = own, enemy
        else:
            self.white, self.black = own, enemy

    @property
    def number_of_black_and_white(self):
        return bit_count(self.black), bit_count(self.white)
