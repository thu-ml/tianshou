#!/usr/bin/env python
import numpy as np
import copy


class ZOTree:
    def __init__(self, size):
        self.size = size
        self.depth = self.size * 2

    def simulate_step_forward(self, state, action):
        self._check_state(state)
        seq, color = copy.deepcopy(state)
        if len(seq) == self.depth:
            winner = self.executor_get_reward(state)
            return None, color * winner
        else:
            seq.append(int(action))
            return [seq, 0 - color], 0

    def simulate_hashable_conversion(self, state):
        self._check_state(state)
        # since go is MDP, we only need the last board for hashing
        return tuple(state[0])

    def executor_get_reward(self, state):
        self._check_state(state)
        seq = np.array(state[0], dtype='int16')
        length = len(seq)
        if length != self.depth:
            raise ValueError("The game is not terminated!")
        ones = 0
        zeros = 0
        for i in range(len(seq)):
            if seq[i] == 0:
                zeros += 1
            if seq[i] == 1:
                ones += 1
        result = ones - zeros
        if result > 0:
            winner = 1
        elif result < 0:
            winner = -1
        else:
            winner = 0
        return winner

    def executor_do_move(self, state, action):
        self._check_state(state)
        seq, color = state
        if len(seq) == self.depth:
            return False
        else:
            seq.append(int(action))
            if len(seq) == self.depth:
                return False
            return True

    def v_value(self, state):
        self._check_state(state)
        seq, color = state
        ones = 0
        zeros = 0
        for i in range(len(seq)):
            if seq[i] == 0:
                zeros += 1
            if seq[i] == 1:
                ones += 1
        choosen_result = ones - zeros
        if color == 1:
            if choosen_result > 0:
                return 1
            elif choosen_result < 0:
                return -1
            else:
                return 0
        elif color == -1:
            if choosen_result > 1:
                return 1
            elif choosen_result < 1:
                return -1
            else:
                return 0
        else:
            raise ValueError("Wrong color")

    def _check_state(self, state):
        seq, color = state
        if color == 1:
            if len(seq) % 2:
                raise ValueError("Color is 1 but the length of seq is odd!")
        elif color == -1:
            if not len(seq) % 2:
                raise ValueError("Color is -1 but the length of seq is even!")
        else:
            raise ValueError("Wrong color!")

if __name__ == "__main__":
    size = 2
    game = ZOTree(size)
    seq = [1, -1, 1, 1]
    result = game.executor_do_move([seq, 1], 1)
    print(result)
    print(seq)