from functools import partial
from typing import Optional, Tuple

import gym
import numpy as np

from tianshou.env import MultiAgentEnv


class TicTacToeEnv(MultiAgentEnv):
    """This is a simple implementation of the Tic-Tac-Toe game, where two
    agents play against each other.

    The implementation is intended to show how to wrap an environment to
    satisfy the interface of :class:`~tianshou.env.MultiAgentEnv`.

    :param size: the size of the board (square board)
    :param win_size: how many units in a row is considered to win
    """

    def __init__(self, size: int = 3, win_size: int = 3):
        super().__init__()
        assert size > 0, f'board size should be positive, but got {size}'
        self.size = size
        assert win_size > 0, f'win-size should be positive, but got {win_size}'
        self.win_size = win_size
        assert win_size <= size, f'win-size {win_size} should not ' \
            f'be larger than board size {size}'
        self.convolve_kernel = np.ones(win_size)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(size, size), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(size * size)
        self.current_board = None
        self.current_agent = None
        self._last_move = None
        self.step_num = None

    def reset(self) -> dict:
        self.current_board = np.zeros((self.size, self.size), dtype=np.int32)
        self.current_agent = 1
        self._last_move = (-1, -1)
        self.step_num = 0
        return {
            'agent_id': self.current_agent,
            'obs': np.array(self.current_board),
            'mask': self.current_board.flatten() == 0
        }

    def step(self, action: [int,
                            np.ndarray]) -> Tuple[dict, np.ndarray, np.ndarray, dict]:
        if self.current_agent is None:
            raise ValueError("calling step() of unreset environment is prohibited!")
        assert 0 <= action < self.size * self.size
        assert self.current_board.item(action) == 0
        _current_agent = self.current_agent
        self._move(action)
        mask = self.current_board.flatten() == 0
        is_win, is_opponent_win = False, False
        is_win = self._test_win()
        # the game is over when one wins or there is only one empty place
        done = is_win
        if sum(mask) == 1:
            done = True
            self._move(np.where(mask)[0][0])
            is_opponent_win = self._test_win()
        if is_win:
            reward = 1
        elif is_opponent_win:
            reward = -1
        else:
            reward = 0
        obs = {
            'agent_id': self.current_agent,
            'obs': np.array(self.current_board),
            'mask': mask
        }
        rew_agent_1 = reward if _current_agent == 1 else (-reward)
        rew_agent_2 = reward if _current_agent == 2 else (-reward)
        vec_rew = np.array([rew_agent_1, rew_agent_2], dtype=np.float32)
        if done:
            self.current_agent = None
        return obs, vec_rew, np.array(done), {}

    def _move(self, action):
        row, col = action // self.size, action % self.size
        if self.current_agent == 1:
            self.current_board[row, col] = 1
        else:
            self.current_board[row, col] = -1
        self.current_agent = 3 - self.current_agent
        self._last_move = (row, col)
        self.step_num += 1

    def _test_win(self):
        """test if someone wins by checking the situation around last move"""
        row, col = self._last_move
        rboard = self.current_board[row, :]
        cboard = self.current_board[:, col]
        current = self.current_board[row, col]
        rightup = [
            self.current_board[row - i, col + i] for i in range(1, self.size - col)
            if row - i >= 0
        ]
        leftdown = [
            self.current_board[row + i, col - i] for i in range(1, col + 1)
            if row + i < self.size
        ]
        rdiag = np.array(leftdown[::-1] + [current] + rightup)
        rightdown = [
            self.current_board[row + i, col + i] for i in range(1, self.size - col)
            if row + i < self.size
        ]
        leftup = [
            self.current_board[row - i, col - i] for i in range(1, col + 1)
            if row - i >= 0
        ]
        diag = np.array(leftup[::-1] + [current] + rightdown)
        results = [
            np.convolve(k, self.convolve_kernel, mode='valid')
            for k in (rboard, cboard, rdiag, diag)
        ]
        return any([(np.abs(x) == self.win_size).any() for x in results])

    def seed(self, seed: Optional[int] = None) -> int:
        pass

    def render(self, **kwargs) -> None:
        print(f'board (step {self.step_num}):')
        pad = '==='
        top = pad + '=' * (2 * self.size - 1) + pad
        print(top)

        def f(i, data):
            j, number = data
            last_move = i == self._last_move[0] and j == self._last_move[1]
            if number == 1:
                return 'X' if last_move else 'x'
            if number == -1:
                return 'O' if last_move else 'o'
            return '_'

        for i, row in enumerate(self.current_board):
            print(pad + ' '.join(map(partial(f, i), enumerate(row))) + pad)
        print(top)

    def close(self) -> None:
        pass
