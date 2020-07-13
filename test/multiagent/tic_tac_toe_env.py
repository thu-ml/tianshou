import gym
import numpy as np
from typing import Tuple, Optional
from scipy.signal import convolve2d

from tianshou.env import MultiAgentEnv


class TicTacToeEnv(MultiAgentEnv):
    """
    This is a simple implementation of the Tic-Tac-Toe game, where
    two agents play against each other.

    The implementation is intended to show how to wrap an environment
    to satisfy the interface of :class:`~tianshou.env.MultiAgentEnv`.
    """
    def __init__(self, size: int = 3, win_size: int = 3):
        """
        :param size: the size of the board (square board)
        :param win_size: how many units in a row is considered to win
        """
        super().__init__(None)
        assert size > 0, f'board size should be positive, but got {size}'
        self.size = size
        assert win_size > 0, f'win-size should be positive, ' \
                             f'but got {win_size}'
        self.win_size = win_size
        assert win_size <= size, f'win-size {win_size} should not' \
                                 f' be larger than board size {size}'
        self.kernels = TicTacToeEnv._construct_kernels(win_size)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(size, size), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(size * size)
        self.current_board = None
        self.current_agent = None

    def reset(self) -> dict:
        self.current_board = np.zeros((self.size, self.size), dtype=np.int32)
        self.current_agent = 1
        return {
            'agent_id': self.current_agent,
            'obs': np.array(self.current_board),
            'legal_actions': set(i for i in range(self.size * self.size))
        }

    def step(self, action: np.ndarray
             ) -> Tuple[dict, np.ndarray, np.ndarray, dict]:
        if self.current_agent is None:
            raise ValueError(
                "calling step() of unreset environment is prohibited!")
        action = action.item(0)
        assert 0 <= action < self.size * self.size
        assert self.current_board.item(action) == 0
        _current_agent = self.current_agent
        self._move(action)
        legal_actions = {i for i, b in enumerate(
            self.current_board.flatten()) if b == 0}
        is_win, is_opponent_win = False, False
        is_win = self._test_win()
        # the game is over when one wins or there is only one empty place
        done = is_win
        if len(legal_actions) == 1:
            done = True
            self._move(list(legal_actions)[0])
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
            'legal_actions': legal_actions
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

    def _test_win(self):
        """
        test if some wins
        """
        results = [convolve2d(self.current_board, k, mode='valid')
                   for k in self.kernels]
        return any([(np.abs(x) == self.win_size).any() for x in results])

    @staticmethod
    def _construct_kernels(win_size):
        kernels = []
        holder = np.zeros((win_size, win_size))
        for i in range(win_size):
            row = holder.copy()
            row[i, :] = 1
            col = holder.copy()
            col[:, i] = 1
            kernels += [row, col]
        diag = np.eye(win_size)
        kernels += [diag, np.rot90(diag)]
        return kernels

    def seed(self, seed: Optional[int] = None) -> int:
        pass

    def render(self, **kwargs) -> None:
        print('board:')
        pad = '==='
        top = pad + '=' * (2 * self.size - 1) + pad
        print(top)

        def f(number):
            if number == 1:
                return 'X'
            if number == -1:
                return 'O'
            return '_'
        for row in self.current_board:
            print(pad + ' '.join(map(f, row)) + pad)
        print(top)

    def close(self) -> None:
        pass
