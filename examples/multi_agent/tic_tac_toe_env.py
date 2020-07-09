import gym
import numpy as np
from typing import Tuple, Optional

from tianshou.env import MultiAgentEnv


class TicTacToeEnv(MultiAgentEnv):
    """
    This is a simple implementation of the Tic-Tac-Toe game, where
    two agents play against each other.

    The implementation is intended to show how to wrap an environment
    to satisfy the interface of :class:`~tianshou.env.MAEnv`.
    """
    board = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
    observation_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(3, 3), dtype=np.float32)
    action_space = gym.spaces.Discrete(9)

    def __init__(self):
        super().__init__(None)
        self.current_board = np.array(TicTacToeEnv.board)
        self.current_player = 0

    def reset(self) -> dict:
        self.current_board = np.array(TicTacToeEnv.board)
        self.current_player = 0
        return {
            'agent_id': 0,
            'obs': np.array(self.current_board),
            'legal_actions': set(i for i in range(9))
        }

    def step(self, action: np.ndarray
             ) -> Tuple[dict, np.ndarray, np.ndarray, dict]:
        action = action.item(0)
        assert 0 <= action < 9
        assert self.current_board.item(action) == 0
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
            'agent_id': self.current_player,
            'obs': np.array(self.current_board),
            'legal_actions': legal_actions
        }
        return obs, np.array(reward), np.array(done), {}

    def _move(self, action):
        tmp_board = self.current_board.flatten().tolist()
        if self.current_player == 0:
            tmp_board[action] = 1
        else:
            tmp_board[action] = -1
        self.current_board = np.array(tmp_board).reshape(
            self.current_board.shape)
        self.current_player = 1 - self.current_player

    def _test_win(self):
        """
        test if some wins
        """
        test = np.sum(self.current_board, axis=0).tolist() \
            + np.sum(self.current_board, axis=1).tolist() \
            + [np.sum(np.diag(self.current_board)).item(0)] \
            + [np.sum(np.diag(np.rot90(self.current_board))).item(0)]
        is_win = (np.abs(np.array(test)) == 3).any()
        return is_win

    def seed(self, seed: Optional[int] = None) -> int:
        pass

    def render(self, **kwargs) -> None:
        print('======board=====')
        number_board = self.current_board

        def f(number):
            if number == 1:
                return 'X'
            if number == -1:
                return 'O'
            return '_'
        for row in number_board:
            print(' '.join(map(f, row)))

    def close(self) -> None:
        pass
