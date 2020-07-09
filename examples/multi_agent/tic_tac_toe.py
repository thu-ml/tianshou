import os
import pprint
from torch.utils.tensorboard import SummaryWriter
import torch
import argparse
import numpy as np
from typing import Tuple, Optional

from tianshou.env import MAEnv, VectorEnv
from tianshou.policy import MADQNPolicy
from tianshou.data import Collector, ReplayBuffer
from tianshou.trainer import offpolicy_trainer

from MANet import Net


class TicTacToeEnv(MAEnv):
    board = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]

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
            self.current_board.flatten().tolist()) if b == 0}
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
        number_board = self.current_board.tolist()

        def f(number):
            if number == 1:
                return 'X'
            if number == -1:
                return 'O'
            return '_'
        for row in number_board:
            print(' '.join([f(each) for each in row]))

    def close(self) -> None:
        pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def tic_tac_toe(args=get_args()):
    args.state_shape = (3, 3)
    args.action_shape = 9
    train_envs = VectorEnv(
        [lambda: TicTacToeEnv() for _ in range(args.training_num)])
    test_envs = VectorEnv(
        [lambda: TicTacToeEnv() for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(args.layer_num, args.state_shape, args.action_shape, args.device)
    net = net.to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = MADQNPolicy(
        net, optim, args.gamma, args.n_step,
        use_target_network=args.target_update_freq > 0,
        target_update_freq=args.target_update_freq,
        agent_id=0)

    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size)
    # log
    log_path = os.path.join(args.logdir, 'tic_tac_toe', 'dqn')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        return x >= 1.0

    def train_fn(x):
        policy.set_eps(args.eps_train)

    def test_fn(x):
        policy.set_eps(args.eps_test)

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, save_fn=save_fn, writer=writer)

    assert stop_fn(result['best_reward'])
    train_collector.close()
    test_collector.close()
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = TicTacToeEnv()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


if __name__ == '__main__':
    tic_tac_toe(get_args())
