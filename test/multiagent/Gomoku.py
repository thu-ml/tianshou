import os
import pprint
from copy import deepcopy

import numpy as np
from tic_tac_toe import get_agents, get_parser, train_agent, watch
from tic_tac_toe_env import TicTacToeEnv
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import RandomPolicy
from tianshou.utils import TensorboardLogger


def get_args():
    parser = get_parser()
    parser.add_argument('--self_play_round', type=int, default=20)
    args = parser.parse_known_args()[0]
    return args


def gomoku(args=get_args()):
    Collector._default_rew_metric = lambda x: x[args.agent_id - 1]
    if args.watch:
        watch(args)
        return

    policy, optim = get_agents(args)
    agent_learn = policy.policies[args.agent_id - 1]
    agent_opponent = policy.policies[2 - args.agent_id]

    # log
    log_path = os.path.join(args.logdir, 'Gomoku', 'dqn')
    writer = SummaryWriter(log_path)
    args.logger = TensorboardLogger(writer)

    opponent_pool = [agent_opponent]

    def env_func():
        return TicTacToeEnv(args.board_size, args.win_size)

    test_envs = DummyVectorEnv([env_func for _ in range(args.test_num)])
    for r in range(args.self_play_round):
        rews = []
        agent_learn.set_eps(0.0)
        # compute the reward over previous learner
        for opponent in opponent_pool:
            policy.replace_policy(opponent, 3 - args.agent_id)
            test_collector = Collector(policy, test_envs)
            results = test_collector.collect(n_episode=100)
            rews.append(results['rews'].mean())
        rews = np.array(rews)
        # weight opponent by their difficulty level
        rews = np.exp(-rews * 10.0)
        rews /= np.sum(rews)
        total_epoch = args.epoch
        args.epoch = 1
        for epoch in range(total_epoch):
            # sample one opponent
            opp_id = np.random.choice(len(opponent_pool), size=1, p=rews)
            print(f'selection probability {rews.tolist()}')
            print(f'selected opponent {opp_id}')
            opponent = opponent_pool[opp_id.item(0)]
            agent = RandomPolicy()
            # previous learner can only be used for forward
            agent.forward = opponent.forward
            args.model_save_path = os.path.join(
                args.logdir, 'Gomoku', 'dqn', f'policy_round_{r}_epoch_{epoch}.pth'
            )
            result, agent_learn = train_agent(
                args, agent_learn=agent_learn, agent_opponent=agent, optim=optim
            )
            print(f'round_{r}_epoch_{epoch}')
            pprint.pprint(result)
        learnt_agent = deepcopy(agent_learn)
        learnt_agent.set_eps(0.0)
        opponent_pool.append(learnt_agent)
        args.epoch = total_epoch
    if __name__ == '__main__':
        # Let's watch its performance!
        opponent = opponent_pool[-2]
        watch(args, agent_learn, opponent)


if __name__ == '__main__':
    gomoku(get_args())
