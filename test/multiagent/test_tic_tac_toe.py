import pprint
from tianshou.data import Collector
from tic_tac_toe import get_args, train_agent, watch


def test_tic_tac_toe(args=get_args()):
    Collector._default_rew_metric = lambda x: x[args.agent_id - 1]
    if args.watch:
        watch(args)
        return

    result, agent = train_agent(args)
    assert result["best_reward"] >= args.win_rate

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        watch(args, agent)


if __name__ == '__main__':
    test_tic_tac_toe(get_args())
