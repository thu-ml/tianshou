import argparse

from tic_tac_toe import get_args, train_agent, watch


def test_tic_tac_toe(args: argparse.Namespace = get_args()) -> None:
    if args.watch:
        watch(args)
        return

    result, agent = train_agent(args)
    assert result.best_reward >= args.win_rate
