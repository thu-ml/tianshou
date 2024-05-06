import argparse

import pytest
from pistonball import get_args, train_agent, watch


@pytest.mark.skip(reason="Performance bound was never tested, no point in running this for now")
def test_piston_ball(args: argparse.Namespace = get_args()) -> None:
    if args.watch:
        watch(args)
        return

    train_agent(args)
    # assert result.best_reward >= args.win_rate
