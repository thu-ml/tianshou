import argparse

import pytest
from pistonball_continuous import get_args, train_agent, watch


@pytest.mark.skip(reason="runtime too long and unstable result")
def test_piston_ball_continuous(args: argparse.Namespace = get_args()) -> None:
    if args.watch:
        watch(args)
        return

    result, agent = train_agent(args)
    # assert result.best_reward >= 30.0
