import pprint

from pistonball import get_args, train_agent, watch


def test_piston_ball(args=get_args()):
    if args.watch:
        watch(args)
        return

    result, agent = train_agent(args)
    # assert result["best_reward"] >= args.win_rate

    if __name__ == "__main__":
        pprint.pprint(result)
        # Let's watch its performance!
        watch(args, agent)


if __name__ == "__main__":
    test_piston_ball(get_args())
