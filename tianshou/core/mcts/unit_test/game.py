import ZOGame
import agent


if __name__ == '__main__':
    print("Our game has 2 players.")
    print("Player 1 has color 1 and plays first. Player 2 has color -1 and plays following player 1.")
    print("Both player choose 1 or 0 for an action.")
    size = 2
    print("This game has {} iterations".format(size))
    print("If the final sequence has more 1 that 0, player 1 wins.")
    print("If the final sequence has less 1 that 0, player 2 wins.")
    print("Otherwise, both players get 0.\n")
    game = ZOGame.ZOTree(size)
    player1 = agent.Agent(size, 1)
    player2 = agent.Agent(size, -1)

    seq = []
    print("Sequence is {}\n".format(seq))
    while True:
        action1 = player1.gen_move(seq)
        print("action1 is {}".format(action1))
        result = game.executor_do_move([seq, 1], action1)
        print("Sequence is {}\n".format(seq))
        if not result:
            winner = game.executor_get_reward([seq, 1])
            break
        action2 = player2.gen_move(seq)
        print("action2 is {}".format(action2))
        result = game.executor_do_move([seq, -1], action2)
        print("Sequence is {}\n".format(seq))
        if not result:
            winner = game.executor_get_reward([seq, 1])
            break

    print("The choice sequence is {}".format(seq))
    print("The game result is {}".format(winner))
