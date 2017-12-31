# Unit Test

This is a two-player zero-sum perfect information extensive game. Player 1 and player 2 iteratively choose actions. At every iteration, player 1 players first and player 2 follows. Both players have choices 0 or 1.

The number of iterations is given as a fixed number. After one game finished, the game counts the number of 0s and 1s that are choosen. If the number of 1 is more than that of 0, player 1 gets 1 and player 2 gets -1. If the number of 1 is less than that of 0, player 1 gets -1 and player 2 gets 1. Otherwise, they both get 0.

## Files

+ game.py: run this file to play the game.
+ agent.py: a class for players. MCTS is used here.
+ ZOgame.py: the game environment.
+ mcts.py: MCTS method.
+ Evaluator: evaluator for MCTS. Rollout policy is also here.

## Parameters

Three paramters are given in game.py.

+ size: the number of iterations
+ searching_step: the number of searching times of MCTS for one step
+ temp: the temporature paramter used to tradeoff exploitation and exploration
