# Atari General

The sample speed is \~3000 env step per second (\~12000 Atari frame per second in fact since we use frame_stack=4) under the normal mode (use a CNN policy and a collector, also storing data into the buffer). The main bottleneck is training the convolutional neural network.

The Atari env seed cannot be fixed due to the discussion [here](https://github.com/openai/gym/issues/1478), but it is not a big issue since on Atari it will always have the similar results.

The env wrapper is a crucial thing. Without wrappers, the agent cannot perform well enough on Atari games. Many existing RL codebases use [OpenAI wrapper](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py), but it is not the original DeepMind version ([related issue](https://github.com/openai/baselines/issues/240)). Dopamine has a different [wrapper](https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/atari_lib.py) but unfortunately it cannot work very well in our codebase.

# DQN (single run)

One epoch here is equal to 100,000 env step, 100 epochs stand for 10M.

| task                        | best reward | reward curve                          | parameters                                                   | time cost           |
| --------------------------- | ----------- | ------------------------------------- | ------------------------------------------------------------ | ------------------- |
| PongNoFrameskip-v4          | 20          | ![](results/dqn/Pong_rew.png)         | `python3 atari_dqn.py --task "PongNoFrameskip-v4" --batch_size 64` | ~30 min (~15 epoch) |
| BreakoutNoFrameskip-v4      | 316         | ![](results/dqn/Breakout_rew.png)     | `python3 atari_dqn.py --task "BreakoutNoFrameskip-v4" --test_num 100`  | 3~4h (100 epoch)    |
| EnduroNoFrameskip-v4        | 670         | ![](results/dqn/Enduro_rew.png)       | `python3 atari_dqn.py --task "EnduroNoFrameskip-v4 " --test_num 100`  | 3~4h (100 epoch)    |
| QbertNoFrameskip-v4         | 7307        | ![](results/dqn/Qbert_rew.png)        | `python3 atari_dqn.py --task "QbertNoFrameskip-v4" --test_num 100`  | 3~4h (100 epoch)    |
| MsPacmanNoFrameskip-v4      | 2107        | ![](results/dqn/MsPacman_rew.png)     | `python3 atari_dqn.py --task "MsPacmanNoFrameskip-v4" --test_num 100`  | 3~4h (100 epoch)    |
| SeaquestNoFrameskip-v4      | 2088        | ![](results/dqn/Seaquest_rew.png)     | `python3 atari_dqn.py --task "SeaquestNoFrameskip-v4" --test_num 100`  | 3~4h (100 epoch)    |
| SpaceInvadersNoFrameskip-v4 | 812.2       | ![](results/dqn/SpaceInvader_rew.png) | `python3 atari_dqn.py --task "SpaceInvadersNoFrameskip-v4" --test_num 100`  | 3~4h (100 epoch)    |

Note: The eps_train_final and eps_test in the original DQN paper is 0.1 and 0.01, but [some works](https://github.com/google/dopamine/tree/master/baselines) found that smaller eps helps improve the performance. Also, a large batchsize (say 64 instead of 32) will help faster convergence but will slow down the training speed. 

We haven't tuned this result to the best, so have fun with playing these hyperparameters!
