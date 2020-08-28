# Use DQN to play Atari Games (single run)

Use random policy, the sample speed is \~3000 env step per second (\~12000 Atari frame per second in fact since we use frame_stack=4). The main bottleneck is training the convolutional neural network.

| task                        | best reward | reward curve                          | parameters                                                   | time cost           |
| --------------------------- | ----------- | ------------------------------------- | ------------------------------------------------------------ | ------------------- |
| PongNoFrameskip-v4          | 20          | ![](results/dqn/Pong_rew.png)         | `python3 atari_dqn.py`                                       | ~30 min (~15 epoch) |
| BreakoutNoFrameskip-v4      | 316         | ![](results/dqn/Breakout_rew.png)     | `python3 atari_dqn.py --task "BreakoutNoFrameskip-v4" --test_num 100` | 3~4h (100 epoch)    |
| EnduroNoFrameskip-v4        | 670         | ![](results/dqn/Enduro_rew.png)       | `python3 atari_dqn.py --task "EnduroNoFrameskip-v4 " --test_num 100` | 3~4h (100 epoch)    |
| QbertNoFrameskip-v4         | 7307        | ![](results/dqn/Qbert_rew.png)        | `python3 atari_dqn.py --task "QbertNoFrameskip-v4" --test_num 100` | 3~4h (100 epoch)    |
| MsPacmanNoFrameskip-v4      | 2107        | ![](results/dqn/MsPacman_rew.png)     | `python3 atari_dqn.py --task "MsPacmanNoFrameskip-v4" --test_num 100` | 3~4h (100 epoch)    |
| SeaquestNoFrameskip-v4      | 2088        | ![](results/dqn/Seaquest_rew.png)     | `python3 atari_dqn.py --task "SeaquestNoFrameskip-v4" --test_num 100` | 3~4h (100 epoch)    |
| SpaceInvadersNoFrameskip-v4 | 812.2       | ![](results/dqn/SpaceInvader_rew.png) | `python3 atari_dqn.py --task "SpaceInvadersNoFrameskip-v4" --test_num 100` | 3~4h (100 epoch)    |

Note: the eps_train_final and eps_test in the original DQN paper is 0.1 and 0.01, but [some works](https://github.com/google/dopamine/tree/master/baselines) found that smaller eps helps improve the performance. Also, a large batchsize (say 64 instead of 32) will help but will slow down the speed. 

Have fun with playing these hyperparameters!
