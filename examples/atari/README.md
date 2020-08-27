Use DQN to play Atari Games

|task                          |best reward|  reward curve                       | parameters | time cost|
|  ----                        |  ----     |  ----                                 | ----      |----|
| PongNoFrameskip-v4           | 20        |  ![avatar](/results/dqn/pong_rew.png)      | python3 atari_dqn.py      | ~ 30 min(15 epoch)|
| BreakoutNoFrameskip-v4       | 316       |  ![avatar](/results/dqn/Breakout_rew.png)     | python3 atari_dqn.py --task "BreakoutNoFrameskip-v4" --test_num 100      |3~4h(100 epoch)|
| EnduroNoFrameskip-v4         | 670       |  ![avatar](/results/dqn/Enduro_rew.png)      | python3 atari_dqn.py --task "EnduroNoFrameskip-v4 " --test_num 100      |3~4h(100 epoch)|
| QbertNoFrameskip-v4          | 7307      |  ![avatar](/results/dqn/Qbert_rew.png)      | python3 atari_dqn.py --task "QbertNoFrameskip-v4" --test_num 100      |3~4h(100 epoch)|
| MsPacmanNoFrameskip-v4       | 2107      |  ![avatar](/results/dqn/MsPacman_rew.png)      | python3 atari_dqn.py --task "MsPacmanNoFrameskip-v4" --test_num 100      |3~4h(100 epoch)|
| SeaquestNoFrameskip-v4       | 2088      |  ![avatar](/results/dqn/Seaquest_rew.png)      | python3 atari_dqn.py --task "SeaquestNoFrameskip-v4" --test_num 100      |3~4h(100 epoch)|
| SpaceInvadersNoFrameskip-v4  | 812.2     |  ![avatar](/results/dqn/SpaceInvader_rew.png)      | python3 atari_dqn.py --task "SpaceInvadersNoFrameskip-v4" --test_num 100      |  3~4h(100 epoch)|
