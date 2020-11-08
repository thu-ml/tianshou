# Mujoco General

# DDPG
TBD

# TD3 (single run, seed=0)

For TD3 algorithm, we try to let our parameters be consistent with [Github: TD3](https://github.com/sfujim/TD3). However, evaluations below only report max of rewards over 1 trail(1M steps) using a fixed seed 0, where each reward represents average return over 10 episodes with no exploration noise. Parameter configuration can be found at td3_experiments.sh.
| task                        | best reward | reward curve (statistics after 1M steps are not used)                        | time cost |
| --------------------------- | ----------- | ---------------------------------------------------------------------------- | --------- |
| HalfCheetah-v3              | 8013        | ![](results/td3/HalfCheetah_rew.svg)                                         | ~3.5h     |
| Hopper-v3                   | 3281        | ![](results/td3/Hopper_rew.svg)                                              | ~3.5h     |
| Walker2d-v3                 | 3668        | ![](results/td3/Walker2d_rew.svg)                                            | ~3.5h     |
| Ant-v3                      | 5727        | ![](results/td3/Ant_rew.svg)                                                 | ~4h       |
| Humanoid-v3                 | 5149        | ![](results/td3/Humanoid_rew.svg)                                            | ~4.5h     |
| InvertedPendulum-v2         | 2088        | ![](results/td3/InvertedPendulum_rew.svg)                                    | ~3.5h     |
| InvertedDoublePendulum-v2   | 2088        | ![](results/td3/InvertedDoublePendulum.svg)                                  | ~3.5h     |
| Reacher-v2                  | -7.3        | ![](results/td3/Reacher_rew.svg)                                             | ~3.5h     |

# SAC
TBD


# Bibtex

```
@inproceedings{fujimoto2018addressing,
  title={Addressing Function Approximation Error in Actor-Critic Methods},
  author={Fujimoto, Scott and Hoof, Herke and Meger, David},
  booktitle={International Conference on Machine Learning},
  pages={1582--1591},
  year={2018}
}
```