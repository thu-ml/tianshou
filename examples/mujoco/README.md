# Mujoco Result



## SAC (single run)

The best reward computes from 100 episodes returns in the test phase.

| task           | 3M best reward    | parameters                                              | time cost (3M) |
| -------------- | ----------------- | ------------------------------------------------------- | -------------- |
| HalfCheetah-v3 | 10157.70 ± 171.70 | `python3 mujoco_sac.py --task HalfCheetah-v3`           | 2~3h           |
| Walker2d-v3    | 5143.04 ± 15.57   | `python3 mujoco_sac.py --task Walker2d-v3`              | 2~3h           |
| Hopper-v3      | 3604.19 ± 169.55  | `python3 mujoco_sac.py --task Hopper-v3`                | 2~3h           |
| Humanoid-v3    | 6579.20 ± 1470.57 | `python3 mujoco_sac.py --task Humanoid-v3 --alpha 0.05` | 2~3h           |
| Ant-v3         | 6281.65 ± 686.28  | `python3 mujoco_sac.py --task Ant-v3`                   | 2~3h           |

![](results/sac/all.png)



