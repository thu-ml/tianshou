# Bipedal-Hardcore-SAC

- Our default choice: remove the done flag penalty, will soon converge to \~280 reward within 100 epochs (10M env steps, 3~4 hours, see the image below)
- If the done penalty is not removed, it converges much slower than before, about 200 epochs (20M env steps) to reach the same performance (\~200 reward)

![](results/sac/BipedalHardcore.png)
