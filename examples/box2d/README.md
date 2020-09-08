# Bipedal-Hardcore-SAC

- Our default choice: remove the done flag penalty, will soon converge to \~250 reward within 100 epochs (10M env steps, 3~4 hours, see the image below)
- If the done penalty is not removed, it converges much slower than before, about 200 epochs (20M env steps) to reach the same performance (\~200 reward)
- Action noise is only necessary in the beginning. It is a negative impact at the end of the training. Removing it can reach \~255 (our best result under the original env, no done penalty removed).

![](results/sac/BipedalHardcore.png)