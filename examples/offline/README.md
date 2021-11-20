# Offline

In offline reinforcement learning setting, the agent learns a policy from a fixed dataset which is collected once with any policy. And the agent does not interact with environment anymore. 

Once the dataset is collected, it will not be changed during training. We use [d4rl](https://github.com/rail-berkeley/d4rl) datasets to train offline agent. You can refer to [d4rl](https://github.com/rail-berkeley/d4rl) to see how to use d4rl datasets. 



## Train

Tianshou provides an `offline_trainer` for offline reinforcement learning. You can parse d4rl datasets into a `ReplayBuffer` , and set it as the parameter `buffer` of `offline_trainer`.  `offline_bcq.py` is an example of offline RL using the d4rl dataset.



To train an agent:

```bash
python offline_bcq.py --task halfcheetah-expert-v1
```

After 1M steps:

![halfcheetah-expert-v1_reward](/media/thk/新加卷/MachineLearning/tianshou/examples/offline/results/bcq/halfcheetah-expert-v1_reward.png)

`halfcheetah-expert-v1` is a mujoco environment. The setting of hyperparameters can refer to the offpolicy algorithms in mujoco.



## Results

| Environment \\ Algorithm | BCQ           |      |
| ------------------------ | ------------- | ---- |
| halfcheetah-expert-v1    | 10624.0±181.4 |      |
|                          |               |      |

