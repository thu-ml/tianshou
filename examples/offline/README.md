# Offline

In offline reinforcement learning setting, the agent learns a policy from a fixed dataset which is collected once with any policy. And the agent does not interact with environment anymore. 

## Continuous control

Once the dataset is collected, it will not be changed during training. We use [d4rl](https://github.com/rail-berkeley/d4rl) datasets to train offline agent for continuous control. You can refer to [d4rl](https://github.com/rail-berkeley/d4rl) to see how to use d4rl datasets. 

We provide implementation of BCQ and CQL algorithm for continuous control.

### Train

Tianshou provides an `offline_trainer` for offline reinforcement learning. You can parse d4rl datasets into a `ReplayBuffer` , and set it as the parameter `buffer` of `offline_trainer`.  `d4rl_bcq.py` is an example of offline RL using the d4rl dataset.

## Results

### IL (Imitation Learning, aka, Behavior Cloning)

| Environment           | Dataset               | IL              | Parameters                                               |
| --------------------- | --------------------- | --------------- | -------------------------------------------------------- |
| HalfCheetah-v2        | halfcheetah-expert-v2 | 11355.31        | `python3 d4rl_il.py --task HalfCheetah-v2 --expert-data-task halfcheetah-expert-v2` |
| HalfCheetah-v2        | halfcheetah-medium-v2 | 5098.16        | `python3 d4rl_il.py --task HalfCheetah-v2 --expert-data-task halfcheetah-medium-v2` |

### BCQ

| Environment           | Dataset               | BCQ             | Parameters                                               |
| --------------------- | --------------------- | --------------- | -------------------------------------------------------- |
| HalfCheetah-v2        | halfcheetah-expert-v2 | 11509.95        | `python3 d4rl_bcq.py --task HalfCheetah-v2 --expert-data-task halfcheetah-expert-v2` |
| HalfCheetah-v2        | halfcheetah-medium-v2 | 5147.43        | `python3 d4rl_bcq.py --task HalfCheetah-v2 --expert-data-task halfcheetah-medium-v2` |

### CQL

| Environment           | Dataset               | CQL             | Parameters                                               |
| --------------------- | --------------------- | --------------- | -------------------------------------------------------- |
| HalfCheetah-v2        | halfcheetah-expert-v2 | 2864.37         | `python3 d4rl_cql.py --task HalfCheetah-v2 --expert-data-task halfcheetah-expert-v2` |
| HalfCheetah-v2        | halfcheetah-medium-v2 | 6505.41         | `python3 d4rl_cql.py --task HalfCheetah-v2 --expert-data-task halfcheetah-medium-v2` |

### TD3+BC

| Environment           | Dataset               | CQL             | Parameters                                               |
| --------------------- | --------------------- | --------------- | -------------------------------------------------------- |
| HalfCheetah-v2        | halfcheetah-expert-v2 | 11788.25         | `python3 d4rl_td3_bc.py --task HalfCheetah-v2 --expert-data-task halfcheetah-expert-v2` |
| HalfCheetah-v2        | halfcheetah-medium-v2 | 5741.13         | `python3 d4rl_td3_bc.py --task HalfCheetah-v2 --expert-data-task halfcheetah-medium-v2` |

#### Observation normalization

Following the original paper, we use observation normalization by default. You can turn it off by setting `--norm-obs 0`. The difference are small but consistent.

| Dataset | w/ norm-obs | w/o norm-obs |
| :--- | :--- | :--- |
| halfcheeta-medium-v2 | 5741.13 | 5724.41 |
| halfcheeta-expert-v2 | 11788.25 | 11665.77 |
| walker2d-medium-v2 | 4051.76 | 3985.59 |
| walker2d-expert-v2 | 5068.15 | 5027.75 |

## Discrete control

For discrete control, we currently use ad hoc Atari data generated from a trained QRDQN agent.

### Gather Data

To running CQL algorithm on Atari, you need to do the following things:

- Train an expert, by using the command listed in the QRDQN section of Atari examples: `python3 atari_qrdqn.py --task {your_task}`
- Generate buffer with noise: `python3 atari_qrdqn.py --task {your_task} --watch --resume-path log/{your_task}/qrdqn/policy.pth --eps-test 0.2 --buffer-size 1000000 --save-buffer-name expert.hdf5` (note that 1M Atari buffer cannot be saved as `.pkl` format because it is too large and will cause error);
- Train offline model: `python3 atari_{bcq,cql,crr}.py --task {your_task} --load-buffer-name expert.hdf5`.

### IL

We test our IL implementation on two example tasks (different from author's version, we use v4 instead of v0; one epoch means 10k gradient step):

| Task                   | Online QRDQN | Behavioral | IL                               | parameters                                                   |
| ---------------------- | ---------- | ---------- | --------------------------------- |  ------------------------------------------------------------ |
| PongNoFrameskip-v4     | 20.5       | 6.8        | 20.0 (epoch 5)                    | `python3 atari_il.py --task PongNoFrameskip-v4 --load-buffer-name log/PongNoFrameskip-v4/qrdqn/expert.hdf5 --epoch 5` |
| BreakoutNoFrameskip-v4 | 394.3      | 46.9       | 121.9 (epoch 12, could be higher)  | `python3 atari_il.py --task BreakoutNoFrameskip-v4 --load-buffer-name log/BreakoutNoFrameskip-v4/qrdqn/expert.hdf5 --epoch 12` |

### BCQ

We test our BCQ implementation on two example tasks (different from author's version, we use v4 instead of v0; one epoch means 10k gradient step):

| Task                   | Online QRDQN | Behavioral | BCQ                               | parameters                                                   |
| ---------------------- | ---------- | ---------- | --------------------------------- |  ------------------------------------------------------------ |
| PongNoFrameskip-v4     | 20.5       | 6.8        | 20.1 (epoch 5)                    | `python3 atari_bcq.py --task PongNoFrameskip-v4 --load-buffer-name log/PongNoFrameskip-v4/qrdqn/expert.hdf5 --epoch 5` |
| BreakoutNoFrameskip-v4 | 394.3      | 46.9       | 64.6 (epoch 12, could be higher)  | `python3 atari_bcq.py --task BreakoutNoFrameskip-v4 --load-buffer-name log/BreakoutNoFrameskip-v4/qrdqn/expert.hdf5 --epoch 12` |

### CQL

We test our CQL implementation on two example tasks (different from author's version, we use v4 instead of v0; one epoch means 10k gradient step):

| Task                   | Online QRDQN | Behavioral | CQL                               | parameters                                                   |
| ---------------------- | ---------- | ---------- | --------------------------------- | ------------------------------------------------------------ |
| PongNoFrameskip-v4     | 20.5         | 6.8        | 20.4 (epoch 5)                      | `python3 atari_cql.py --task PongNoFrameskip-v4 --load-buffer-name log/PongNoFrameskip-v4/qrdqn/expert.hdf5 --epoch 5` |
| BreakoutNoFrameskip-v4 | 394.3        | 46.9       | 129.4 (epoch 12) | `python3 atari_cql.py --task BreakoutNoFrameskip-v4 --load-buffer-name log/BreakoutNoFrameskip-v4/qrdqn/expert.hdf5 --epoch 12 --min-q-weight 50` |

We reduce the size of the offline data to 10% and 1% of the above and get:

Buffer size 100000:

| Task                   | Online QRDQN | Behavioral | CQL                               | parameters                                                   |
| ---------------------- | ---------- | ---------- | --------------------------------- | ------------------------------------------------------------ |
| PongNoFrameskip-v4     | 20.5         | 5.8        | 21 (epoch 5)                      | `python3 atari_cql.py --task PongNoFrameskip-v4 --load-buffer-name log/PongNoFrameskip-v4/qrdqn/expert.size_1e5.hdf5 --epoch 5` |
| BreakoutNoFrameskip-v4 | 394.3        | 41.4       | 40.8 (epoch 12) | `python3 atari_cql.py --task BreakoutNoFrameskip-v4 --load-buffer-name log/BreakoutNoFrameskip-v4/qrdqn/expert.size_1e5.hdf5 --epoch 12 --min-q-weight 20` |

Buffer size 10000:

| Task                   | Online QRDQN | Behavioral | CQL                               | parameters                                                   |
| ---------------------- | ---------- | ---------- | --------------------------------- | ------------------------------------------------------------ |
| PongNoFrameskip-v4     | 20.5         | nan        | 1.8 (epoch 5)                      | `python3 atari_cql.py --task PongNoFrameskip-v4 --load-buffer-name log/PongNoFrameskip-v4/qrdqn/expert.size_1e4.hdf5 --epoch 5 --min-q-weight 1` |
| BreakoutNoFrameskip-v4 | 394.3        | 31.7       | 22.5 (epoch 12) | `python3 atari_cql.py --task BreakoutNoFrameskip-v4 --load-buffer-name log/BreakoutNoFrameskip-v4/qrdqn/expert.size_1e4.hdf5 --epoch 12 --min-q-weight 10` |

### CRR

We test our CRR implementation on two example tasks (different from author's version, we use v4 instead of v0; one epoch means 10k gradient step):

| Task                   | Online QRDQN | Behavioral | CRR            | CRR w/ CQL        | parameters                                                   |
| ---------------------- | ---------- | ---------- | ---------------- | ----------------- | ------------------------------------------------------------ |
| PongNoFrameskip-v4     | 20.5         | 6.8        | -21 (epoch 5)   |  17.7 (epoch 5)  | `python3 atari_crr.py --task PongNoFrameskip-v4 --load-buffer-name log/PongNoFrameskip-v4/qrdqn/expert.hdf5 --epoch 5` |
| BreakoutNoFrameskip-v4 | 394.3        | 46.9       | 23.3 (epoch 12) | 76.9 (epoch 12) | `python3 atari_crr.py --task BreakoutNoFrameskip-v4 --load-buffer-name log/BreakoutNoFrameskip-v4/qrdqn/expert.hdf5 --epoch 12 --min-q-weight 50` |

Note that CRR itself does not work well in Atari tasks but adding CQL loss/regularizer helps.

### RL Unplugged Data

We provide a script to convert the Atari datasets of [RL Unplugged](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged) to Tianshou ReplayBuffer.

For example, the following command will download the first shard of the first run of Breakout game to `~/.rl_unplugged/datasets/Breakout/run_1-00001-of-00100` then convert it to a `tianshou.data.ReplayBuffer` and save it to `~/.rl_unplugged/buffers/Breakout/run_1-00001-of-00100.hdf5` (use `--dataset-dir` and `--buffer-dir` to change the default directories):

```bash
python3 convert_rl_unplugged_atari.py --task Breakout --run-id 1 --shard-id 1
```

Then you can use it to train an agent by:

```bash
python3 atari_bcq.py --task BreakoutNoFrameskip-v4 --load-buffer-name ~/.rl_unplugged/buffers/Breakout/run_1-00001-of-00100.hdf5 --buffer-from-rl-unplugged --epoch 12
```

Note:
 - Each shard contains about 500k transitions.
 - This conversion script depends on Tensorflow.
 - It takes about 1 hour to process one shard on my machine. YMMV.
