# Tianshou's Mujoco Benchmark

We benchmarked Tianshou algorithm implementations in 9 out of 13 environments from the MuJoCo Gym task suite<sup>[[1]](#footnote1)</sup>.

For each supported algorithm and supported mujoco environments, we provide:
- Default hyperparameters used for benchmark and scripts to reproduce the benchmark;
- A comparison of performance (or code level details) with other open source implementations or classic papers;
- Graphs and raw data that can be used for research purposes<sup>[[2]](#footnote2)</sup>;
- Log details obtained during training<sup>[[2]](#footnote2)</sup>;
- Pretrained agents<sup>[[2]](#footnote2)</sup>;
- Some hints on how to tune the algorithm.
  

Supported algorithms are listed below:
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf), [commit id](https://github.com/thu-ml/tianshou/tree/v0.4.0)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf), [commit id](https://github.com/thu-ml/tianshou/tree/v0.4.0)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf), [commit id](https://github.com/thu-ml/tianshou/tree/v0.4.0)

## Offpolicy algorithms

#### Usage

Run

```bash
$ python mujoco_sac.py --task Ant-v3
```

Logs is saved in `./log/` and can be monitored with tensorboard.

```bash
$ tensorboard --logdir log
```

You can also reproduce the benchmark (e.g. SAC in Ant-v3) with the example script we provide under `examples/mujoco/`:

```bash
$ ./run_experiments.sh Ant-v3
```

This will start 10 experiments with different seeds.

#### Example benchmark

<img src="./benchmark/Ant-v3/figure.png" width="500" height="450">

Other graphs can be found under `/examples/mujuco/benchmark/`

#### Hints

In offpolicy algorithms(DDPG, TD3, SAC), the shared hyperparameters are almost the same<sup>[[8]](#footnote8)</sup>, and most hyperparameters are consistent with those used for benchmark in SpinningUp's implementations<sup>[[9]](#footnote9)</sup>.

By comparison to both classic literature and open source implementations (e.g., SpinningUp)<sup>[[1]](#footnote1)</sup><sup>[[2]](#footnote2)</sup>, Tianshou's implementations of DDPG, TD3, and SAC are roughly at-parity with or better than the best reported results for these algorithms.

### DDPG

|      Environment       |     Tianshou      | [SpinningUp (PyTorch)](https://spinningup.openai.com/en/latest/spinningup/bench.html) | [TD3 paper (DDPG)](https://arxiv.org/abs/1802.09477) | [TD3 paper (OurDDPG)](https://arxiv.org/abs/1802.09477) |
| :--------------------: | :---------------: | :----------------------------------------------------------: | :--------------------------------------------------: | :-----------------------------------------------------: |
|          Ant           |     990.4±4.3     |                             ~840                             |                      **1005.3**                      |                          888.8                          |
|      HalfCheetah       | **11718.7±465.6** |                            ~11000                            |                        3305.6                        |                         8577.3                          |
|         Hopper         | **2197.0±971.6**  |                            ~1800                             |                      **2020.5**                      |                         1860.0                          |
|        Walker2d        |   1400.6±905.0    |                            ~1950                             |                        1843.6                        |                       **3098.1**                        |
|        Swimmer         |   **144.1±6.5**   |                             ~137                             |                          N                           |                            N                            |
|        Humanoid        |  **177.3±77.6**   |                              N                               |                          N                           |                            N                            |
|        Reacher         |   **-3.3±0.3**    |                              N                               |                        -6.51                         |                          -4.01                          |
|    InvertedPendulum    |  **1000.0±0.0**   |                              N                               |                      **1000.0**                      |                       **1000.0**                        |
| InvertedDoublePendulum |   8364.3±2778.9   |                              N                               |                      **9355.5**                      |                         8370.0                          |

\* details<sup>[[5]](#footnote5)</sup><sup>[[6]](#footnote6)</sup><sup>[[7]](#footnote7)</sup>

### TD3

|      Environment       |     Tianshou      | [SpinningUp (Pytorch)](https://spinningup.openai.com/en/latest/spinningup/bench.html) |   [TD3 paper](https://arxiv.org/abs/1802.09477)    |
| :--------------------: | :---------------: | :-------------------: | :--------------: |
|          Ant           | **5116.4±799.9**  |         ~3800         |  4372.4±1000.3   |
|      HalfCheetah       | **10201.2±772.8** |         ~9750         |   9637.0±859.1   |
|         Hopper         |   3472.2±116.8    |         ~2860         | **3564.1±114.7** |
|        Walker2d        |   3982.4±274.5    |         ~4000         | **4682.8±539.6** |
|        Swimmer         |  **104.2±34.2**   |          ~78          |        N         |
|        Humanoid        | **5189.5±178.5**  |           N           |        N         |
|        Reacher         |   **-2.7±0.2**    |           N           |     -3.6±0.6     |
|    InvertedPendulum    |  **1000.0±0.0**   |           N           |  **1000.0±0.0**  |
| InvertedDoublePendulum |  **9349.2±14.3**  |           N           | **9337.5±15.0**  |

\* details<sup>[[5]](#footnote5)</sup><sup>[[6]](#footnote6)</sup><sup>[[7]](#footnote7)</sup>

### SAC

|      Environment       |      Tianshou      | [SpinningUp (Pytorch)](https://spinningup.openai.com/en/latest/spinningup/bench.html) | [SAC paper](https://arxiv.org/abs/1801.01290) |
| :--------------------: | :----------------: | :-------------------: | :---------: |
|          Ant           |  **5850.2±475.7**  |         ~3980         |    ~3720    |
|      HalfCheetah       | **12138.8±1049.3** |        ~11520         |   ~10400    |
|         Hopper         |  **3542.2±51.5**   |         ~3150         |    ~3370    |
|        Walker2d        |  **5007.0±251.5**  |         ~4250         |    ~3740    |
|        Swimmer         |    **44.4±0.5**    |         ~41.7         |      N      |
|        Humanoid        |  **5488.5±81.2**   |           N           |    ~5200    |
|        Reacher         |    **-2.6±0.2**    |           N           |      N      |
|    InvertedPendulum    |   **1000.0±0.0**   |           N           |      N      |
| InvertedDoublePendulum |   **9359.5±0.4**   |           N           |      N      |

\* details<sup>[[5]](#footnote5)</sup><sup>[[6]](#footnote6)</sup>

#### Hints for SAC

0. DO NOT share the same network with two critic networks.
1. The sigma (of the Gaussian policy) should be conditioned on input.
2. The network size should not be less than 256.
3. The deterministic evaluation helps a lot :)

## Onpolicy Algorithms

TBD




## Note

<a name="footnote1">[1]</a>  Supported environments include HalfCheetah-v3, Hopper-v3, Swimmer-v3, Walker2d-v3, Ant-v3, Humanoid-v3, Reacher-v2, InvertedPendulum-v2 and InvertedDoublePendulum-v2. Pusher, Thrower, Striker and HumanoidStandup are not supported because they are not commonly seen in literatures.

<a name="footnote2">[2]</a>  Pretrained agents, detailed graphs (single agent, single game) and log details can all be found [here](https://cloud.tsinghua.edu.cn/d/356e0f5d1e66426b9828/).

<a name="footnote3">[3]</a>  We used the latest version of all mujoco environments in gym (0.17.3 with mujoco==2.0.2.13), but it's not often the case with other benchmarks. Please check for details yourself in the original paper. (Different version's outcomes are usually similar, though)

<a name="footnote4">[4]</a>  We didn't compare offpolicy algorithms to OpenAI baselines [benchmark](https://github.com/openai/baselines/blob/master/benchmarks_mujoco1M.htm), because for now it seems that they haven't provided benchmark for offpolicy algorithms, but in [SpinningUp docs](https://spinningup.openai.com/en/latest/spinningup/bench.html) they stated that "SpinningUp implementations of DDPG, TD3, and SAC are roughly at-parity with the best-reported results for these algorithms", so we think lack of comparisons with OpenAI baselines is okay.

<a name="footnote5">[5]</a>  ~ means the number is approximated from the graph because accurate numbers is not provided in the paper. N means graphs not provided.

<a name="footnote6">[6]</a>  Reward metric: The meaning of the table value is the max average return over 10 trails (different seeds) ± a single standard deviation over trails. Each trial is averaged on another 10 test seeds. Only the first 1M steps data will be considered. The shaded region on the graph also represents a single standard deviation. It is the same as [TD3 evaluation method](https://github.com/sfujim/TD3/issues/34).

<a name="footnote7">[7]</a>  In TD3 paper, shaded region represents only half of standard deviation.

<a name="footnote8">[8]</a>  SAC's start-timesteps is set to 10000 by default while it is 25000 is DDPG/TD3. TD3's learning rate is set to 3e-4 while it is 1e-3 for DDPG/SAC. However, there is NO enough evidence to support our choice of such hyperparameters (we simply choose them because of SpinningUp) and you can try playing with those hyperparameters to see if you can improve performance. Do tell us if you can!

<a name="footnote9">[9]</a>  We use batchsize of 256 in DDPG/TD3/SAC while SpinningUp use 100. Minor difference also lies with `start-timesteps`, data loop method `step_per_collect`, method to deal with/bootstrap truncated steps because of timelimit and unfinished/collecting episodes (contribute to performance improvement), etc.
