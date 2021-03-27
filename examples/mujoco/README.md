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
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf), [commit id](https://github.com/thu-ml/tianshou/tree/e605bdea942b408126ef4fbc740359773259c9ec)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf), [commit id](https://github.com/thu-ml/tianshou/tree/e605bdea942b408126ef4fbc740359773259c9ec)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf), [commit id](https://github.com/thu-ml/tianshou/tree/e605bdea942b408126ef4fbc740359773259c9ec)
- [REINFORCE algorithm](https://papers.nips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf), [commit id](https://github.com/thu-ml/tianshou/tree/e27b5a26f330de446fe15388bf81c3777f024fb9)
- A2C, commit id (TODO)

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

<img src="./benchmark/Ant-v3/offpolicy.png" width="500" height="450">

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

### REINFORCE

|      Environment       | Tianshou(10M steps) |
| :--------------------: | :-----------------: |
|          Ant           |  **1108.1±323.1**   |
|      HalfCheetah       |  **1138.8±104.7**   |
|         Hopper         |   **416.0±104.7**   |
|        Walker2d        |   **440.9±148.2**   |
|        Swimmer         |    **35.6±2.6**     |
|        Humanoid        |   **464.3±58.4**    |
|        Reacher         |    **-5.5±0.2**     |
|    InvertedPendulum    |   **1000.0±0.0**    |
| InvertedDoublePendulum |  **7726.2±1287.3**  |


|      Environment       |      Tianshou(3M steps)      | [SpinningUp (VPG Pytorch)](https://spinningup.openai.com/en/latest/spinningup/bench_vpg.html)<sup>[[10]](#footnote10)</sup> |
| :--------------------: | :--------------------------: | :------------------------: |
|          Ant           |       **474.9+-133.5**       |             ~5             |
|      HalfCheetah       |       **884.0+-41.0**        |            ~600            |
|         Hopper         |         395.8+-64.5*         |          **~800**          |
|        Walker2d        |         412.0+-52.4          |          **~460**          |
|        Swimmer         |          35.3+-1.4           |          **~51**           |
|        Humanoid        |       **438.2+-47.8**        |             N              |
|        Reacher         |        **-10.5+-0.7**        |             N              |
|    InvertedPendulum    |        **999.2+-2.4**        |             N              |
| InvertedDoublePendulum |      **1059.7+-307.7**       |             N              |

\* details<sup>[[5]](#footnote5)</sup><sup>[[6]](#footnote6)</sup>

#### Hints for REINFORCE

0. Following [Andrychowicz, Marcin, et al](https://arxiv.org/abs/2006.05990), we downscale last layer of policy network by a factor of 0.01 after orthogonal initialization.
1. We choose "tanh" function to squash sampled action from range (-inf, inf) to (-1, 1) rather than usually used clipping method (As in StableBaselines3). We did full scale ablation studies and results show that tanh squashing performs a tiny little bit better than clipping overall, and is much better than no action bounding. However, "clip" method is still a very good method, considering its simplicity.
2. We use global observation normalization and global rew-to-go (value) normalization by default. Both are crucial to good performances of REINFORCE algorithm. Since we minus mean when doing rew-to-go normalization, you can treat global mean of rew-to-go as a naive version of "baseline".
3. Since we do not have a value estimator, we use global rew-to-go mean to bootstrap truncated steps because of timelimit and unfinished collecting, while most other implementations use 0. We feel this would help because mean is more likely a better estimate than 0 (no ablation study has been done).
4. We have done full scale ablation study on learning rate and lr decay strategy. We experiment with lr of 3e-4, 5e-4, 1e-3, each have 2 options: no lr decay or linear decay to 0. Experiments show that 3e-4 learning rate will cause slowly learning and make agent step in local optima easily for certain environments like InvertedDoublePendulum, Ant, HalfCheetah, and 1e-3 lr helps a lot. However, after training agents with lr 1e-3 for 5M steps or so, agents in certain environments like InvertedPendulum will become unstable. Conclusion is that we should start with a large learning rate and linearly decay it, but for a small initial learning rate or if you only train agents for limited timesteps, DO NOT decay it.
5. We didn't tune `step-per-collect` option and `training-num` option. Default values are finetuned with PPO algorithm so we assume they are also good for REINFORCE. You can play with them if you want, but remember that `buffer-size` should always be larger than `step-per-collect`, and if `step-per-collect` is too small and `training-num` too large, episodes will be truncated and bootstrapped very often, which will harm performances. If `training-num` is too small (e.g., less than 8), speed will go down.
6. Sigma of action is not fixed (normally seen in other implementation) or conditioned on observation, but is an independent parameter which can be updated by gradient descent. We choose this setting because it works well in PPO, and is recommended by [Andrychowicz, Marcin, et al](https://arxiv.org/abs/2006.05990). See Fig. 23.

### A2C

|      Environment       | Tianshou(3M steps) | [Spinning Up(Pytorch)](https://spinningup.openai.com/en/latest/spinningup/bench_vpg.html)|
| :--------------------: | :----------------: | :--------------------: |
|          Ant           | **5236.8+-236.7**  |           ~5           |
|      HalfCheetah       | **2377.3+-1363.7** |          ~600          |
|         Hopper         | **1608.6+-529.5**  |          ~800          |
|        Walker2d        | **1805.4+-1055.9** |          ~460          |
|        Swimmer         |     40.2+-1.8      |        **~51**         |
|        Humanoid        | **5316.6+-554.8**  |           N            |
|        Reacher         |   **-5.2+-0.5**    |           N            |
|    InvertedPendulum    |  **1000.0+-0.0**   |           N            |
| InvertedDoublePendulum |  **9351.3+-12.8**  |           N            |

|      Environment       |      Tianshou      | [PPO paper](https://arxiv.org/abs/1707.06347) A2C | [PPO paper](https://arxiv.org/abs/1707.06347) A2C + Trust Region |
| :--------------------: | :----------------: | :-------------: | :-------------: |
|          Ant           | **3485.4+-433.1**  |        N        |        N        |
|      HalfCheetah       | **1829.9+-1068.3** |      ~1000      |      ~930       |
|         Hopper         | **1253.2+-458.0**  |      ~900       |      ~1220      |
|        Walker2d        | **1091.6+-709.2**  |      ~850       |      ~700       |
|        Swimmer         |   **36.6+-2.1**    |       ~31       |     **~36**     |
|        Humanoid        | **1726.0+-1070.1** |        N        |        N        |
|        Reacher         |   **-6.7+-2.3**    |      ~-24       |      ~-27       |
|    InvertedPendulum    |  **1000.0+-0.0**   |    **~1000**    |    **~1000**    |
| InvertedDoublePendulum | **9257.7+-277.4**  |      ~7100      |      ~8100      |

\* details<sup>[[5]](#footnote5)</sup><sup>[[6]](#footnote6)</sup>

#### Hints for A2C

0. We choose `clip` action method in A2C instead `tanh` option as used in REINFORCE simply to be consistent with original implementation. `tanh` may be better or equally well but we didn't try.
1. (Initial) learning rate, lr decay, and `step-per-collect`, `training-num` affect the performance of A2C to a great extend. These 4 hyperparameters also affect each other and should be tuned together. We have done full scale ablation studies on these 4 hyperparameters (more than 800 agents trained), below are our findings.
2. `step-per-collect`/`training-num` = `bootstrap-lenghth`, which is max length of an "episode" used in GAE estimator, 80/16=5 in default settings. When `bootstrap-lenghth` is small, (maybe) because GAE can at most looks forward 5 steps, and use bootstrap strategy very often, the critic is less well-trained, so they actor cannot converge to very high scores. However, if we increase `step-per-collect` to increase `bootstrap-lenghth` (e.g. 256/16=16), actor/critic will be updated less often, so sample efficiency is low, which will make training process slow. To conclude, If you don't restrict env timesteps, you can try to use larger `bootstrap-lenghth`, and train for more steps, which perhaps will give you better converged scores. Train slower, achieve higher.
3. 7e-4 learning rate with decay strategy if proper for `step-per-collect=80`, `training-num=16`, but if you use larger `step-per-collect`(e.g. 256 - 2048), 7e-4 `lr` is a little bit small, because now you have more data and less noise for each update, and will be more confidence if taking larger steps; so higher learning rate(e.g. 1e-3) is more appropriate and usually boost performance in this setting. If plotting results arises fast in early stages and become unstable later, consider lr decay before decreasing lr.
4. `max-grad-norm` doesn't really help in our experiments, we simply keep it for consistency with other open-source implementations (e.g. SB3).
5. We original paper of A3C use RMSprop optimizer, we find that Adam with the same learning rate works equally well. We use RMSprop anyway. Again, for consistency.
6. We notice that in SB3's implementation of A2C that set `gae-lambda` to 1 by default, we don't know why and after doing some experiments, results show 0.95 is better overall.
7. We find out that `step-per-collect=256`, `training-num=8` are also good hyperparameters. You can have a try.

## Note

<a name="footnote1">[1]</a>  Supported environments include HalfCheetah-v3, Hopper-v3, Swimmer-v3, Walker2d-v3, Ant-v3, Humanoid-v3, Reacher-v2, InvertedPendulum-v2 and InvertedDoublePendulum-v2. Pusher, Thrower, Striker and HumanoidStandup are not supported because they are not commonly seen in literatures.

<a name="footnote2">[2]</a>  Pretrained agents, detailed graphs (single agent, single game) and log details can all be found [here](https://cloud.tsinghua.edu.cn/d/356e0f5d1e66426b9828/).

<a name="footnote3">[3]</a>  We used the latest version of all mujoco environments in gym (0.17.3 with mujoco==2.0.2.13), but it's not often the case with other benchmarks. Please check for details yourself in the original paper. (Different version's outcomes are usually similar, though)

<a name="footnote4">[4]</a>  We didn't compare offpolicy algorithms to OpenAI baselines [benchmark](https://github.com/openai/baselines/blob/master/benchmarks_mujoco1M.htm), because for now it seems that they haven't provided benchmark for offpolicy algorithms, but in [SpinningUp docs](https://spinningup.openai.com/en/latest/spinningup/bench.html) they stated that "SpinningUp implementations of DDPG, TD3, and SAC are roughly at-parity with the best-reported results for these algorithms", so we think lack of comparisons with OpenAI baselines is okay.

<a name="footnote5">[5]</a>  ~ means the number is approximated from the graph because accurate numbers is not provided in the paper. N means graphs not provided.

<a name="footnote6">[6]</a>  Reward metric: The meaning of the table value is the max average return over 10 trails (different seeds) ± a single standard deviation over trails. Each trial is averaged on another 10 test seeds. Only the first 1M steps data will be considered, if not otherwise stated. The shaded region on the graph also represents a single standard deviation. It is the same as [TD3 evaluation method](https://github.com/sfujim/TD3/issues/34).

<a name="footnote7">[7]</a>  In TD3 paper, shaded region represents only half of standard deviation.

<a name="footnote8">[8]</a>  SAC's start-timesteps is set to 10000 by default while it is 25000 is DDPG/TD3. TD3's learning rate is set to 3e-4 while it is 1e-3 for DDPG/SAC. However, there is NO enough evidence to support our choice of such hyperparameters (we simply choose them because of SpinningUp) and you can try playing with those hyperparameters to see if you can improve performance. Do tell us if you can!

<a name="footnote9">[9]</a>  We use batchsize of 256 in DDPG/TD3/SAC while SpinningUp use 100. Minor difference also lies with `start-timesteps`, data loop method `step_per_collect`, method to deal with/bootstrap truncated steps because of timelimit and unfinished/collecting episodes (contribute to performance improvement), etc.

<a name="footnote10">[10]</a>  Comparing Tianshou's REINFORCE algorithm with SpinningUp's VPG is quite unfair because SpinningUp's VPG uses a generative advantage estimator (GAE) which requires a dnn value predictor (critic network), which makes so called "VPG" more like A2C (advantage actor critic) algorithm. Even so, you can see that we are roughly at-parity with each other even if tianshou's REINFORCE do not use a critic or GAE.
