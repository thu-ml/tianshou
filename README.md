# tianshou
Tianshou(天授) is a reinforcement learning platform. The following image illustrate its architecture.

<img src="https://github.com/sproblvem/tianshou/blob/master/docs/figures/tianshou_architecture.png" height="200"/>

## agent
&nbsp;&nbsp;&nbsp;&nbsp;Examples

&nbsp;&nbsp;&nbsp;&nbsp;Self-play Framework

## core

### Policy Wrapper
&nbsp;&nbsp;&nbsp;&nbsp;Stochastic policies (OnehotCategorical, Gaussian), deterministic policies (policy as in DQN, DDPG)

&nbsp;&nbsp;&nbsp;&nbsp;Specific network architectures in original paper of DQN, TRPO, A3C, etc. Policy-Value Network of AlphaGo Zero

### Algorithm

#### losses
&nbsp;&nbsp;&nbsp;&nbsp;policy gradient (and its variants), DQN (and its variants), DDPG, TRPO, PPO

#### optimizer
&nbsp;&nbsp;&nbsp;&nbsp;TRPO, natural gradient (and TensorFlow optimizers (sgd, adam))

### Planning
&nbsp;&nbsp;&nbsp;&nbsp;MCTS

## data
&nbsp;&nbsp;&nbsp;&nbsp;Training style - Batch, Replay (and its variants)

&nbsp;&nbsp;&nbsp;&nbsp;Advantage Estimation Function

&nbsp;&nbsp;&nbsp;&nbsp;Multithread Read/Write

## environment
&nbsp;&nbsp;&nbsp;&nbsp;DQN repeat frames, Reward Reshaping, image preprocessing (not sure where)

## simulator
&nbsp;&nbsp;&nbsp;&nbsp;Go, Othello/Reversi, Warzone

<img src="https://github.com/sproblvem/tianshou/blob/master/docs/figures/go.png" height="150"/> <img src="https://github.com/sproblvem/tianshou/blob/master/docs/figures/reversi.jpg" height="150"/> <img src="https://github.com/sproblvem/tianshou/blob/master/docs/figures/warzone.jpg" height="150"/>

## TODO
Search based method parallel.

`Please Write comments.`

`Please do not use abbreviations unless others can know it well. (e.g. adv can short for advantage/adversarial, please use the full name instead)`

`Please name the module formally. (e.g. use more lower case "_", I think a module called "Batch" seems terrible)`

YongRen: Policy Wrapper, in order of Gaussian, DQN and DDPG

TongzhengRen: losses, in order of ppo, pg, DQN, DDPG with management of placeholders

YouQiaoben: data/Batch, implement num_timesteps, fix memory growth in num_episodes; adv_estimate.gae_lambda (need to write a value network in tf)

ShihongSong: data/Replay; then adv_estimate.dqn after YongRen's DQN

HaoshengZou: collaborate mainly on Policy and losses; interfaces and architecture

Note: install openai/gym first to run the Atari environment; note that interfaces between modules may not be finalized; the management of placeholders and `feed_dict` may have to be done manually for the time being;

Without preprocessing and other tricks, this example will not train to any meaningful results. Codes should past two tests: individual module test and run through this example code.