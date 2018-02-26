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

#### brief intro of current implementation:

how to write your own network:
- define the observation placeholder yourself, pass it to `observation_placeholder` when initializing a policy instance
- pass a callable when initializing a policy instance. The callable should satisfy only three conditions:
    - it accepts no parameters
    - it does not create any new placeholders
    - it returns `action-related tensors, value_head`

Our lib will take care of your observation placeholder from now on, as well as
all the placeholders that will be created by our lib.

The other placeholders, such as `keep_prob` in dropout and `clip_param` in ppo loss
should be managed by your own (see examples/ppo_cartpole_alternative.py)

The `weight_update` parameter:
- 0 means manually update target network
- 1 means no target network (the target network is updated every 1 minibatch)
- (0, 1) is the target network as used in DDPG
- greater than 1 is the target network as used in DQN

Other comments are in the python files in example/ and in the lib codes.
Refactor is definitely needed so don't dwell too much on annoying details...

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

## examples

During development, run examples under `./examples/` directory with, e.g. `python ppo_example.py`.
Running them under this directory with `python examples/ppo_example.py` will not work.


## About coding style

Please follow [google python coding style](https://google.github.io/styleguide/pyguide.html)

There's a more detailed Chinese version [google python coding style in Chinese](http://www.runoob.com/w3cnote/google-python-styleguide.html)

All files/folders should be named with lower case letters and underline (except specified names such as `AlphaGo`).

Try to use full names. Don't use abbrevations for class/function/variable names except common abbrevations (such as `num` for number, `dim` for dimension, `env` for environment, `op` for operation). For now we use `pi` to refer to the policy in examples/ppo_example.py.

The """xxx""" comment should be written right after class/function. Also comment the part that's not intuitive during the code. We must comment, but for now we don't need to polish them.

# High Priority TODO

For Haosheng and Tongzheng: separate actor and critic, rewrite the interfaces for policy

Others can still focus on the task below.

## TODO
Search based method parallel.

YongRen: Policy Wrapper, in order of Gaussian, DQN and DDPG

TongzhengRen: losses, in order of ppo, pg, DQN, DDPG with management of placeholders

YouQiaoben: data/Batch, implement num_timesteps, fix memory growth in num_episodes; adv_estimate.gae_lambda (need to write a value network in tf)

ShihongSong: data/Replay; then adv_estimate.dqn after YongRen's DQN

HaoshengZou: collaborate mainly on Policy and losses; interfaces and architecture

Note: install openai/gym first to run the Atari environment; note that interfaces between modules may not be finalized; the management of placeholders and `feed_dict` may have to be done manually for the time being;

Without preprocessing and other tricks, this example will not train to any meaningful results. Codes should past two tests: individual module test and run through this example code.

## Dependencies
TensorFlow (Version >= 1.4)

gym
