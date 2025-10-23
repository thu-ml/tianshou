# Tianshou's Dual API Architecture

Tianshou provides two distinct APIs to serve different use cases and user preferences:

1. **High-Level API**: A declarative, configuration-based interface designed for ease of use
2. **Procedural API**: A flexible, imperative interface providing maximum control

Both APIs access the same underlying algorithm implementations, allowing you to choose the level 
of abstraction that best fits your needs without sacrificing functionality.

## Overview

### High-Level API

The high-level API is built around the **builder pattern** and **declarative semantics**. 
Instead of writing procedural code that sequentially constructs and connects components, 
you declare _what_ you want through configuration objects and let Tianshou handle _how_ to 
build and execute the experiment.

**Key characteristics:**
- Centered around `ExperimentBuilder` classes (e.g., `DQNExperimentBuilder`, `PPOExperimentBuilder`, etc.)
- Uses configuration dataclasses and factories for all relevant parameters
- Automatically handles component creation and "wiring"
- Provides sensible defaults that adapt to the nature of your environment
- Includes built-in persistence, logging, and experiment management
- Excellent IDE support with auto-completion

### Procedural API

The procedural API provides explicit control over every component in the RL pipeline. 
You manually create environments, networks, policies, algorithms, collectors, and 
trainers, then wire them together.

**Key characteristics:**
- Direct instantiation of all components
- Explicit control over the training loop
- Lower-level access to internal mechanisms
- Minimal abstraction (closer to the implementation)
- Ideal for algorithm development and research

## When to Use Which API

### Use the High-Level API when:

- **You're applying existing algorithms** to new problems
- **You want to get started quickly** with minimal boilerplate
- **You need experiment management** with persistence, logging, and reproducibility
- **You prefer declarative code** that focuses on configuration
- **You're building applications** rather than developing new algorithms
- **You want strong IDE support** with auto-completion and type hints

### Use the Procedural API when:

- **You're developing new algorithms** or modifying existing ones
- **You need fine-grained control** over the training process
- **You want to understand** the internal workings of Tianshou
- **You're implementing custom components** not supported by the high-level API
- **You prefer imperative programming** where each step is explicit
- **You need maximum flexibility** for experimental research

## Comparison by Example

Let's compare both APIs by implementing the same DQN learning task on the CartPole environment.

### High-Level API Example

```python
from tianshou.highlevel.config import OffPolicyTrainingConfig
from tianshou.highlevel.env import EnvFactoryRegistered, VectorEnvType
from tianshou.highlevel.experiment import DQNExperimentBuilder, ExperimentConfig
from tianshou.highlevel.params.algorithm_params import DQNParams
from tianshou.highlevel.trainer import EpochStopCallbackRewardThreshold

# Build the experiment through configuration
experiment = (
    DQNExperimentBuilder(
        # Environment configuration
        EnvFactoryRegistered(
            task="CartPole-v1",
            venv_type=VectorEnvType.DUMMY,
            train_seed=0,
            test_seed=10,
        ),
        # Experiment settings
        ExperimentConfig(
            persistence_enabled=False,
            watch=True,
            watch_render=1 / 35,
            watch_num_episodes=100,
        ),
        # Training configuration
        OffPolicyTrainingConfig(
            max_epochs=10,
            epoch_num_steps=10000,
            batch_size=64,
            num_train_envs=10,
            num_test_envs=100,
            buffer_size=20000,
            collection_step_num_env_steps=10,
            update_step_num_gradient_steps_per_sample=1 / 10,
        ),
    )
    # Algorithm-specific parameters
    .with_dqn_params(
        DQNParams(
            lr=1e-3,
            gamma=0.9,
            n_step_return_horizon=3,
            target_update_freq=320,
            eps_training=0.3,
            eps_inference=0.0,
        ),
    )
    # Network architecture
    .with_model_factory_default(hidden_sizes=(64, 64))
    # Stop condition
    .with_epoch_stop_callback(EpochStopCallbackRewardThreshold(195))
    .build()
)

# Run the experiment
experiment.run()
```

**What's happening here:**
1. We create an `ExperimentBuilder` with three main configuration objects
2. We chain builder methods to specify algorithm parameters, model architecture, and callbacks
3. We call `.build()` to construct the experiment
4. We call `.run()` to execute the entire training pipeline

The high-level API handles:
- Creating and configuring environments
- Building the neural network
- Instantiating the policy and algorithm
- Setting up collectors and replay buffer
- Managing the training loop
- Watching the trained agent

### Procedural API Example

```python
import gymnasium as gym
import tianshou as ts
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import CollectStats
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo
from torch.utils.tensorboard import SummaryWriter

# Define hyperparameters
task = "CartPole-v1"
lr, epoch, batch_size = 1e-3, 10, 64
num_train_envs, num_test_envs = 10, 100
gamma, n_step, target_freq = 0.9, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
epoch_num_steps, collection_step_num_env_steps = 10000, 10

# Set up logging
logger = ts.utils.TensorboardLogger(SummaryWriter("log/dqn"))

# Create environments
train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(num_train_envs)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)])

# Build the network
env = gym.make(task, render_mode="human")
space_info = SpaceInfo.from_env(env)
state_shape = space_info.observation_info.obs_shape
action_shape = space_info.action_info.action_shape
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])

# Create policy and algorithm
policy = DiscreteQLearningPolicy(
    model=net,
    action_space=env.action_space,
    eps_training=eps_train,
    eps_inference=eps_test,
)
algorithm = ts.algorithm.DQN(
    policy=policy,
    optim=AdamOptimizerFactory(lr=lr),
    gamma=gamma,
    n_step_return_horizon=n_step,
    target_update_freq=target_freq,
)

# Set up collectors
train_collector = ts.data.Collector[CollectStats](
    algorithm,
    train_envs,
    ts.data.VectorReplayBuffer(buffer_size, num_train_envs),
    exploration_noise=True,
)
test_collector = ts.data.Collector[CollectStats](
    algorithm,
    test_envs,
    exploration_noise=True,
)

# Define stop condition
def stop_fn(mean_rewards: float) -> bool:
    if env.spec and env.spec.reward_threshold:
        return mean_rewards >= env.spec.reward_threshold
    return False

# Train the algorithm
result = algorithm.run_training(
    OffPolicyTrainerParams(
        train_collector=train_collector,
        test_collector=test_collector,
        max_epochs=epoch,
        epoch_num_steps=epoch_num_steps,
        collection_step_num_env_steps=collection_step_num_env_steps,
        test_step_num_episodes=num_test_envs,
        batch_size=batch_size,
        update_step_num_gradient_steps_per_sample=1 / collection_step_num_env_steps,
        stop_fn=stop_fn,
        logger=logger,
        test_in_train=True,
    )
)
print(f"Finished training in {result.timing.total_time} seconds")

# Watch the trained agent
collector = ts.data.Collector[CollectStats](algorithm, env, exploration_noise=True)
collector.collect(n_episode=100, render=1 / 35)
```

**What's happening here:**
1. We explicitly define all hyperparameters as variables
2. We manually create the logger
3. We construct training and test environments
4. We build the neural network by extracting space information from the environment
5. We create the policy and algorithm objects
6. We set up collectors with a replay buffer
7. We define callback functions
8. We call `algorithm.run_training()` with explicit parameters
9. We manually set up and run the evaluation collector

The procedural API requires:
- Explicit creation of every component
- Manual extraction of environment properties
- Direct specification of all connections
- Custom callback function definitions

## Key Concepts in the High-Level API

### ExperimentBuilder

The `ExperimentBuilder` is the core abstraction. 
Each algorithm has its own builder (e.g., `DQNExperimentBuilder`, `PPOExperimentBuilder`, `SACExperimentBuilder`).

**Some methods you will find in experiment builders:**
- `.with_<algorithm>_params()` - Set algorithm-specific parameters
- `.with_model_factory()`, `.with_model_factory_default()` - Configure network architecture
- `.with_critic_factory()` - Configure critic network (for actor-critic methods)
- `.with_epoch_train_callback()` - Add function to be called at the beginning of the training step in each epoch
- `.with_epoch_test_callback()` - Add function to be called at the beginning of the test step in each epoch
- `.with_epoch_stop_callback()` - Define stopping conditions
- `.with_algorithm_wrapper_factory()` - Add algorithm wrappers (e.g., ICM)

### Configuration Objects

Three main configuration objects are required when constructing an experiment builder:

1. **Environment Configuration** (`EnvFactory` subclasses)
   - Defines how to create and configure environments
   - Existing factories:
     - `EnvFactoryRegistered` - For the creation of environments registered in Gymnasium
     - `AtariEnvFactory` - For Atari environments with preprocessing
   - Custom factories for your own environments can be created by subclassing `EnvFactory`

2. **Experiment Configuration** (`ExperimentConfig`): 
   General settings for the experiment, particularly related to 
   - logging
   - randomization
   - persistence
   - watching the trained agent's performance after training

3. **Training Configuration** (`OffPolicyTrainingConfig`, `OnPolicyTrainingConfig`): 
   Defines all parameters related to the training process

### Parameter Classes

Algorithm parameters are defined in dataclasses specific to each algorithm (e.g., `DQNParams`, `PPOParams`).
The parameters are extensively documented.

```{note}
Make sure to use a modern IDE to take advantage of auto-completion and inline documentation!
```

### Factories

The high-level API uses factories extensively:
- **Model Factories**: Create neural networks (e.g., `IntermediateModuleFactoryAtariDQN()`)
- **Environment Factories**: Create and configure environments
- **Optimizer Factories**: Create optimizers with specific configurations

### Extensibility

The high-level API is designed to be extensible. 
You can create custom factories (e.g. for your own models or your own environments) by subclassing the appropriate base classes
and then use them in the experiment builder.

If we have created a torch module in `CustomNetwork`, which we want to use within our policy, 
we simply need to define a factory for it in order to apply it in the high-level API:

```python
from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.core import TDevice
from tianshou.highlevel.module.intermediate import IntermediateModuleFactory, IntermediateModule

class CustomNetFactory(IntermediateModuleFactory):
    def __init__(self, hidden_sizes: tuple[int, ...] = (128, 128)):
        self.hidden_sizes = hidden_sizes
    
    def create_intermediate_module(self, envs: Environments, device: TDevice) -> IntermediateModule:
        obs_shape = envs.get_observation_shape()
        action_shape = envs.get_action_shape()
        
        # Your custom network creation logic
        net = CustomNetwork(
            obs_shape=obs_shape,
            action_shape=action_shape,
            hidden_sizes=self.hidden_sizes,
        ).to(device)
        
        return IntermediateModule(net, net.output_dim)

experiment = (
    DQNExperimentBuilder(...)
    .with_model_factory(CustomNetFactory(hidden_sizes=(256, 256)))
    .build()
)
```

## Key Concepts in the Procedural API

### Core Components

You manually create and connect:

1. **Environments**: Using `gym.make()` and vectorization (`DummyVectorEnv`, `SubprocVectorEnv`)
2. **Networks**: Using `Net` or custom PyTorch modules
3. **Policies**: Using algorithm-specific policy classes (e.g., `DiscreteQLearningPolicy`)
4. **Algorithms**: Using algorithm classes (e.g., `DQN`, `PPO`, `SAC`)
5. **Collectors**: Using `Collector` to gather experience
6. **Buffers**: Using `VectorReplayBuffer` or `ReplayBuffer`
7. **Trainers**: Using the respective trainer class and corresponding parameter class (e.g., `OffPolicyTrainer` and `OffPolicyTrainerParams`)

### Training Loop

The training is executed via `algorithm.run_training()`, which takes a trainer parameter object. 
You can alternatively implement custom training loops (or even your own trainer class) for maximum flexibility.


## Choosing Your Path

**Use the high-level API** if ...
- you are new to Tianshou,
- you are focused on applying RL to problems,
- you prefer declarative code.

**Use the procedural API** if ...
- you are developing new algorithms,
- you need maximum flexibility,
- you are comfortable with RL internals,
- you prefer imperative code.

## Additional Resources

- **High-Level API Examples**: See `examples/` directory (scripts ending in `_hl.py`)
- **Procedural API Examples**: See `examples/` directory (scripts without suffix)
