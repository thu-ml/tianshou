# Core Abstractions

Tianshou's architecture is built around a number of key abstractions that work together to provide a modular and flexible reinforcement learning framework. 
This document describes the conceptual foundation and functionality of each abstraction, helping you understand how they interact to enable RL agent training.

Knowing these abstractions is primarily relevant when using the procedural API – and particularly when implementing one's own learning algoriithms.

## Algorithm

The **{class}`~tianshou.algorithm.algorithm_base.Algorithm`** is the central abstraction representing the core of a reinforcement learning method (such as DQN, PPO, or SAC). 
It implements the key steps within the {ref}`learning process <structuring-the-process>`, containing a {ref}`policy` and defining how to update it from experience data.

Since an Algorithm contains neural networks and manages their training, the class inherits from `torch.nn.Module`. 

### Core Responsibilities

An Algorithm implements the details of an {ref}`update step <structuring-the-process>`:

1. **preprocessing**: Before the actual update begins, the algorithm prepares the training data. 
   This includes computing derived quantities that depend on temporal sequences, such as n-step returns, GAE advantages, or terminal state handling. 
   The {meth}`~tianshou.algorithm.algorithm_base.Algorithm._preprocess_batch` method handles this phase, often leveraging static methods like 
   {meth}`~tianshou.algorithm.algorithm_base.Algorithm.compute_nstep_return` and 
   {meth}`~tianshou.algorithm.algorithm_base.Algorithm.compute_episodic_return` to 
   efficiently compute returns using the buffer's temporal structure.

2. **network update**: The algorithm performs the actual neural network updates based on its specific learning method. 
   Each algorithm implements its own {meth}`~tianshou.algorithm.algorithm_base.Algorithm._update_with_batch` logic that defines how to update 
   the policy networks using the preprocessed batch data.

3. **postprocessing**: After the update, the algorithm may perform cleanup operations, such as updating prioritized replay buffer weights or other 
   algorithm-specific bookkeeping.

### Learning Orchestration

The Algorithm orchestrates the {ref}`update step <structuring-the-process>` through its 
{meth}`~tianshou.algorithm.algorithm_base.Algorithm.update` method, which ensures these three phases execute in proper sequence. 
It also manages optimizer state and learning rate schedulers, making them available for state persistence through
{meth}`~tianshou.algorithm.algorithm_base.Algorithm.state_dict` and
{meth}`~tianshou.algorithm.algorithm_base.Algorithm.load_state_dict` methods.

Each algorithm type (on-policy, off-policy, offline) creates its appropriate trainer through the
{meth}`~tianshou.algorithm.algorithm_base.Algorithm.create_trainer` method, 
establishing the connection between the learning logic and the training loop.

(policy)=
## Policy

The **{class}`~tianshou.algorithm.algorithm_base.Policy`** represents the agent's decision-making component, i.e. the mapping from observations to actions. 
While the Algorithm defines how to learn, the Policy defines what is learned and how to act.

Like Algorithm, the class inherits from `torch.nn.Module`.

### States of Operation

A Policy operates in two main modes:

- **training mode**: During training, the policy may employ exploration strategies, sample from action distributions, or add noise to encourage discovery. 
  Training mode is further divided into:
  - *collecting state*: When gathering experience from environment interaction
  - *updating state*: When performing network updates during learning
  
- **testing/inference mode**: During evaluation, the policy typically acts deterministically or uses the mode of predicted distributions to showcase 
  learned behavior without exploration.

The flag `is_within_training_step` controls the collection strategy, distinguishing between training and inference behavior.

### Key Methods

The Policy provides several essential methods:

- **{meth}`~tianshou.algorithm.algorithm_base.Policy.forward`**: 
  The core computation method that processes batched observations to produce action distributions or Q-values. 
  It takes a batch of environment data and optional hidden state (for recurrent policies), 
  returning a batch containing at minimum the "act" key, 
  and potentially "state" (hidden state) and "policy" (intermediate results to be stored in the buffer).

- **{meth}`~tianshou.algorithm.algorithm_base.Policy.compute_action`**: 
  A convenient method for inference that takes a single observation and returns a concrete action suitable for the environment. 
  This method internally calls `forward` with proper batching and unbatching.

- **{meth}`~tianshou.algorithm.algorithm_base.Policy.map_action`**: Transforms the raw neural network output to the environment's action space format, handling any necessary scaling or discretization.

The separation between `forward` (which works with batches) and 
{meth}`~tianshou.algorithm.algorithm_base.Policy.compute_action` (which works with single observations) provides efficiency 
during training and convenience during inference.

## Collector

The class **{class}`~tianshou.data.Collector`** bridges the gap between the policy and the environment(s), 
managing the process of gathering experience data. 
It enables efficient interaction with both single environments and vectorized environments (multiple parallel environments).

### Data Collection

The Collector's primary method, {meth}`~tianshou.data.Collector.collect`, orchestrates the environment interaction loop. It can collect either:
- a specified number of steps (`n_step`): useful for maintaining consistent training batch sizes
- a specified number of episodes (`n_episode`): useful for evaluation or when episode-level statistics are important

During collection, the Collector ...
1. obtains observations from the environment(s),
2. calls the policy to compute actions,
3. steps the environment(s) with these actions,
4. stores the resulting transitions (observation, action, reward, next observation, termination flags, and info) in the replay buffer,
5. manages episode boundaries and reset logic,
6. collects statistics such as episode returns, lengths, and collection speed.

### Hooks and Extensibility

The Collector supports customization through hooks that can be triggered at different points in the collection process:
- **step hooks**: called after each environment step
- **episode done hooks**: called when episodes complete

These hooks enable custom logging, curriculum learning, or other dynamic behaviors during data collection.

### Vectorized Environments

The Collector seamlessly handles vectorized environments, where multiple environment instances run in parallel. 
This significantly speeds up data collection while maintaining correct episode boundaries and statistics for each environment instance.

## Trainer

The **{class}`~tianshou.trainer.Trainer`** orchestrates the complete training loop, coordinating data collection, policy updates, and evaluation. 
It provides the high-level control flow that brings all components together.

### Trainer Types

Tianshou provides three main trainer types, each suited to different algorithm families:

- **{class}`~tianshou.trainer.OnPolicyTrainer`**: for algorithms that must learn from freshly collected data (e.g., PPO, A2C). 
  After each collection phase, the buffer is used for updates and thereafter is cleared.

- **{class}`~tianshou.trainer.OffPolicyTrainer`**: for algorithms that can learn from any past experience (e.g., DQN, SAC, DDPG).
  Data accumulates in the replay buffer over time, and updates sample from this growing pool of experience.

- **{class}`~tianshou.trainer.OfflineTrainer`**: for algorithms that learn exclusively from a fixed dataset without any environment interaction (e.g., BCQ, CQL).

### Training Loop Structure

The training process is organized into epochs, where each epoch consists of:

1. **data collection**: The trainer uses the train collector to gather experience according to its algorithm type's needs
2. **policy update**: The algorithm performs one or more update steps using the collected data
3. **evaluation**: Periodically, the trainer uses the test collector to evaluate the current policy's performance
4. **logging**: Statistics from collection, updates, and evaluation are logged
5. **checkpointing**: The best policy (according to a scoring function) is saved

The trainer handles the detailed choreography of these steps, including determining when to collect more data, 
how many update steps to perform, when to evaluate, and when to stop training (based on maximum epochs, timesteps, or early stopping criteria).

### Configuration

Trainers are configured through parameter dataclasses 
({class}`~tianshou.trainer.OnPolicyTrainerParams`, {class}`~tianshou.trainer.OffPolicyTrainerParams`, {class}`~tianshou.trainer.OfflineTrainerParams`) 
that specify in particular:
- training duration (number of epochs, steps per epoch)
- collectors for training and testing
- update frequency and batch size
- evaluation frequency
- logging and checkpointing settings
- early stopping criteria

## Batch

The class **{class}`~tianshou.data.Batch`** is Tianshou's flexible data structure for passing information between components. 
It serves as the lingua franca of the framework, carrying everything from raw environment observations to computed returns and policy outputs.

### Design Philosophy

Batch is designed to be ...
- **flexible**: can contain any key-value pairs, with nested structures supported,
- **numpy/torch-compatible**: automatically converts lists to arrays and seamlessly works with both NumPy arrays and PyTorch tensors,
- **sliceable**: supports indexing and slicing operations that work across all contained data,
- **composable**: can be concatenated, stacked, and split to support batching operations.

### Common Use Cases

Batches flow through the system carrying different types of information:

1. **environment data**: observations, rewards, done flags, and info from environment steps
2. **policy outputs**: actions, hidden states, and intermediate computations
3. **training data**: returns, advantages, and other computed quantities needed for learning
4. **sampling results**: batches sampled from the replay buffer for training

### Operations

Key operations on batches include:
- **attribute access**: dot notation (`batch.obs`) or dictionary-style access (`batch['obs']`)
- **slicing**: extract subsets with standard indexing (`batch[0:10]`, `batch[[1,3,5]]`)
- **stacking**: combine multiple batches along a new dimension
- **type conversion**: convert between NumPy and PyTorch with `to_numpy()` and `to_torch()`
- **null handling**: detect and remove null values with `hasnull()`, `isnull()`, and `dropnull()`

The first dimension of all data in a Batch represents the batch size, enabling vectorized operations.

## Buffer

A **buffer** (i.e. class {class}`~tianshou.data.buffer.ReplayBuffer` and its variants) manages the storage and retrieval of experience data. 
It acts as the memory of the learning system, preserving the temporal structure of episodes while providing efficient access patterns.

### Storage Structure

Buffers store data in a circular queue fashion with a fixed maximum size. When the buffer fills, new data overwrites the oldest stored experiences. 
All data is stored within a single underlying Batch object, with the buffer managing:
- **pointer tracking**: current insertion position
- **episode boundaries**: which transitions belong to which episodes
- **temporal relationships**: the sequential order of transitions

### Reserved Keys

Buffers use a standard set of keys for storing transitions:
- `obs`: Observation at time t
- `act`: Action taken at time t
- `rew`: Reward received at time t
- `terminated`: True if the episode ended naturally at time t
- `truncated`: True if the episode was cut off at time t (e.g., time limit)
- `done`: Automatically inferred as `terminated or truncated`
- `obs_next`: Observation at time t+1
- `info`: Additional information from the environment
- `policy`: Intermediate policy computations to be stored

### Core Operations

**adding data**: The {meth}`~tianshou.data.buffer.ReplayBuffer.add` method stores new transitions, 
automatically handling episode boundaries and computing episode statistics (return, length) 
when episodes complete.

**sampling**: The {meth}`~tianshou.data.buffer.ReplayBuffer.sample` method retrieves batches of experiences for training, 
returning both the sampled batch and the corresponding indices. 
The sample size can be specified, or set to 0 to retrieve all available data.

**temporal navigation**: The {meth}`~tianshou.data.buffer.ReplayBuffer.prev` and {meth}`~tianshou.data.buffer.ReplayBuffer.next` 
methods enable traversal along the temporal sequence, respecting episode boundaries. 
This is essential for computing n-step returns and other time-dependent quantities.

**persistence**: Buffers support saving and loading via pickle or HDF5 format, enabling dataset collection and offline learning.

### Buffer Variants

Tianshou provides specialized buffer types:

- **{class}`~tianshou.data.buffer.ReplayBuffer`**: the standard buffer for single environments
- **{class}`~tianshou.data.buffer.VectorReplayBuffer`**: manages separate sub-buffers for multiple parallel environments while maintaining chronological order
- **{class}`~tianshou.data.buffer.PrioritizedReplayBuffer`**: samples transitions based on their TD-error or other priority metrics, using an efficient segment tree implementation

### Advanced Features

Buffers support sophisticated use cases:
- **frame stacking**: automatically stacks consecutive observations (useful for RNN inputs or Atari)
- **memory optimization**: option to skip storing next observations (useful for Atari where they can be inferred)
- **multi-modal observations**: handle observations with multiple components (e.g., image + vector)

## Logger

The **{class}`~tianshou.utils.logger.BaseLogger`** abstraction provides a unified interface for recording and tracking training progress, metrics, and statistics. 
It decouples the training loop from the specifics of where and how data is logged.

### Purpose

Loggers serve several essential functions:
- **progress tracking**: record timesteps, episodes, and epochs as training progresses
- **metric collection**: store performance indicators like rewards, losses, and success rates
- **experiment organization**: manage different data scopes (training, testing, updating)
- **reproducibility**: save training curves and hyperparameters for later analysis

### Logging Scopes

The framework organizes logged data into distinct scopes:
- **train data**: metrics from the training collector (episode returns, steps, collection speed)
- **test data**: evaluation metrics from the test collector
- **update data**: learning statistics from the algorithm (losses, gradients, learning rates)
- **info data**: additional custom metrics or metadata

Each scope has a corresponding log method (`log_train_data`, `log_test_data`, `log_update_data`, `log_info_data`) that the trainer calls at appropriate times.

### Implementations

Tianshou provides several logger implementations:
- **{class}`~tianshou.utils.logger.TensorboardLogger`**: writes to TensorBoard format for visualization with TensorBoard
- **{class}`~tianshou.utils.logger.WandbLogger`**: integrates with Weights & Biases for cloud-based experiment tracking
- **{class}`~tianshou.utils.logger.BasicLogger`**: a simple logger that prints to the console or writes to a file

All implementations inherit from {class}`~tianshou.utils.logger.BaseLogger` and share a common interface, 
making it easy to switch between logging backends or use multiple loggers simultaneously.

### Data Preparation

Before writing, loggers prepare data through the `prepare_dict_for_logging` method, which can filter, transform, or aggregate metrics. 
The `write` method then persists the prepared data to the logging backend with an associated step count.

## How They Work Together

These seven abstractions collaborate to enable reinforcement learning:

1. The **Trainer** initializes and orchestrates the training process.
2. The **Collector** uses the **Policy** to gather experience from environments.
3. Collected transitions are stored in the **Buffer** extracted as **Batches**.
4. The **Algorithm** samples from the **Buffer**, preprocesses the data, and updates the **Policy**.
5. The **Logger** records metrics throughout the process.
6. The cycle repeats until training completes.

This modular design allows each component to focus on its specific responsibility while maintaining clean interfaces. 
You can customize individual components (e.g., implementing a new Algorithm or Buffer) without affecting the others, 
making Tianshou both powerful and flexible.
